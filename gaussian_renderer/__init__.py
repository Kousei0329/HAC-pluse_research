#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from utils.gpcc_utils import calculate_morton_order


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]  # [N_vis, 10, 1]

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    # Q_scaling変更
    # Q_scaling = 0.1
    Q_scaling = 0.001
    Q_offsets = 0.2
    if is_training:
        if step > 3000 and step <= 10000:
            # Quantization noise - now deterministic with global seed
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == 10000:
            pc.update_anchor_bound()

        if step > 10000:

            mask_anchor_all = pc.get_mask_anchor.to(torch.bool)[:, 0]   # [N_total]
            all_anchor_msk  = pc.get_anchor[mask_anchor_all]             # [N_mask, 3]
            _aint = torch.round(all_anchor_msk / pc.voxel_size).long()
            _morton  = calculate_morton_order(_aint)
            _unsort  = torch.argsort(_morton)
            _anc_srt = all_anchor_msk[_morton]                           # [N_mask, 3] Morton順

            if pc.use_causal_knn:
                # codec との一致を優先: anchor_feat なしで集約（codec も anchor_feat を使わない）
                with torch.no_grad():
                    _hf_srt  = pc.calc_interp_feat(_anc_srt)
                    _ctx_srt = pc.causal_knn.aggregate_only(_hf_srt, _anc_srt)  # [N_mask, D]
                _ctx_msk = _ctx_srt[_unsort]

                _mask_idx = torch.full((mask_anchor_all.shape[0],), -1,
                                       dtype=torch.long, device=all_anchor_msk.device)
                _mask_idx[mask_anchor_all] = torch.arange(
                    mask_anchor_all.sum(), device=all_anchor_msk.device)

                _vis_mpos  = _mask_idx[visible_mask]
                _vis_inmsk = _vis_mpos >= 0
                _ctx_dim   = _ctx_msk.shape[-1]
                feat_ctx_render = torch.zeros(anchor.shape[0], _ctx_dim, device=anchor.device)
                if _vis_inmsk.any():
                    _vis_hf = pc.calc_interp_feat(anchor[_vis_inmsk])
                    _vis_ctx = _ctx_msk[_vis_mpos[_vis_inmsk]]
                    feat_ctx_render[_vis_inmsk] = pc.causal_knn.apply_fusion(_vis_hf, _vis_ctx)
            else:
                # causal_knn なし: visible アンカーのハッシュ特徴を直接使用
                # Unconditioned, matching the causal_knn branch above: conditioning this
                # noise-scale pre-pass on the anchor's own true feat value (as this used to do
                # via anchor_feat=feat[...]) is a self-referential loop -- the current feat value
                # would decide how much noise gets added back onto itself -- and empirically drove
                # Q_feat_adj to saturate at its ceiling (Q_feat -> ~2, very coarse quantization
                # noise), degrading PSNR/SSIM/LPIPS without a compensating rate benefit.
                _morton_vis = calculate_morton_order(torch.round(anchor / pc.voxel_size).long())
                _unsort_vis = torch.argsort(_morton_vis)
                feat_ctx_render = pc.calc_interp_feat(anchor[_morton_vis])[_unsort_vis]

            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                pc.forward_grid(feat_ctx_render)

            if step % 1000 == 0:
                print(f"[DEBUG Q_feat_prepass] step={step} Q_feat_adj: {Q_feat_adj.mean().item():.4f}/{Q_feat_adj.std().item():.4f} "
                      f"Q_feat: {(Q_feat*(1+torch.tanh(Q_feat_adj))).mean().item():.4f}")
                if pc.use_level_gate:
                    for gate_name in ['gate_xyz', 'gate_xy', 'gate_xz', 'gate_yz']:
                        g = getattr(pc.encoding_xyz, gate_name, None)
                        if g is not None and hasattr(g, '_last_gate_mean'):
                            gm = g._last_gate_mean.tolist()
                            gs = g._last_gate_std.tolist()
                            print(f"[DEBUG LevelSEGate {gate_name}] step={step} "
                                  f"per-level mean={['%.3f' % v for v in gm]} "
                                  f"std={['%.3f' % v for v in gs]} "
                                  f"(init was a uniform 0.881 for every level)")
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            # entropy estimation: mask_anchorの5%をサンプリング
            _choose = torch.rand(all_anchor_msk.shape[0], device=all_anchor_msk.device) <= 0.05
            anchor_chosen            = all_anchor_msk[_choose]
            feat_chosen              = pc._anchor_feat[mask_anchor_all][_choose]
            grid_offsets_chosen      = pc._offset[mask_anchor_all][_choose]
            grid_scaling_chosen      = pc.get_scaling[mask_anchor_all][_choose]
            binary_grid_masks_chosen = pc.get_mask[mask_anchor_all][_choose]
            mask_anchor_chosen       = pc.get_mask_anchor[mask_anchor_all][_choose]
            # Pruned (masked-off) offset slots are never trained against any rendering loss, so
            # their raw value is unconstrained garbage; zero them out before conditioning feat's
            # context on them, matching what the real decoder actually sees (0 for those slots).
            # Conditioning on the raw value here is a train/inference mismatch with
            # conduct_encoding()/conduct_decoding(), which always mask before conditioning.
            grid_offsets_chosen_masked = grid_offsets_chosen * binary_grid_masks_chosen.repeat(1, 1, 3)

            if pc.use_causal_knn:
                # Two-stage, matching the non-causal_knn branch below: stage 1 (unconditioned)
                # drives scaling/offset's own entropy params (nothing to condition on yet -- they
                # are decoded first); stage 2 FiLM-conditions this anchor's own hash feature on its
                # true scaling/offset before fusing with the *same* precomputed neighbor context
                # (_ctx_msk), so feat's entropy params can actually see scaling/offset. Reusing
                # _ctx_msk avoids re-running the expensive O(N*K) causal KNN search a second time --
                # only the cheap per-anchor calc_interp_feat + apply_fusion is duplicated, exactly
                # mirroring the cost the non-causal_knn branch already pays for its two calc_interp_feat calls.
                _chosen_ctx = _ctx_msk[_choose]
                _chosen_hf_no_cond = pc.calc_interp_feat(anchor_chosen)
                feat_context_no_cond = pc.causal_knn.apply_fusion(_chosen_hf_no_cond, _chosen_ctx)
                _chosen_hf_cond = pc.calc_interp_feat(anchor_chosen,
                                                       anchor_scale=grid_scaling_chosen,
                                                       anchor_offset=grid_offsets_chosen_masked.view(grid_offsets_chosen.shape[0], pc.n_offsets, 3))
                feat_context_orig = pc.causal_knn.apply_fusion(_chosen_hf_cond, _chosen_ctx)
            else:
                # two-stage: scaling/offset を先に計算し、feat は scaling/offset で条件付け
                feat_context_no_cond = pc.calc_interp_feat(anchor_chosen)
                feat_context_orig = pc.calc_interp_feat(anchor_chosen,
                                                        anchor_scale=grid_scaling_chosen,
                                                        anchor_offset=grid_offsets_chosen_masked.view(grid_offsets_chosen.shape[0], pc.n_offsets, 3))

            '''GMM使用'''
            _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj_no_cond, Q_scaling_adj, Q_offsets_adj = \
                pc.forward_grid(feat_context_no_cond)
            mean, scale, prob, _, _, _, _, Q_feat_adj_cond, _, _ = \
                pc.forward_grid(feat_context_orig)
            # Always read Q_feat_adj from the unconditioned stage-1 pass, never from the
            # FiLM-conditioned one. This was already required for causal_knn (its fusion module
            # collapsed Q_feat toward its floor when driven by a conditioned input). The
            # non-causal_knn branch was left on the conditioned source because it looked stable
            # historically, but that data predates the offsets-masking fix above -- with offsets
            # now correctly zeroed for pruned slots before conditioning, the conditioned context's
            # distribution changed, and empirically Q_feat_adj_cond now saturates toward its
            # ceiling here (Q_feat -> ~2, i.e. very coarse feat quantization): smaller feat bits
            # but visibly worse PSNR/SSIM/LPIPS. Q_feat_adj_cond is computed only for debug
            # comparison below.
            if abs(Q_feat_adj_no_cond.mean().item() - Q_feat_adj_cond.mean().item()) > 1e-6 and step % 1000 == 0:
                print(f"[DEBUG Q_feat_adj] step={step} no_cond: {Q_feat_adj_no_cond.mean().item():.4f}/{Q_feat_adj_no_cond.std().item():.4f} "
                      f"cond: {Q_feat_adj_cond.mean().item():.4f}/{Q_feat_adj_cond.std().item():.4f} "
                      f"Q_feat_if_cond: {(1*(1+torch.tanh(Q_feat_adj_cond))).mean().item():.4f}")
            Q_feat_adj = Q_feat_adj_no_cond

            Q_feat = 1
            # Q_scaling変更
            # Q_scaling = 0.1
            Q_scaling = 0.001
            Q_offsets = 0.2
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            # Quantization noise for entropy calculation - now deterministic with global seed
            feat_chosen = feat_chosen + torch.empty_like(feat_chosen).uniform_(-0.5, 0.5) * Q_feat
            mean_list, scale_list, probs_list = pc.get_feat_mixture(feat_chosen, mean, scale, prob)

            # Debug: NaN/Inf check before entropy calculation
            for dbg_name, dbg_t in [('mean', mean), ('scale', scale), ('mean_adj', mean_list[1]), ('scale_adj', scale_list[1]),
                                    ('feat_chosen', feat_chosen), ('Q_feat', Q_feat)]:
                if torch.is_tensor(dbg_t):
                    n_nan = torch.isnan(dbg_t).sum().item()
                    n_inf = torch.isinf(dbg_t).sum().item()
                    if n_nan > 0 or n_inf > 0:
                        print(f"[renderer] WARNING: {dbg_name} has {n_nan} NaN, {n_inf} Inf  (step={step})")

            grid_scaling_chosen = grid_scaling_chosen + torch.empty_like(grid_scaling_chosen).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets_chosen = grid_offsets_chosen + torch.empty_like(grid_offsets_chosen).uniform_(-0.5, 0.5) * Q_offsets
            grid_offsets_chosen = grid_offsets_chosen.view(-1, 3 * pc.n_offsets)

            binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).view(-1, 3*pc.n_offsets)

            EG = pc.EG_mix_prob_3 if pc.use_3gmm else pc.EG_mix_prob_2
            bit_feat = EG.forward(feat_chosen, *mean_list, *scale_list, *probs_list,
                                   Q=Q_feat, x_mean=pc._anchor_feat.mean())
            bit_feat = bit_feat * mask_anchor_chosen
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_scaling = bit_scaling * mask_anchor_chosen
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets.view(-1, 3*pc.n_offsets), pc._offset.mean())
            bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen

            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()
        anchor_int = torch.round(anchor / pc.voxel_size).long()
        morton_idx = calculate_morton_order(anchor_int)
        unsort_idx = torch.argsort(morton_idx)
        anchor_sorted = anchor[morton_idx]
        if pc.use_causal_knn:
            # Two-stage, mirroring the non-causal_knn branch below. The expensive O(N*K) neighbor
            # search (aggregate_only) runs once and is reused for both stages; only the cheap
            # per-anchor calc_interp_feat + apply_fusion is duplicated so stage 2 can actually see
            # the quantized scaling/offset via FiLM (AnchorCondNorm).
            hash_feats = pc.calc_interp_feat(anchor_sorted)
            causal_ctx = pc.causal_knn.aggregate_only(hash_feats, anchor_sorted)
            feat_context_no_cond = pc.causal_knn.apply_fusion(hash_feats, causal_ctx)[unsort_idx]
            # Q_feat_adj (the feat quantization-step multiplier) is deliberately read from the
            # *unconditioned* stage-1 context, not stage 2 below -- this is the same context
            # mlp_grid always used for it under causal_knn historically, so it stays in the regime
            # mlp_grid is actually calibrated for. Predicting it from the FiLM-conditioned stage-2
            # context (as the non-causal_knn branch does) let it collapse toward its floor early
            # in training here, exploding the quantized value range and OOMing the arithmetic coder.
            _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                pc.forward_grid(feat_context_no_cond)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
            grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()
            # Mask before conditioning only (not the shared grid_offsets, which downstream
            # rendering still needs raw for the offset_selection_mask path) -- see the training
            # branch above for why unmasked pruned slots corrupt feat's FiLM context.
            grid_offsets_for_cond = grid_offsets * binary_grid_masks.repeat(1, 1, 3)

            hash_feats_cond = pc.calc_interp_feat(anchor_sorted,
                                                   grid_scaling[morton_idx],
                                                   grid_offsets_for_cond[morton_idx])
            feat_context_cond = pc.causal_knn.apply_fusion(hash_feats_cond, causal_ctx)[unsort_idx]
            mean, scale, prob, _, _, _, _, _, _, _ = \
                pc.forward_grid(feat_context_cond)
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        else:
            # Stage 1: get Q_scaling/Q_offsets from unconditioned context
            feat_context_no_cond = pc.calc_interp_feat(anchor_sorted)[unsort_idx]
            _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                pc.forward_grid(feat_context_no_cond)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
            grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()
            grid_offsets_for_cond = grid_offsets * binary_grid_masks.repeat(1, 1, 3)
            # Stage 2: feat's mean/scale/prob still come from the conditioned context; Q_feat_adj
            # (captured above from stage 1) deliberately does not -- see the training branch above.
            feat_context_cond = pc.calc_interp_feat(anchor_sorted,
                                                     grid_scaling[morton_idx],
                                                     grid_offsets_for_cond[morton_idx])[unsort_idx]
            mean, scale, prob, _, _, _, _, _, _, _ = \
                pc.forward_grid(feat_context_cond)
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    binary_grid_masks_pergaussian = binary_grid_masks.view(-1, 1)
    if is_training:
        opacity = opacity * binary_grid_masks_pergaussian[mask]
        scaling = scaling * binary_grid_masks_pergaussian[mask]
    else:
        the_mask = (binary_grid_masks_pergaussian[mask]).to(torch.bool)
        the_mask = the_mask[:, 0]
        xyz = xyz[the_mask]
        color = color[the_mask]
        opacity = opacity[the_mask]
        scaling = scaling[the_mask]
        rot = rot[the_mask]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0
