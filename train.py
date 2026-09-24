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

import os
import math
import numpy as np

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel, HierarchicalGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import get_binary_vxl_size

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# from lpipsPyTorch import lpips

bit2MB_scale = 8 * 1024 * 1024
run_codec = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    # Set random seed for reproducibility
    set_random_seed(args_param.seed)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))

    # Choose model type based on use_hierarchical flag
    if dataset.use_hierarchical:
        print(f"Using HierarchicalGaussianModel with:")
        print(f"  level1_voxel_scale={dataset.level1_voxel_scale}")
        print(f"  level2_per_level1={dataset.level2_per_level1}")
        print(f"  intra_anchor_type={dataset.intra_anchor_type}")
        gaussians = HierarchicalGaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            plane_fusion=args_param.plane_fusion,
            use_level_gate=args_param.use_level_gate,
            is_synthetic_nerf=is_synthetic_nerf,
            use_hierarchical=dataset.use_hierarchical,
            level1_voxel_scale=dataset.level1_voxel_scale,
            level2_per_level1=dataset.level2_per_level1,
            intra_anchor_type=dataset.intra_anchor_type,
            mamba_hidden_dim=dataset.mamba_hidden_dim,
            mamba_d_state=dataset.mamba_d_state,
            mamba_d_conv=dataset.mamba_d_conv,
            mamba_n_layers=dataset.mamba_n_layers,
            use_gated_mlp=args_param.use_gated_mlp,
            use_spatial_context=args_param.use_spatial_context,
            use_joint_context=args_param.use_joint_context,
        )
    else:
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            plane_fusion=args_param.plane_fusion,
            use_level_gate=args_param.use_level_gate,
            is_synthetic_nerf=is_synthetic_nerf,
            use_gated_mlp=args_param.use_gated_mlp,
            use_spatial_context=args_param.use_spatial_context,
            use_joint_context=args_param.use_joint_context,
            use_anchor_cond_norm=args_param.use_anchor_cond_norm,
            use_causal_knn=args_param.use_causal_knn,
            causal_knn_K=args_param.causal_knn_K,
            causal_knn_hidden_mult=args_param.causal_knn_hidden_mult,
            mlp_grid_hidden_mult=args_param.mlp_grid_hidden_mult,
            use_3gmm=args_param.use_3gmm,
            use_reno=args_param.use_reno,
            reno_ckpt_path=args_param.reno_ckpt_path,
        )
    scene = Scene(dataset, gaussians, ply_path=ply_path)
    gaussians.update_anchor_bound()

    gaussians.training_setup(opt)

    reno_optimizer = None
    _reno_avg_bpp = None  # last RENO bpp measurement, used as a frozen per-anchor rate price in the main loss
    _reno_avg_bpp_ema = None  # EMA-smoothed price actually charged in the rate loss (see below)
    # Half-life kept at ~34 updates so the smoothing window stays ~700 iterations regardless of
    # reno_train_interval: at interval=100 that was 0.9 (~7 updates); at the new default interval=20
    # it needs a slower per-update decay (~34 updates) to cover the same iteration span, otherwise
    # more frequent updates would erode the stability this EMA was added for.
    _RENO_EMA_DECAY = 0.5 ** (1.0 / (700.0 / args_param.reno_train_interval))
    if args_param.use_reno and args_param.train_reno:
        from utils.reno_utils import get_reno_net
        _reno_net = get_reno_net(args_param.reno_ckpt_path)
        reno_optimizer = torch.optim.Adam(_reno_net.parameters(), lr=args_param.reno_lr)
        logger.info(f"[RENO] Fine-tuning enabled: interval={args_param.reno_train_interval}, lr={args_param.reno_lr}")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    for iteration in range(first_iter, opt.iterations + 1):


        if iteration == 10:
            print("==== MASK-RELATED PARAMS ====")
            for name, p in gaussians.named_parameters():
                if "mask" in name.lower():
                    print(f"{name}: shape={p.shape}, requires_grad={p.requires_grad}")
            # network gui not available in scaffold-gs yet


        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

        if iteration % 1000 == 0 and bit_per_param is not None:

            # 全anchorを使ってビット数を正確に計算
            with torch.no_grad():
                ttl_bits_feat, ttl_bits_scaling, ttl_bits_offsets = gaussians.compute_total_bits()
                ttl_size_feat_MB = ttl_bits_feat / bit2MB_scale
                ttl_size_scaling_MB = ttl_bits_scaling / bit2MB_scale
                ttl_size_offsets_MB = ttl_bits_offsets / bit2MB_scale
                ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

                # visible anchorからの推定値も参考として計算
                ttl_size_feat_MB_approx = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
                ttl_size_scaling_MB_approx = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
                ttl_size_offsets_MB_approx = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale

            logger.info("\n----------------------------------------------------------------------------------------")
            logger.info("\n-----[ITER {}] bits info (accurate): ttl_size_feat_MB={}, ttl_size_scaling_MB={}, ttl_size_offsets_MB={}, ttl_size_MB={}-----".format(
                iteration, ttl_size_feat_MB, ttl_size_scaling_MB, ttl_size_offsets_MB, ttl_size_MB))
            logger.info("\n-----[ITER {}] bits info (approx from visible): bit_per_feat_param={}, ttl_size_feat_MB_approx={}-----".format(
                iteration, bit_per_feat_param.item(), ttl_size_feat_MB_approx))
            logger.info("\n-----[ITER {}] bits info: anchor_num={}-----".format(iteration, gaussians.get_anchor.shape[0]))
            with torch.no_grad():
                binary_grid_masks_anchor = gaussians.get_mask_anchor.float()
                mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks_anchor + 0.0)  # [0, 1] -> [-1, 1]
            logger.info("\n-----[ITER {}] bits info: 1_rate_mask={}, mask_numel={}, mask_size_MB={}-----".format(iteration, mask_1_rate, mask_numel, mask_size_MB))

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        # scaling_reg = scaling.prod(dim=1).mean()
        scaling_reg = (scaling[:, 0] * scaling[:, 1] * scaling[:, 2]).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            loss = loss + args_param.lmbda * (bit_per_param + bit_hash_grid / denom)

            if reno_optimizer is not None and _reno_avg_bpp_ema is not None:
                # RENO's bpp isn't end-to-end differentiable w.r.t. which anchors are kept: the
                # vendored network derives occupancy purely from hard integer coordinates
                # (submodules/reno/network.py forward() discards the `feats` field entirely), so
                # there's no gradient path from "drop this anchor" to "bpp goes down". Instead we
                # treat the last empirically measured RENO bpp as a frozen price-per-anchor and
                # charge it against the differentiable (STE) anchor count, so lambda gives the mask
                # a direct incentive to prune anchors whose position-coding cost isn't earning its
                # keep, calibrated to RENO's actual measured compression efficiency.
                #
                # The raw per-update bpp measurement is noisy (single mini-batch of anchors,
                # network mid-finetune), and charging it directly against the mask can spike the
                # prune pressure for ~100 iterations at a time; anchors dropped while densification
                # is still running (before update_until) never come back, so a bad spike can carve
                # a permanent hole out of the scene. Use an EMA of the measured bpp instead so the
                # price the mask sees moves smoothly, while RENO itself still trains on the raw signal.
                anchor_count_soft = gaussians.get_mask_anchor.sum()
                loss = loss + args_param.lmbda * (_reno_avg_bpp_ema * anchor_count_soft / denom)

                if iteration % 1000 == 0:
                    logger.info(f"\n-----[ITER {iteration}] RENO anchor rate term: avg_bpp_per_anchor={_reno_avg_bpp_ema:.4f} (raw={_reno_avg_bpp:.4f}), anchor_count_soft={anchor_count_soft.item():.1f}-----")

        loss.backward()

        # ★ ここからデバッグ
        if iteration in [1000, 5000, 10000, 20000]:
            print("==== MASK GRAD DEBUG ====")
            for name, p in gaussians.named_parameters():
                if "mask" in name.lower():
                    mean_val = p.data.mean().item()
                    grad_none = (p.grad is None)
                    grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
                    print(f"{name}: mean={mean_val:.4f}, grad_none={grad_none}, grad_norm={grad_norm:.4e}")

            # 直接アクセス版（名前が分かっているので）
            if hasattr(gaussians, "_mask_level1"):
                p = gaussians._mask_level1
                m = p.data
                g = p.grad
                print(f"_mask_level1 sigmoid mean={m.sigmoid().mean().item():.4f}, "
                      f"grad_none={g is None}, grad_norm={(g.norm().item() if g is not None else 0.0):.4e}")

            if hasattr(gaussians, "_mask"):
                p = gaussians._mask
                m = p.data
                g = p.grad
                print(f"_mask sigmoid mean={m.sigmoid().mean().item():.4f}, "
                      f"grad_none={g is None}, grad_norm={(g.norm().item() if g is not None else 0.0):.4e}")
        # ★ ここまで




        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize(); t_start_log = time.time()
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, args_param.model_path, quantize_mlp_bits=args_param.quantize_mlp_bits, quantize_mlp_fp16=args_param.quantize_mlp_fp16, quantize_mlp_fp8=args_param.quantize_mlp_fp8, fp8_variant=args_param.fp8_variant, prune_mlp_ratio=args_param.prune_mlp_ratio, prune_finetune_iters=args_param.prune_finetune_iters, prune_finetune_lr=args_param.prune_finetune_lr, prune_finetune_lambda_q=args_param.prune_finetune_lambda_q)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                # Level 1の統計情報も追跡（階層的モデルの場合）
                if hasattr(gaussians, 'training_statis_level1'):
                    gaussians.training_statis_level1(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                if iteration not in range(3000, 4000):  # let the model get fit to quantization
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom

                # Level 1の統計情報も削除（階層的モデルの場合）
                if hasattr(gaussians, 'opacity_accum_level1'):
                    del gaussians.opacity_accum_level1
                    del gaussians.offset_gradient_accum_level1
                    del gaussians.offset_denom_level1
                    del gaussians.anchor_demon_level1

                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                # No gradient clipping existed anywhere in this codebase before; the enlarged
                # mlp_grid/mlp_deform/causal_knn (2x width, +1 layer) diverged to NaN almost
                # immediately once the rate loss engages at step>10000 without it. This is a
                # general safety net, not tuned to any one config, so it stays on unconditionally.
                grad_norm = torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    gaussians.optimizer.step()
                else:
                    # clip_grad_norm_ scales every gradient by max_norm/total_norm -- if even a
                    # handful of entries are already NaN/Inf, total_norm (and thus the scale
                    # factor applied to ALL parameters) becomes NaN too, turning a localized
                    # blowup into instant, total divergence. Skipping the step avoids making it
                    # worse, but a skip alone can't recover: if a *parameter* (not just its
                    # gradient) already went NaN on some earlier step, every later forward pass
                    # re-derives a NaN loss from that same value forever, so grad_norm stays NaN
                    # and every step keeps getting skipped with no way out. Sanitizing the
                    # parameters themselves back to finite values breaks that loop.
                    _n_bad = 0
                    with torch.no_grad():
                        for _p in gaussians.parameters():
                            _bad = ~torch.isfinite(_p.data)
                            if _bad.any():
                                _n_bad += _bad.sum().item()
                                _p.data[_bad] = 0.0
                    logger.info(f"[WARNING] Non-finite gradient norm ({grad_norm.item()}) at iteration {iteration}; "
                                f"skipped optimizer step and zeroed {_n_bad} non-finite parameter entries to recover.")
                gaussians.optimizer.zero_grad(set_to_none = True)

            # RENO fine-tuning step
            if reno_optimizer is not None and iteration % args_param.reno_train_interval == 0:
                from utils.reno_utils import ensure_path as _ensure_reno_path
                _ensure_reno_path()
                from torchsparse import SparseTensor as _ST
                from torchsparse.nn import functional as _F
                _cfg = _F.conv_config.get_default_conv_config()
                _cfg.kmap_mode = "hashmap"
                _F.conv_config.set_global_conv_config(_cfg)

                with torch.no_grad():
                    # Match the anchor set actually used at encode time (estimate_final_bits /
                    # conduct_encoding both filter by get_mask_anchor): otherwise RENO is fine-tuned
                    # on ~20% more anchors than what conduct_encoding() will compress, and the bpp
                    # measured during training understates the real encoded bpp.
                    _mask_anchor = gaussians.get_mask_anchor.detach().to(torch.bool)[:, 0]
                    _anchor = gaussians.get_anchor[_mask_anchor].detach()

                # Every anchor's mask gate can be transiently closed at this exact iteration (e.g.
                # right after a heavy prune) -- nothing to fine-tune RENO on this step, and
                # _anchor_int.min(dim=0) on an empty tensor raises IndexError, so skip instead.
                if _anchor.shape[0] == 0:
                    logger.info(f"[RENO] WARNING: iter={iteration} all anchors masked out; skipping this RENO fine-tune step")
                else:
                    with torch.no_grad():
                        _anchor_int = torch.round(_anchor / gaussians.voxel_size).int()
                        _shift = _anchor_int.min(dim=0)[0]
                        _anchor_shifted = (_anchor_int - _shift)
                        _coords = torch.cat((_anchor_shifted[:, :1] * 0, _anchor_shifted), dim=-1).int()
                        _feats = torch.ones((_coords.shape[0], 1), device='cuda')

                        # Subsample to avoid OOM on large scenes; bpp estimate is still valid with a subset
                        _MAX_RENO_ANCHORS = 20_000
                        if _coords.shape[0] > _MAX_RENO_ANCHORS:
                            _sub_idx = torch.randperm(_coords.shape[0], device='cuda')[:_MAX_RENO_ANCHORS]
                            _coords = _coords[_sub_idx]
                            _feats = _feats[_sub_idx]

                    with torch.enable_grad():
                        _reno_net.train()
                        reno_optimizer.zero_grad()
                        _bpp = _reno_net(_ST(coords=_coords, feats=_feats))
                        _bpp.backward()
                    # RENO has its own optimizer, entirely separate from gaussians.optimizer -- the
                    # gradient clipping added for the main loss's NaN-divergence never covered this
                    # step. An unclipped bad RENO update can make _bpp (hence _reno_avg_bpp_ema, which
                    # feeds directly into the main anchor-mask rate loss below) NaN/Inf, which then
                    # poisons gaussians' own gradients on the very next main-loss backward() even
                    # though *those* are clipped -- clipping a NaN gradient still leaves it NaN.
                    _reno_grad_norm = torch.nn.utils.clip_grad_norm_(_reno_net.parameters(), max_norm=1.0)
                    if torch.isfinite(_reno_grad_norm):
                        reno_optimizer.step()
                    else:
                        logger.info(f"[RENO] WARNING: iter={iteration} non-finite grad norm ({_reno_grad_norm.item()}); skipping reno_optimizer step")
                    reno_optimizer.zero_grad(set_to_none=True)
                    _new_bpp = _bpp.item()
                    if math.isfinite(_new_bpp):
                        _reno_avg_bpp = _new_bpp  # frozen price-per-anchor for next iterations' rate loss
                        _reno_avg_bpp_ema = (_reno_avg_bpp if _reno_avg_bpp_ema is None else
                                              _RENO_EMA_DECAY * _reno_avg_bpp_ema + (1 - _RENO_EMA_DECAY) * _reno_avg_bpp)
                    else:
                        logger.info(f"[RENO] WARNING: iter={iteration} bpp is non-finite ({_new_bpp}); "
                                    f"keeping previous price ema={_reno_avg_bpp_ema}")

                    if iteration % (args_param.reno_train_interval * 10) == 0:
                        logger.info(f"[RENO] iter={iteration} bpp={_bpp.item():.4f} anchor_num={_coords.shape[0]}")

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))

    if reno_optimizer is not None:
        _reno_save_path = os.path.join(dataset.model_path, 'reno_adapted.pt')
        torch.save(_reno_net.state_dict(), _reno_save_path)
        logger.info(f"[RENO] Saved adapted weights to {_reno_save_path}")
        gaussians.reno_ckpt_path = _reno_save_path

    return gaussians.x_bound_min, gaussians.x_bound_max

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, pre_path_name='', quantize_mlp_bits=0, quantize_mlp_fp16=False, quantize_mlp_fp8=False, fp8_variant='e4m3', prune_mlp_ratio=0.0, prune_finetune_iters=300, prune_finetune_lr=1e-4, prune_finetune_lambda_q=1000.0):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()

        if 1:
            if iteration == testing_iterations[-1]:
                if prune_mlp_ratio > 0:
                    # Snapshot the full-shape, unpruned MLPs before pruning mutates them in place,
                    # so lambda_q_reg (or the prune ratio itself) can be swept offline afterward
                    # without repeating the ~2h main training loop each time.
                    scene.gaussians.save_mlp_checkpoints(os.path.join(pre_path_name, 'pre_prune_checkpoint.pth'))
                    # Structural pruning first (shrinks the actual matrices), then quantize the
                    # now-smaller network -- the two compound (fewer params x fewer bits each).
                    scene.gaussians.structured_prune_mlps_(ratio=prune_mlp_ratio)
                    if prune_finetune_iters > 0:
                        # Pruning's magnitude heuristic has no gradient signal and can leave the
                        # entropy heads badly miscalibrated for outlier anchors; recover with a
                        # brief rate-loss-only fine-tune before quantizing/reporting final sizes.
                        scene.gaussians.finetune_pruned_mlps_(iters=prune_finetune_iters, lr=prune_finetune_lr, lambda_q_reg=prune_finetune_lambda_q)
                if quantize_mlp_fp8:
                    # Quantize before estimate_final_bits/conduct_encoding/the render+eval loop
                    # below, so every number reported past this point (bitstream sizes AND
                    # PSNR/SSIM/LPIPS) consistently reflects the fp8 entropy MLPs, not a mix of
                    # fp32 compute with a fake smaller reported size.
                    scene.gaussians.quantize_mlps_fp8_(fp8_variant=fp8_variant)
                elif quantize_mlp_fp16:
                    scene.gaussians.quantize_mlps_fp16_()
                elif quantize_mlp_bits > 0:
                    scene.gaussians.quantize_mlps_(bits=quantize_mlp_bits)
                with torch.no_grad():
                    log_info = scene.gaussians.estimate_final_bits()
                    logger.info(log_info)
                if run_codec:  # conduct encoding and decoding
                    with torch.no_grad():
                        bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
                        # conduct encoding
                        log_info = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                        logger.info(log_info)
                        # conduct decoding
                        log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path)
                        logger.info(log_info)
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                # if config['name'] == 'test': assert len(config['cameras']) == 200
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0

                    if wandb is not None:
                        gt_image_list = []
                        render_image_list = []
                        errormap_list = []

                    t_list = []

                    for idx, viewpoint in enumerate(config['cameras']):
                        torch.cuda.synchronize(); t_start = time.time()
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                        render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                        image = torch.clamp(render_output["render"], 0.0, 1.0)
                        time_sub = render_output["time_sub"]
                        torch.cuda.synchronize(); t_end = time.time()
                        t_list.append(t_end - t_start - time_sub)

                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 30):
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                            if wandb:
                                render_image_list.append(image[None])
                                errormap_list.append((gt_image[None]-image[None]).abs())

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                if wandb:
                                    gt_image_list.append(gt_image[None])
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += lpips_fn(image, gt_image, normalize=False).detach().mean().double()
                        # lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                    test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                    logger.info(f'Test FPS: {test_fps.item():.5f}')
                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
                    if wandb is not None:
                        wandb.log({"test_fps": test_fps, })

                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if wandb is not None:
                        wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test}, f"ssim{ssim_test}", f"lpips{lpips_test}")

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    psnr_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        #
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    print('testing_float_psnr=:', sum(psnr_list) / len(psnr_list))

    return t_list, visible_count_list


def render_sets(args_param, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None, x_bound_min=None, x_bound_max=None):
    with torch.no_grad():
        is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            plane_fusion=args_param.plane_fusion,
            use_level_gate=args_param.use_level_gate,
            decoded_version=run_codec,
            is_synthetic_nerf=is_synthetic_nerf,
            use_gated_mlp=args_param.use_gated_mlp,
            use_3gmm=args_param.use_3gmm,
            use_reno=args_param.use_reno,
            reno_ckpt_path=args_param.reno_ckpt_path,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        if x_bound_min is not None:
            gaussians.x_bound_min = x_bound_min
            gaussians.x_bound_max = x_bound_max

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx], normalize=False).detach().mean().double())
            # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--lmbda", type=float, default = 0.001)

    parser.add_argument("--plane_fusion", type=str, default='concat', choices=['concat', 'hadamard', 'sum'],
                        help='How to combine the xy/xz/yz 2D hash-grid planes: '
                             '"concat" (original HAC++ behavior), '
                             '"hadamard" (elementwise product, K-Planes style), '
                             '"sum" (elementwise sum, EG3D style)')
    parser.add_argument('--use_level_gate', action='store_true', default=True,
                        help='Enable SE-Net style content-based gating across each hash grid\'s '
                             'resolution levels (xyz/xy/xz/yz independently) before concatenation, '
                             'conditioned on each level\'s own activation and known resolution')
    
    
    parser.add_argument("--use_gated_mlp", action='store_true', default=False, help='Use gated MLP for channel context')
    parser.add_argument("--use_spatial_context", action='store_true', default=False, help='Use spatial context module for hash features')
    parser.add_argument('--use_joint_context', action='store_true', default=False, help='Use joint context module for joint features')
    
    parser.add_argument('--use_anchor_cond_norm', action='store_true', default=True, help='Condition hash grid features on anchor scale/offset via FiLM normalization')
    parser.add_argument('--no_use_anchor_cond_norm', dest='use_anchor_cond_norm', action='store_false', help='Disable use_anchor_cond_norm (baseline/ablation)')
    parser.add_argument('--use_causal_knn', action='store_true', default=True, help='Enable causal K-NN context aggregation for hash grid features (Morton order)')
    parser.add_argument('--no_use_causal_knn', dest='use_causal_knn', action='store_false', help='Disable use_causal_knn (baseline/ablation)')
    parser.add_argument('--causal_knn_K', type=int, default=16, help='Number of causal (Morton-order, backward-only) neighbors aggregated per anchor by CausalKNNContext')
    parser.add_argument('--causal_knn_hidden_mult', type=int, default=8, help='Hidden-width multiplier (x hash_dim) for CausalKNNContext.correction. Must be even. Default 8 matches the original widened design (~484K params at hash_dim=96); lower (e.g. 2 or 4) trades capacity for a much smaller transmitted size now that get_mlp_size() counts causal_knn.')
    parser.add_argument('--mlp_grid_hidden_mult', type=int, default=8, help='Hidden-width multiplier (x feat_dim) for mlp_grid. Must be even. Default 8 matches the widened/deepened design (~245K params); lower (e.g. 4) trades capacity for a smaller transmitted size.')
    parser.add_argument('--use_3gmm', action='store_true', default=False, help="Use a 3-component Gaussian mixture (Entropy_gaussian_mix_prob_3) for feat's entropy model instead of the default 2-component mixture")
    parser.add_argument('--quantize_mlp_bits', type=int, default=0, help="If >0, post-training fake-quantize mlp_grid/mlp_deform weight matrices to this many bits before the final size/quality report (0 = disabled, keep fp32)")
    parser.add_argument('--quantize_mlp_fp16', action='store_true', default=False, help="Post-training round-trip mlp_grid/mlp_deform (and causal_knn, if enabled) parameters through true IEEE half precision (fp16) before the final size/quality report. Takes priority over --quantize_mlp_bits if both are set.")
    parser.add_argument('--quantize_mlp_fp8', action='store_true', default=False, help="Post-training round-trip mlp_grid/mlp_deform (and causal_knn, if enabled) parameters through true 8-bit float (fp8) before the final size/quality report. Takes priority over --quantize_mlp_fp16 and --quantize_mlp_bits if multiple are set.")
    parser.add_argument('--fp8_variant', type=str, default='e4m3', choices=['e4m3', 'e5m2'], help="Which 8-bit float format --quantize_mlp_fp8 uses: e4m3 (more mantissa, less range -- better for weight-scale tensors) or e5m2 (more range, less mantissa)")
    parser.add_argument('--prune_mlp_ratio', type=float, default=0.0, help="If >0, post-training structurally prune this fraction of GEGLU hidden units out of mlp_grid/mlp_deform (shrinks the actual matrices) before quantization/the final size/quality report (0 = disabled)")
    parser.add_argument('--prune_finetune_iters', type=int, default=300, help="Number of rate-loss-only fine-tuning steps applied to mlp_grid/mlp_deform right after structured pruning (0 = skip fine-tuning)")
    parser.add_argument('--prune_finetune_lr', type=float, default=1e-4, help="Learning rate for the post-prune fine-tuning steps")
    parser.add_argument('--prune_finetune_lambda_q', type=float, default=1000.0, help="Weight on the penalty keeping Q_scaling/Q_offsets/Q_feat close to their just-pruned values during fine-tuning (prevents the rate-only loss from inflating quantization step sizes at PSNR's expense)")
    parser.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--use_reno", action='store_true', default=True, help='Use RENO neural codec for anchor position compression instead of G-PCC')
    parser.add_argument("--no_use_reno", dest='use_reno', action='store_false', help='Disable use_reno, falling back to G-PCC (baseline/ablation)')
    parser.add_argument("--reno_ckpt_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'submodules', 'reno', 'model', 'Ford', 'ckpt.pt'),
                        help='Path to RENO checkpoint')
    parser.add_argument("--train_reno", action='store_true', default=True, help='Fine-tune RENO network weights during training')
    parser.add_argument("--no_train_reno", dest='train_reno', action='store_false', help='Disable train_reno (baseline/ablation)')
    parser.add_argument("--reno_train_interval", type=int, default=20, help='Update RENO weights every N iterations')
    parser.add_argument("--reno_lr", type=float, default=1e-4, help='Learning rate for RENO fine-tuning optimizer')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    '''try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')'''

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Generate port before setting seed (to avoid interference)
    args.port = np.random.randint(10000, 20000)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    x_bound_min, x_bound_max = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        x_bound_min, x_bound_max = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(args, lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger, x_bound_min=x_bound_min, x_bound_max=x_bound_max)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
