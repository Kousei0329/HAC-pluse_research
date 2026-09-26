"""
Standalone loader: loads an already-trained model (from a completed train.py run's
saved point_cloud/checkpoint.pth at a given iteration), applies ONE post-training MLP
quantization variant, and reports the resulting total transmitted size (via
estimate_final_bits(), unchanged) plus PSNR/SSIM/LPIPS on the test cameras.

Exists so the MLP-quantization-width ablation (fp32/fp16/fp8/fp4/int8/int4, across every
lambda) doesn't need a full ~1-2h retrain per variant -- quantization is a post-training,
in-place operation applied only at the very end of a normal run, so this reruns just that
tail end against a checkpoint that was already trained once with no quantization flags set.
"""
import os
import sys
import json
import re
import torch
import lpips
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import prefilter_voxel, render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--plane_fusion", type=str, default='concat', choices=['concat', 'hadamard', 'sum'])
    parser.add_argument('--use_level_gate', action='store_true', default=True)
    parser.add_argument("--use_gated_mlp", action='store_true', default=False)
    parser.add_argument("--use_spatial_context", action='store_true', default=False)
    parser.add_argument('--use_joint_context', action='store_true', default=False)
    parser.add_argument('--use_anchor_cond_norm', action='store_true', default=True)
    parser.add_argument('--no_use_anchor_cond_norm', dest='use_anchor_cond_norm', action='store_false')
    parser.add_argument('--use_causal_knn', action='store_true', default=True)
    parser.add_argument('--no_use_causal_knn', dest='use_causal_knn', action='store_false')
    parser.add_argument('--causal_knn_K', type=int, default=16)
    parser.add_argument('--causal_knn_hidden_mult', type=int, default=8)
    parser.add_argument('--mlp_grid_hidden_mult', type=int, default=8)
    parser.add_argument('--use_3gmm', action='store_true', default=False)
    parser.add_argument("--use_reno", action='store_true', default=True)
    parser.add_argument("--no_use_reno", dest='use_reno', action='store_false')
    parser.add_argument("--reno_ckpt_path", type=str, default="submodules/reno/model/Ford/ckpt.pt")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--variant", type=str, required=True,
                         choices=['fp32', 'fp16', 'fp8', 'fp4', 'int8', 'int6', 'int4', 'int2'])
    parser.add_argument("--fp8_variant", type=str, default='e4m3', choices=['e4m3', 'e5m2'])
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        plane_fusion=args.plane_fusion,
        use_level_gate=args.use_level_gate,
        is_synthetic_nerf=is_synthetic_nerf,
        use_gated_mlp=args.use_gated_mlp,
        use_spatial_context=args.use_spatial_context,
        use_joint_context=args.use_joint_context,
        use_anchor_cond_norm=args.use_anchor_cond_norm,
        use_causal_knn=args.use_causal_knn,
        causal_knn_K=args.causal_knn_K,
        causal_knn_hidden_mult=args.causal_knn_hidden_mult,
        mlp_grid_hidden_mult=args.mlp_grid_hidden_mult,
        use_3gmm=args.use_3gmm,
        use_reno=args.use_reno,
        reno_ckpt_path=args.reno_ckpt_path,
        # training_report() runs conduct_encoding()+conduct_decoding() (when run_codec=True,
        # the module-level default in train.py) before scene.save() -- the saved point_cloud.ply
        # holds the POST-decode state, where get_scaling/get_mask/get_anchor must skip their
        # normal activation (decoded values are already in activated space; re-applying exp() to
        # an already-exp'd scale explodes catastrophically, e.g. 0.07 -> exp(0.07)=1.07 typically
        # but exp(30)=1e13 for a large scale, exactly what was observed). render_sets() already
        # gets this right (decoded_version=run_codec); this script needs the same.
        decoded_version=True,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.eval()
    # x_bound_min/max are the hash grid's coordinate-normalization range, fixed ONCE from the
    # original (pre-densification) point cloud right when training starts (train.py calls
    # update_anchor_bound() exactly once, before the training loop) and never recomputed
    # afterward -- recomputing it here from the final, densified anchor set gives a different
    # (usually much larger) spatial extent, desyncing every hash-grid lookup from what the model
    # was actually trained against and producing garbage (observed: values exploding to 1e8+
    # with NaNs cascading from there). conduct_encoding() pickles the actual training-time bound
    # into bitstreams/ for exactly this reason (conduct_decoding() reloads it the same way) --
    # reuse that instead of update_anchor_bound().
    x_bound_path = os.path.join(dataset.model_path, "bitstreams")
    x_bound_min_path = os.path.join(x_bound_path, "x_bound_min.pkl")
    x_bound_max_path = os.path.join(x_bound_path, "x_bound_max.pkl")
    if os.path.exists(x_bound_min_path) and os.path.exists(x_bound_max_path):
        gaussians.x_bound_min = torch.load(x_bound_min_path).cuda()
        gaussians.x_bound_max = torch.load(x_bound_max_path).cuda()
    else:
        raise FileNotFoundError(
            f"{x_bound_min_path} / {x_bound_max_path} not found -- these are only written when "
            f"training ran with run_codec=True (conduct_encoding/_decoding). Without them, "
            f"hash-grid coordinate normalization can't match what the model was trained with.")

    with torch.no_grad():
        if args.variant == 'fp8':
            gaussians.quantize_mlps_fp8_(fp8_variant=args.fp8_variant)
        elif args.variant == 'fp4':
            gaussians.quantize_mlps_fp4_()
        elif args.variant == 'fp16':
            gaussians.quantize_mlps_fp16_()
        elif args.variant.startswith('int'):
            bits = int(args.variant[3:])
            gaussians.quantize_mlps_(bits=bits)
        # fp32: no quantization, model stays exactly as trained.

        log_info = gaussians.estimate_final_bits()
        print(log_info)
        mlp_size_bits, mlp_size_MB = gaussians.get_mlp_size()

        m = re.search(r"Total ([\d.]+)", log_info)
        total_MB = float(m.group(1)) if m else None

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
        cameras = scene.getTestCameras()
        l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        for viewpoint in cameras:
            voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, pipe, background)
            render_output = render(viewpoint, gaussians, pipe, background, visible_mask=voxel_visible_mask)
            image = torch.clamp(render_output["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            l1_sum += l1_loss(image, gt_image).mean().item()
            psnr_sum += psnr(image, gt_image).mean().item()
            ssim_sum += ssim(image, gt_image).mean().item()
            lpips_sum += lpips_fn(image, gt_image, normalize=False).detach().mean().item()
        n = max(1, len(cameras))

        result = {
            'variant': args.variant,
            'iteration': args.iteration,
            'mlp_size_MB': mlp_size_MB,
            'total_size_MB': total_MB,
            'psnr': psnr_sum / n,
            'ssim': ssim_sum / n,
            'lpips': lpips_sum / n,
            'l1': l1_sum / n,
            'n_test_cameras': n,
            'estimate_final_bits_log': log_info,
        }
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
