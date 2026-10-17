import os
import re
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Post-hoc quantization-variant evaluation for the clean-fp32 checkpoints trained by
# run_shrunk_causalknn_bungee_mip360.py. Loads each already-trained checkpoint once per variant
# via eval_quant_variant.py (no retraining, no in-place weight modification of the checkpoint on
# disk) and reports PSNR/SSIM/LPIPS/size for 'fp32' (baseline, no quantization) and 'int8' (this
# study's best-performing width on truck/train/drjohnson/playroom/hotdog so far).
EXPERIMENT_DIR = "outputs/experiments/shrunk_causalknn_bungee_mip360_2026-10-15_07-09-07"
CAUSAL_KNN_K = 8
CAUSAL_KNN_HIDDEN_MULT = 2
MLP_GRID_HIDDEN_MULT = 4
GPU_ID = int(os.environ.get('GPU_ID', 0))
VARIANTS = os.environ.get('VARIANTS', 'fp32,int8').split(',')
PYTHON_BIN = "/opt/conda/envs/HAC_env_pytorch2/bin/python"

LAMBDAS = [0.001, 0.005]
DATASET_CONFIGS = {
    'bungeenerf': {
        'scenes': ['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome'],
        'data_base': 'data/bungeenerf',
        'voxel_size': 0,
        'update_init_factor': 128,
        'lod': 30,
    },
    'mipnerf360': {
        'scenes': ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai'],
        'data_base': 'data/mipnerf360',
        'voxel_size': 0.001,
        'update_init_factor': 16,
        'lod': 0,
    },
}
ACTIVE_DATASETS = ['bungeenerf', 'mipnerf360']


def resolve_active_scenes():
    pairs = []
    for dataset_name in ACTIVE_DATASETS:
        for scene in DATASET_CONFIGS[dataset_name]['scenes']:
            pairs.append((dataset_name, scene))
    return pairs


def extract_test_fps(training_log_path):
    """The LAST 'Test FPS' line in training.log is from render_sets() on the post-decode
    (decoded_version=True) checkpoint -- the same state eval_quant_variant.py loads. MLP
    quantization doesn't touch the rasterization path, so this one FPS number is valid for
    every variant; no need to re-measure it per variant."""
    if not os.path.exists(training_log_path):
        return None
    last = None
    with open(training_log_path, errors='replace') as f:
        for line in f:
            # The final (post-decode) line wraps the number in an ANSI color code
            # (\x1b[1;35m...\x1b[0m); a lazy .*? before the digits matches the "1" inside that
            # escape sequence instead, so skip one optional ANSI sequence explicitly.
            m = re.search(r'Test FPS:\s*(?:\x1b\[[\d;]*m)?([\d.]+)', line)
            if m:
                last = float(m.group(1))
    return last


def create_command(scene, dataset_name, lmbda, variant, model_path, out_json):
    cfg = DATASET_CONFIGS[dataset_name]
    data_path = f"{cfg['data_base']}/{scene}"
    cmd = (
        f'CUDA_VISIBLE_DEVICES={GPU_ID} {PYTHON_BIN} eval_quant_variant.py '
        f'-s {data_path} --eval '
        f'--lod {cfg["lod"]} --voxel_size {cfg["voxel_size"]} --update_init_factor {cfg["update_init_factor"]} '
        f'--causal_knn_K {CAUSAL_KNN_K} '
        f'--causal_knn_hidden_mult {CAUSAL_KNN_HIDDEN_MULT} '
        f'--mlp_grid_hidden_mult {MLP_GRID_HIDDEN_MULT} '
        f'-m {model_path} --iteration 30000 '
        f'--variant {variant} --out_json {out_json}'
    )
    return cmd


def run_one(scene, dataset_name, lmbda, variant, exp_num, total):
    model_path = f"{EXPERIMENT_DIR}/{dataset_name}/{scene}/lambda_{lmbda}"
    out_json = f"{model_path}/quant_eval_{variant}.json"
    log_file = f"{model_path}/quant_eval_{variant}.log"
    cmd = create_command(scene, dataset_name, lmbda, variant, model_path, out_json)

    print(f"\n{'='*80}\n[{exp_num}/{total}] {dataset_name}/{scene}/lambda_{lmbda} variant={variant}\n{'='*80}\n")
    start_time = time.time()
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time
    print(f"Completed [{exp_num}/{total}] {dataset_name}/{scene}/lambda_{lmbda} variant={variant}  "
          f"Time: {elapsed/60:.2f}min  Return code: {result.returncode}")
    return result.returncode


def main():
    active_scenes = resolve_active_scenes()
    jobs = [
        (ds, sc, lmbda, variant)
        for ds, sc in active_scenes
        for lmbda in LAMBDAS
        for variant in VARIANTS
    ]
    total = len(jobs)
    print(f"Total eval jobs: {total} (variants={VARIANTS}, GPU={GPU_ID})")

    failures = []
    for i, (ds, sc, lmbda, variant) in enumerate(jobs, 1):
        model_path = f"{EXPERIMENT_DIR}/{ds}/{sc}/lambda_{lmbda}"
        out_json = f"{model_path}/quant_eval_{variant}.json"
        if os.path.exists(out_json):
            print(f"[{i}/{total}] {ds}/{sc}/lambda_{lmbda} variant={variant} already done, skipping")
            continue
        rc = run_one(sc, ds, lmbda, variant, i, total)
        if rc != 0:
            failures.append((ds, sc, lmbda, variant))

    print(f"\n{'='*80}\nEval run finished. Failures: {failures}\n{'='*80}\n")

    # Aggregate into one table-ready json: per (dataset, scene, lambda) -> {variant: metrics} + fps
    rows = []
    for ds, sc in active_scenes:
        for lmbda in LAMBDAS:
            model_path = f"{EXPERIMENT_DIR}/{ds}/{sc}/lambda_{lmbda}"
            fps = extract_test_fps(f"{model_path}/training.log")
            row = {'dataset': ds, 'scene': sc, 'lambda': lmbda, 'fps': fps}
            for variant in VARIANTS:
                out_json = f"{model_path}/quant_eval_{variant}.json"
                if os.path.exists(out_json):
                    with open(out_json) as f:
                        row[variant] = json.load(f)
            rows.append(row)

    agg_path = f"{EXPERIMENT_DIR}/quant_eval_aggregate_gpu{GPU_ID}.json"
    with open(agg_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Aggregate written to {agg_path}")


if __name__ == "__main__":
    main()
