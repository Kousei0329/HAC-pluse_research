import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# Follow-up to generalization_K8_fp16_reno_*: mlp_grid/mlp_deform were fp16-quantized but
# causal_knn's own weights were never even counted in get_mlp_size() (its param names don't
# contain 'mlp'), so its full fp32 cost was invisible and effectively "free" in every prior
# report. gaussian_model.py now counts causal_knn as entropy-side-info too and both
# quantize_mlps_fp16_/quantize_mlps_fp8_ round-trip it alongside mlp_grid/mlp_deform. This run
# validates that fix + checks whether fp8 (8 bits/param, half of fp16) claws back enough size to
# beat the plain-HAC baseline again, on a small/fast, medium, and large scene.
LAMBDAS = [0.001, 0.005]
CAUSAL_KNN_K = 8
GPU_ID = 0
EXPERIMENT_NAME = "causalknn_fp16_vs_fp8"
RANDOM_SEED = 42

DATASET_CONFIGS = {
    'tandt': {
        'scenes': ['truck'],
        'data_base': '/workspace/HAC/data/tandt',
        'voxel_size': 0.01,
        'mask_lr_final_base': 0.0001,
        'update_init_factor': 16,
        'lod': 0,
    },
    'nerf_synthetic': {
        'scenes': ['hotdog', 'ship'],
        'data_base': '/workspace/HAC/data/nerf_synthetic',
        'voxel_size': 0.001,
        'mask_lr_final_base': 0.00008,
        'update_init_factor': 4,
        'lod': 0,
    },
}

# 'fp16' = --quantize_mlp_fp16 (now covers causal_knn too), 'fp8' = --quantize_mlp_fp8 (e4m3)
VARIANTS = ['fp16', 'fp8']

ACTIVE_DATASETS = ['nerf_synthetic', 'tandt']

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def resolve_active_scenes():
    pairs = []
    for dataset_name in ACTIVE_DATASETS:
        for scene in DATASET_CONFIGS[dataset_name]['scenes']:
            pairs.append((dataset_name, scene))
    return pairs


def create_command(lmbda, scene, dataset_name, variant, gpu_id, seed):
    cfg = DATASET_CONFIGS[dataset_name]
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001

    data_path = f"{cfg['data_base']}/{scene}"
    output_path = f"{EXPERIMENT_DIR}/{dataset_name}/{scene}/{variant}/lambda_{lmbda}"

    quant_flag = '--quantize_mlp_fp16' if variant == 'fp16' else '--quantize_mlp_fp8'

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} /opt/conda/envs/HAC_plux_env/bin/python train.py '
        f'-s {data_path} '
        f'--eval '
        f'--lod {cfg["lod"]} '
        f'--voxel_size {cfg["voxel_size"]} '
        f'--update_init_factor {cfg["update_init_factor"]} '
        f'--iterations 30_000 '
        f'-m {output_path} '
        f'--lmbda {lmbda} '
        f'--mask_lr_final {mask_lr_final} '
        f'--seed {seed} '
        f'--causal_knn_K {CAUSAL_KNN_K} '
        f'{quant_flag} '
        # RENO stays on its default (True), same as generalization_K8_fp16_reno_*.
    )
    return cmd, output_path


def run_experiment(lmbda, scene, dataset_name, variant, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, scene, dataset_name, variant, gpu_id, seed)

    print(f"\n{'='*80}")
    print(f"Starting experiment {exp_num}/{total}:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Scene:   {scene}")
    print(f"  Variant: {variant}")
    print(f"  Lambda:  {lmbda}")
    print(f"  Output:  {output_path}")
    print(f"{'='*80}\n")

    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)

    start_time = time.time()
    print(f"Running command:\n{cmd}\n")

    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Completed experiment {exp_num}/{total}:")
    print(f"  Dataset: {dataset_name}, Scene: {scene}, Variant: {variant}, Lambda: {lmbda}")
    print(f"  Time: {elapsed/60:.2f} minutes")
    print(f"  Return code: {result.returncode}")
    print(f"  Log: {log_file}")
    print(f"{'='*80}\n")

    return {
        'dataset': dataset_name, 'scene': scene, 'variant': variant, 'lmbda': lmbda,
        'output_path': output_path, 'log_file': log_file,
        'duration_minutes': elapsed / 60, 'return_code': result.returncode,
    }


def main():
    active_scenes = resolve_active_scenes()

    experiments = [
        {'dataset': ds, 'scene': sc, 'variant': v, 'lmbda': lmbda}
        for ds, sc in active_scenes
        for v in VARIANTS
        for lmbda in LAMBDAS
    ]
    total = len(experiments)

    config = {
        "timestamp": TIMESTAMP, "lambdas": LAMBDAS, "causal_knn_K": CAUSAL_KNN_K,
        "variants": VARIANTS, "use_reno": True,
        "active_scenes": [{"dataset": ds, "scene": sc} for ds, sc in active_scenes],
        "gpu_id": GPU_ID, "experiment_name": EXPERIMENT_NAME, "random_seed": RANDOM_SEED,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"causal_knn fp16-vs-fp8 validation (causal_knn now counted + quantized)")
    print(f"{'='*80}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Lambdas: {LAMBDAS}, Variants: {VARIANTS}")
    print(f"Active scenes (in order): {active_scenes}")
    print(f"Total experiments: {total}")
    print(f"{'='*80}\n")

    completed = []
    for i, exp in enumerate(experiments, 1):
        result = run_experiment(
            exp['lmbda'], exp['scene'], exp['dataset'], exp['variant'],
            GPU_ID, RANDOM_SEED, i, total
        )
        completed.append(result)

    print(f"\n{'='*80}")
    print(f"All {len(completed)} experiments completed!")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")

    summary = {
        'experiment_dir': EXPERIMENT_DIR, 'config': config, 'completed_experiments': completed,
        'total_duration_minutes': sum(r['duration_minutes'] for r in completed),
    }
    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results summary: {EXPERIMENT_DIR}/results_summary.json")
    print(f"Total execution time: {summary['total_duration_minutes']:.2f} minutes\n")


if __name__ == "__main__":
    main()
