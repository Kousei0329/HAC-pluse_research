import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# --- Global Settings ---
LAMBDAS = [0.001, 0.002, 0.003, 0.004, 0.005]
CAUSAL_KNN_K = 8
GPU_ID = 0
EXPERIMENT_NAME = "K8_fp16_comparison"
RANDOM_SEED = 42

DATASET_CONFIG = {
    'scene': 'truck',
    'data_base': '/workspace/HAC/data/tandt',
    'voxel_size': 0.01,
    'mask_lr_final_base': 0.0001,
    'update_init_factor': 16,
    'lod': 0,
}

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def create_command(lmbda, gpu_id, seed):
    cfg = DATASET_CONFIG
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001

    data_path = f"{cfg['data_base']}/{cfg['scene']}"
    output_path = f"{EXPERIMENT_DIR}/lambda_{lmbda}"

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
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
        f'--no_use_reno '
        f'--quantize_mlp_fp16 '
        # K fixed at 8 (best-of-sweep candidate from the earlier K ablation); this run isolates
        # the effect of casting mlp_grid/mlp_deform to true fp16 on top of that fixed K.
    )
    return cmd, output_path


def run_experiment(lmbda, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, gpu_id, seed)

    print(f"\n{'='*80}")
    print(f"Starting experiment {exp_num}/{total}:")
    print(f"  Scene:   truck")
    print(f"  K:       {CAUSAL_KNN_K} (fixed)")
    print(f"  MLP:     fp16")
    print(f"  Lambda:  {lmbda}")
    print(f"  GPU:     {gpu_id}")
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
    print(f"  Lambda: {lmbda}")
    print(f"  Time: {elapsed/60:.2f} minutes")
    print(f"  Return code: {result.returncode}")
    print(f"  Log: {log_file}")
    print(f"{'='*80}\n")

    return {
        'lmbda': lmbda,
        'output_path': output_path,
        'log_file': log_file,
        'duration_minutes': elapsed / 60,
        'return_code': result.returncode,
    }


def main():
    total = len(LAMBDAS)

    config = {
        "timestamp": TIMESTAMP,
        "lambdas": LAMBDAS,
        "causal_knn_K": CAUSAL_KNN_K,
        "quantize_mlp_fp16": True,
        "scene": DATASET_CONFIG['scene'],
        "gpu_id": GPU_ID,
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "use_reno": False,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"K=8 + MLP fp16 Suite (Sequential Execution)")
    print(f"{'='*80}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"causal_knn_K: {CAUSAL_KNN_K} (fixed)")
    print(f"MLP: fp16 (--quantize_mlp_fp16)")
    print(f"Scene: truck (RENO disabled)")
    print(f"GPU: {GPU_ID}")
    print(f"Total experiments: {total}")
    print(f"{'='*80}\n")

    completed = []
    for i, lmbda in enumerate(LAMBDAS, 1):
        result = run_experiment(lmbda, GPU_ID, RANDOM_SEED, i, total)
        completed.append(result)

    print(f"\n{'='*80}")
    print(f"All {len(completed)} experiments completed!")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")

    summary = {
        'experiment_dir': EXPERIMENT_DIR,
        'config': config,
        'completed_experiments': completed,
        'total_duration_minutes': sum(r['duration_minutes'] for r in completed),
    }
    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results summary: {EXPERIMENT_DIR}/results_summary.json")
    print(f"Total execution time: {summary['total_duration_minutes']:.2f} minutes\n")


if __name__ == "__main__":
    main()
