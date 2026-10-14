import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# ship/lambda_0.001 and ship/lambda_0.005 both crashed in
# shrunk_causalknn_fp8_generalization_2026-09-23_10-22-23 with the SAME cause: RENO's periodic
# fine-tune step computed _anchor_int.min(dim=0) on a transiently-all-masked-out anchor set
# (train.py, ~line 430), which is now guarded (skip that one fine-tune step instead of crashing).
# This reruns just ship at both lambdas with the identical config, writing into the SAME
# experiment directory so the generalization sweep's results are complete.
LAMBDAS = [0.001, 0.005]
CAUSAL_KNN_K = 8
CAUSAL_KNN_HIDDEN_MULT = 2
MLP_GRID_HIDDEN_MULT = 4
GPU_ID = 0
RANDOM_SEED = 42

EXISTING_EXPERIMENT_DIR = "outputs/experiments/shrunk_causalknn_fp8_generalization_2026-09-23_10-22-23"

DATASET_CONFIG = {
    'scene': 'ship',
    'data_base': '/workspace/HAC/data/nerf_synthetic',
    'voxel_size': 0.001,
    'mask_lr_final_base': 0.00008,
    'update_init_factor': 4,
    'lod': 0,
}


def create_command(lmbda, gpu_id, seed):
    cfg = DATASET_CONFIG
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001
    data_path = f"{cfg['data_base']}/{cfg['scene']}"
    output_path = f"{EXISTING_EXPERIMENT_DIR}/nerf_synthetic/{cfg['scene']}/lambda_{lmbda}"

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
        f'--causal_knn_hidden_mult {CAUSAL_KNN_HIDDEN_MULT} '
        f'--mlp_grid_hidden_mult {MLP_GRID_HIDDEN_MULT} '
        f'--quantize_mlp_fp8 '
    )
    return cmd, output_path


def run_experiment(lmbda, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, gpu_id, seed)
    print(f"\n{'='*80}\nStarting ship retry {exp_num}/{total}: lambda={lmbda}\n{'='*80}\n")
    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    print(f"Running command:\n{cmd}\n")
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time
    print(f"\n{'='*80}\nCompleted ship retry {exp_num}/{total}: lambda={lmbda}  "
          f"Time: {elapsed/60:.2f}min  Return code: {result.returncode}\n{'='*80}\n")
    return {'lmbda': lmbda, 'output_path': output_path, 'log_file': log_file,
            'duration_minutes': elapsed / 60, 'return_code': result.returncode}


def main():
    total = len(LAMBDAS)
    print(f"\n{'='*80}\nShip retry (fixed empty-anchor RENO fine-tune bug)\n{'='*80}")
    print(f"Lambdas: {LAMBDAS}\n{'='*80}\n")
    completed = []
    for i, lmbda in enumerate(LAMBDAS, 1):
        completed.append(run_experiment(lmbda, GPU_ID, RANDOM_SEED, i, total))
    print(f"\n{'='*80}\nAll {len(completed)} ship retries completed!\n{'='*80}\n")
    with open(f"{EXISTING_EXPERIMENT_DIR}/ship_retry_summary.json", "w") as f:
        json.dump(completed, f, indent=2)


if __name__ == "__main__":
    main()
