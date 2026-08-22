import os
import subprocess
from datetime import datetime
import time
import json
from pathlib import Path

# Set environment variable to force compatible CUDA architecture
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'

# Configuration
LAMBDAS = [0.001, 0.002, 0.003, 0.004, 0.005]  # Multiple lambda values to test
SCENES = ['truck']  # Can add 'train' or other scenes
GPU_ID = 0  # Single GPU for sequential execution
EXPERIMENT_NAME = "multi_lambda_comparison"
BASE_OUTPUT_DIR = "outputs/tandt_test"

# Create experiment directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"{BASE_OUTPUT_DIR}/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Save experiment configuration
config = {
    "timestamp": TIMESTAMP,
    "lambdas": LAMBDAS,
    "scenes": SCENES,
    "gpu_id": GPU_ID,
    "experiment_name": EXPERIMENT_NAME
}
with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

def create_command(lmbda, scene, gpu_id):
    """Create training command for given parameters"""
    mask_lr_final = 0.0001 * lmbda / 0.001
    data_path = f'/workspace/HAC/data/tandt/{scene}'

    output_path = f'{EXPERIMENT_DIR}/{scene}/lambda_{lmbda}'

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'-s {data_path} '
        f'--eval '
        # f'--resolution 320 '
        f'--lod 0 '
        f'--voxel_size 0.01 '
        f'--update_init_factor 16 '
        f'--iterations 30_000 '
        f'-m {output_path} '
        f'--lmbda {lmbda} '
        f'--mask_lr_final {mask_lr_final}'
    )

    return cmd, output_path

def run_experiment(lmbda, scene, gpu_id):
    """Run a single experiment and wait for completion"""
    cmd, output_path = create_command(lmbda, scene, gpu_id)

    print(f"\n{'='*80}")
    print(f"Starting experiment {len(completed_experiments) + 1}/{total_experiments}:")
    print(f"  Lambda: {lmbda}")
    print(f"  Scene: {scene}")
    print(f"  GPU: {gpu_id}")
    print(f"  Output: {output_path}")
    print(f"{'='*80}\n")

    # Create log file
    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)

    # Run training and wait for completion
    start_time = time.time()

    print(f"Running command:\n{cmd}\n")

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT
        )

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Completed experiment {len(completed_experiments) + 1}/{total_experiments}:")
    print(f"  Lambda: {lmbda}")
    print(f"  Scene: {scene}")
    print(f"  Time: {elapsed/60:.2f} minutes")
    print(f"  Return code: {result.returncode}")
    print(f"  Log file: {log_file}")
    print(f"{'='*80}\n")

    return {
        'lmbda': lmbda,
        'scene': scene,
        'output_path': output_path,
        'log_file': log_file,
        'duration_minutes': elapsed / 60,
        'return_code': result.returncode
    }

def main():
    global total_experiments, completed_experiments

    print(f"\n{'='*80}")
    print(f"Multi-Lambda Experiment Suite (Sequential Execution)")
    print(f"{'='*80}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Scenes: {SCENES}")
    print(f"GPU: {GPU_ID}")
    print(f"{'='*80}\n")

    # Build experiment list
    experiments = []
    for scene in SCENES:
        for lmbda in LAMBDAS:
            experiments.append({'lmbda': lmbda, 'scene': scene})

    total_experiments = len(experiments)
    completed_experiments = []

    print(f"Total experiments to run: {total_experiments}")
    print(f"Execution mode: Sequential (one at a time)\n")

    # Run experiments sequentially
    for exp in experiments:
        result = run_experiment(exp['lmbda'], exp['scene'], GPU_ID)
        completed_experiments.append(result)

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Total experiments: {len(completed_experiments)}")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")

    # Save results summary
    summary = {
        'experiment_dir': EXPERIMENT_DIR,
        'config': config,
        'completed_experiments': completed_experiments,
        'total_duration_minutes': sum(r['duration_minutes'] for r in completed_experiments)
    }

    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults summary saved to: {EXPERIMENT_DIR}/results_summary.json")
    print(f"Total execution time: {summary['total_duration_minutes']:.2f} minutes\n")
    print(f"\nTo visualize results, run:")
    print(f"  python plot_lambda_results.py {EXPERIMENT_DIR}")

if __name__ == "__main__":
    main()
