import os
import sys
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_metrics_from_log(log_file):
    """Parse training metrics from log file"""
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'loss': [],
        'iteration': []
    }

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return metrics

    with open(log_file, 'r') as f:
        for line in f:
            # Parse evaluation results (adjust patterns based on your log format)
            # Example: "Iteration 30000: PSNR=25.34 SSIM=0.89 LPIPS=0.12"

            # PSNR pattern
            psnr_match = re.search(r'PSNR[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if psnr_match:
                metrics['psnr'].append(float(psnr_match.group(1)))

            # SSIM pattern
            ssim_match = re.search(r'SSIM[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if ssim_match:
                metrics['ssim'].append(float(ssim_match.group(1)))

            # LPIPS pattern
            lpips_match = re.search(r'LPIPS[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if lpips_match:
                metrics['lpips'].append(float(lpips_match.group(1)))

            # Loss pattern
            loss_match = re.search(r'Loss[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if loss_match:
                metrics['loss'].append(float(loss_match.group(1)))

    return metrics

def get_model_size(output_path):
    """Calculate total model size in MB"""
    total_size = 0

    # Look for common model file extensions
    extensions = ['.ply', '.pth', '.pt', '.pkl', '.npz', '.bin']

    if not os.path.exists(output_path):
        return None

    for root, dirs, files in os.walk(output_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

    # Convert to MB
    return total_size / (1024 * 1024) if total_size > 0 else None

def parse_results_file(results_path):
    """Parse results.json file if it exists"""
    if not os.path.exists(results_path):
        return None

    with open(results_path, 'r') as f:
        return json.load(f)

def collect_experiment_results(experiment_dir):
    """Collect all results from experiment directory"""
    results = []

    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {experiment_dir}")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load results summary
    summary_path = os.path.join(experiment_dir, 'results_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = None

    # Collect results for each lambda
    for scene in config['scenes']:
        for lmbda in config['lambdas']:
            output_path = os.path.join(experiment_dir, scene, f'lambda_{lmbda}')
            log_file = os.path.join(output_path, 'training.log')
            results_file = os.path.join(output_path, 'results.json')

            # Parse metrics
            metrics = parse_metrics_from_log(log_file)
            results_data = parse_results_file(results_file)

            # Get final metrics (last values or from results.json)
            final_psnr = metrics['psnr'][-1] if metrics['psnr'] else None
            final_ssim = metrics['ssim'][-1] if metrics['ssim'] else None
            final_lpips = metrics['lpips'][-1] if metrics['lpips'] else None

            # Override with results.json if available
            if results_data:
                final_psnr = results_data.get('PSNR', final_psnr)
                final_ssim = results_data.get('SSIM', final_ssim)
                final_lpips = results_data.get('LPIPS', final_lpips)

            # Get model size
            model_size = get_model_size(output_path)

            results.append({
                'lambda': lmbda,
                'scene': scene,
                'psnr': final_psnr,
                'ssim': final_ssim,
                'lpips': final_lpips,
                'model_size_mb': model_size,
                'metrics': metrics,
                'output_path': output_path
            })

    return results

def plot_results(results, experiment_dir):
    """Create plots with model size on x-axis and quality metrics on y-axis"""

    # Group by scene
    scenes = {}
    for r in results:
        scene = r['scene']
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(r)

    # Create plots for each scene
    for scene_name, scene_results in scenes.items():
        # Sort by model size
        scene_results = sorted(scene_results, key=lambda x: x['model_size_mb'] if x['model_size_mb'] is not None else 0)

        # Filter out results with valid metrics and model size
        valid_results = [r for r in scene_results if r['model_size_mb'] is not None]

        if not valid_results:
            print(f"Warning: No valid results with model size for scene {scene_name}")
            continue

        model_sizes = [r['model_size_mb'] for r in valid_results]
        lambdas = [r['lambda'] for r in valid_results]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Rate-Distortion Curve - Scene: {scene_name}', fontsize=16, fontweight='bold')

        # PSNR vs Model Size
        psnr_data = [(r['model_size_mb'], r['psnr'], r['lambda'])
                     for r in valid_results if r['psnr'] is not None]
        if psnr_data:
            sizes, psnrs, lmds = zip(*psnr_data)
            axes[0, 0].plot(sizes, psnrs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            axes[0, 0].set_xlabel('Model Size (MB)', fontsize=12)
            axes[0, 0].set_ylabel('PSNR (dB)', fontsize=12)
            axes[0, 0].set_title('PSNR vs Model Size', fontsize=13, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            for s, p, l in zip(sizes, psnrs, lmds):
                axes[0, 0].annotate(f'λ={l:.3f}\n{p:.2f}dB', (s, p),
                                   textcoords="offset points",
                                   xytext=(10,5), ha='left', fontsize=8)

        # SSIM vs Model Size
        ssim_data = [(r['model_size_mb'], r['ssim'], r['lambda'])
                     for r in valid_results if r['ssim'] is not None]
        if ssim_data:
            sizes, ssims, lmds = zip(*ssim_data)
            axes[0, 1].plot(sizes, ssims, 'o-', linewidth=2, markersize=8, color='#A23B72')
            axes[0, 1].set_xlabel('Model Size (MB)', fontsize=12)
            axes[0, 1].set_ylabel('SSIM', fontsize=12)
            axes[0, 1].set_title('SSIM vs Model Size', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            for s, ss, l in zip(sizes, ssims, lmds):
                axes[0, 1].annotate(f'λ={l:.3f}\n{ss:.4f}', (s, ss),
                                   textcoords="offset points",
                                   xytext=(10,5), ha='left', fontsize=8)

        # LPIPS vs Model Size (lower is better)
        lpips_data = [(r['model_size_mb'], r['lpips'], r['lambda'])
                      for r in valid_results if r['lpips'] is not None]
        if lpips_data:
            sizes, lpips_vals, lmds = zip(*lpips_data)
            axes[1, 0].plot(sizes, lpips_vals, 'o-', linewidth=2, markersize=8, color='#F18F01')
            axes[1, 0].set_xlabel('Model Size (MB)', fontsize=12)
            axes[1, 0].set_ylabel('LPIPS (lower is better)', fontsize=12)
            axes[1, 0].set_title('LPIPS vs Model Size', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            for s, lp, l in zip(sizes, lpips_vals, lmds):
                axes[1, 0].annotate(f'λ={l:.3f}\n{lp:.4f}', (s, lp),
                                   textcoords="offset points",
                                   xytext=(10,-15), ha='left', fontsize=8)

        # Summary table
        axes[1, 1].axis('off')
        table_data = []
        for r in valid_results:
            row = [
                f"{r['lambda']:.4f}",
                f"{r['model_size_mb']:.2f}" if r['model_size_mb'] else "N/A",
                f"{r['psnr']:.2f}" if r['psnr'] else "N/A",
                f"{r['ssim']:.4f}" if r['ssim'] else "N/A",
                f"{r['lpips']:.4f}" if r['lpips'] else "N/A"
            ]
            table_data.append(row)

        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Lambda', 'Size(MB)', 'PSNR', 'SSIM', 'LPIPS'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0.1, 1, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4A4A4A')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')

        axes[1, 1].set_title('Results Summary', fontsize=13, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(experiment_dir, f'rate_distortion_{scene_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_lambda_results.py <experiment_directory>")
        print("Example: python plot_lambda_results.py outputs/tandt_test/experiments/multi_lambda_comparison_2024-01-15_10-30-00")
        sys.exit(1)

    experiment_dir = sys.argv[1]

    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    print(f"\nCollecting results from: {experiment_dir}")
    print("="*80)

    results = collect_experiment_results(experiment_dir)

    if not results:
        print("Error: Could not collect results")
        sys.exit(1)

    print(f"\nFound {len(results)} experiment results")

    # Print summary
    print("\nResults Summary:")
    print("-"*80)
    for r in results:
        print(f"Lambda: {r['lambda']:.4f} | Scene: {r['scene']}")
        if r['model_size_mb']:
            print(f"  Model Size: {r['model_size_mb']:.2f} MB")
        if r['psnr']:
            print(f"  PSNR: {r['psnr']:.2f} dB")
        if r['ssim']:
            print(f"  SSIM: {r['ssim']:.4f}")
        if r['lpips']:
            print(f"  LPIPS: {r['lpips']:.4f}")
        print()

    print("="*80)
    print("\nGenerating plots...")

    plot_results(results, experiment_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
