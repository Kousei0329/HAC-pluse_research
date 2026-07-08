#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RD plotter for your experiment table.

- Default: uses the hard-coded numbers from the message (no input needed).
- Optionally: read a "long-format" CSV.

Long-format CSV columns example:
method,lambda,total,psnr,ssim,lpips
baseline,0.001,9.6239,26.14,0.8879,0.1446
...

Usage:
  python plot_rd.py
  python plot_rd.py --outdir rd_out
  python plot_rd.py --input results_long.csv --outdir rd_out
  python plot_rd.py --invert-lpips
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import csv


import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


def _prep_curve(x, y):
    """sort by distortion and remove duplicates"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # remove duplicates in x (keep min rate)
    xs, ys = [], []
    i = 0
    while i < len(x):
        j = i + 1
        best = y[i]
        while j < len(x) and np.isclose(x[j], x[i]):
            best = min(best, y[j])
            j += 1
        xs.append(x[i])
        ys.append(best)
        i = j
    return np.array(xs), np.array(ys)


def bd_rate(dist_ref, rate_ref, dist_test, rate_test, deg=3):
    """
    Compute BD-Rate (%) of test vs ref.
    Negative => bitrate saving (better).
    """
    d0, r0 = _prep_curve(dist_ref, rate_ref)
    d1, r1 = _prep_curve(dist_test, rate_test)

    d_min = max(d0.min(), d1.min())
    d_max = min(d0.max(), d1.max())
    if d_max <= d_min:
        raise ValueError("No overlapping distortion range")

    p0 = np.polyfit(d0, np.log(r0), min(deg, len(d0) - 1))
    p1 = np.polyfit(d1, np.log(r1), min(deg, len(d1) - 1))

    P0 = np.polyint(p0)
    P1 = np.polyint(p1)

    i0 = np.polyval(P0, d_max) - np.polyval(P0, d_min)
    i1 = np.polyval(P1, d_max) - np.polyval(P1, d_min)

    avg_diff = (i1 - i0) / (d_max - d_min)
    return (np.exp(avg_diff) - 1.0) * 100.0

def compute_bd_rates(data: Dict[str, RDSeries], baseline_name: str):
    baseline = data[baseline_name]

    results = []

    for name, s in data.items():
        if name == baseline_name:
            continue

        bd_psnr = bd_rate(
            baseline.psnr, baseline.total,
            s.psnr, s.total
        )
        bd_ssim = bd_rate(
            baseline.ssim, baseline.total,
            s.ssim, s.total
        )
        bd_lpips = bd_rate(
            [-x for x in baseline.lpips], baseline.total,
            [-x for x in s.lpips], s.total
        )

        results.append((name, bd_psnr, bd_ssim, bd_lpips))

    return results



@dataclass
class RDSeries:
    lambdas: List[float]
    total: List[float]
    psnr: List[float]
    ssim: List[float]
    lpips: List[float]


def get_default_data() -> Dict[str, RDSeries]:
    # Lambdas are common across all methods in your table
    lam = [0.001, 0.002, 0.003, 0.004, 0.005]

    data: Dict[str, RDSeries] = {
        "Baseline(HAC++)": RDSeries(
            lambdas=lam,
            total=[9.6239, 7.6551, 6.6277, 5.9071, 5.2626],
            psnr=[26.1425838, 26.0183735, 25.9983273, 25.977293, 25.8610497],
            ssim=[0.8879462, 0.8851015, 0.8836405, 0.8830177, 0.8806062],
            lpips=[0.1446863, 0.1507186, 0.1546856, 0.1569068, 0.1614958],
        ),
        # "GMM+GLU": RDSeries(
        #     lambdas=lam,
        #     total=[10.0047, 7.837, 6.7171, 5.9628, 5.391],
        #     psnr=[26.1414948, 26.1072197, 26.0095005, 26.0003204, 26.0260315],
        #     ssim=[0.8887864, 0.8874265, 0.8850371, 0.8827125, 0.8825163],
        #     lpips=[0.1400937, 0.1444082, 0.149715, 0.1546996, 0.1567358],
        # ),
        "Only GMM": RDSeries(
            lambdas=lam,
            total=[9.7941, 7.8631, 6.7166, 5.9215, 5.3002],
            psnr=[26.0721397, 26.0799217, 26.0076523, 25.9796429, 25.897747],
            ssim=[0.8870767, 0.8853942, 0.8844528, 0.8827507, 0.8809143],
            lpips=[0.1468464, 0.1482327, 0.1534469, 0.1553372, 0.160518],
        ),
        "Only GLU": RDSeries(
            lambdas=lam,
            total=[9.7886, 7.761, 6.6983, 5.8057, 5.2301],
            psnr=[26.0704803, 26.0593109, 25.9668655, 26.0014343, 25.8837948],
            ssim=[0.8887366, 0.886593, 0.8844122, 0.8837864, 0.8812649],
            lpips=[0.1393879, 0.1442821, 0.1506099, 0.1535712, 0.1586518],
        ),
        # "GMM+GLU+PointContext": RDSeries(
        #     lambdas=lam,
        #     total=[10.4494, 8.2855, 7.1283, 6.2718, 5.7888],
        #     psnr=[26.0718346, 26.0258312, 25.9919987, 25.965147, 25.8776112],
        #     ssim=[0.8882646, 0.8865145, 0.884631, 0.8830895, 0.8817566],
        #     lpips=[0.1389841, 0.1441815, 0.15096, 0.1556045, 0.1574351],
        # ),
        # "GMM+GLU+ECA1D": RDSeries(
        #     lambdas=lam,
        #     total=[9.9435, 7.8337, 6.6932, 5.9016, 5.3815],
        #     psnr=[26.1399097, 26.0502129, 25.9938812, 26.0129738, 25.9080772],
        #     ssim=[0.8884398, 0.8867716, 0.8851576, 0.8838705, 0.8819289],
        #     lpips=[0.140598, 0.1451813, 0.1485217, 0.1535474, 0.1569721],
        # ),
        # "GMM+GLU+SE": RDSeries(
        #     lambdas=lam,
        #     total=[10.3733, 8.4635, 7.22, 6.1468, 6.0173],
        #     psnr=[26.0749798, 25.9586372, 26.0280342, 25.7957973, 25.9318485],
        #     ssim=[0.8888267, 0.8865365, 0.8852606, 0.8797174, 0.8825368],
        #     lpips=[0.1389553, 0.1445562, 0.1490263, 0.1591134, 0.1561235],
        # ),
        "Only PointTransformer": RDSeries(
            lambdas=lam,
            total=[11.0078, 8.2939, 7.3876, 6.5269, 6.101],
            psnr=[26.0673656, 25.9120388, 25.9160194, 25.9582024, 25.8683739],
            ssim=[0.8889546, 0.8864604, 0.8851676, 0.8838522, 0.8828543],
            lpips=[0.1413817, 0.1472419, 0.1511747, 0.1546244, 0.1571003],
        ),
        "Only PointNet": RDSeries(
            lambdas=lam,
            total=[10.223, 8.0982, 7.0931, 6.2352, 5.7381],
            psnr=[25.9924518, 25.9529324, 25.9620743, 25.9225826, 25.913559],
            ssim=[0.8878354, 0.886513, 0.885496, 0.8843336, 0.8823851],
            lpips=[0.1433601, 0.1462042, 0.1497513, 0.1529928, 0.1569821],
        ),
    }

    return data


def read_long_csv(path: str) -> Dict[str, RDSeries]:
    if pd is None:
        raise RuntimeError("pandas が必要です: pip install pandas")

    df = pd.read_csv(path)

    required = {"method", "lambda", "total", "psnr", "ssim", "lpips"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: missing={sorted(missing)} / required={sorted(required)}")

    # Ensure sorting by lambda within each method
    data: Dict[str, RDSeries] = {}
    for method, g in df.groupby("method"):
        g2 = g.sort_values("lambda")
        data[str(method)] = RDSeries(
            lambdas=[float(x) for x in g2["lambda"].tolist()],
            total=[float(x) for x in g2["total"].tolist()],
            psnr=[float(x) for x in g2["psnr"].tolist()],
            ssim=[float(x) for x in g2["ssim"].tolist()],
            lpips=[float(x) for x in g2["lpips"].tolist()],
        )
    return data


def plot_metric(
    data: Dict[str, RDSeries],
    metric_key: str,
    title: str,
    ylabel: str,
    outpath_base: str,
    invert_y: bool = False,
) -> None:
    plt.figure()
    for method, s in data.items():
        x = s.total
        y = getattr(s, metric_key)
        plt.plot(x, y, marker="o", linewidth=1.6, markersize=4, label=method)

        # Optional: annotate lambda at points (lightly)
        # for xi, yi, li in zip(x, y, s.lambdas):
        #     plt.annotate(f"{li:g}", (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")

    plt.xlabel("Size (MB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=8)

    if invert_y:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(outpath_base + ".png", dpi=200)
    plt.savefig(outpath_base + ".pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="long-format CSV path (optional)")
    ap.add_argument("--outdir", type=str, default="rd_plots", help="output directory")
    ap.add_argument("--invert-lpips", action="store_true", help="invert LPIPS axis (lower is better)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.input:
        data = read_long_csv(args.input)
    else:
        data = get_default_data()

    # RD curves
    plot_metric(
        data=data,
        metric_key="psnr",
        title="RD Curve (PSNR)",
        ylabel="PSNR (dB)",
        outpath_base=os.path.join(args.outdir, "rd_psnr"),
        invert_y=False,
    )
    plot_metric(
        data=data,
        metric_key="ssim",
        title="RD Curve (SSIM)",
        ylabel="SSIM",
        outpath_base=os.path.join(args.outdir, "rd_ssim"),
        invert_y=False,
    )
    plot_metric(
        data=data,
        metric_key="lpips",
        title="RD Curve (LPIPS)",
        ylabel="LPIPS (lower is better)",
        outpath_base=os.path.join(args.outdir, "rd_lpips"),
        invert_y=args.invert_lpips,
    )

    print(f"[OK] Saved RD plots to: {args.outdir}/")
    print("  - rd_psnr.png / rd_psnr.pdf")
    print("  - rd_ssim.png / rd_ssim.pdf")
    print("  - rd_lpips.png / rd_lpips.pdf")

    # ===== BD-Rate computation =====
    baseline_name = "Baseline(HAC++)"
    bd_results = compute_bd_rates(data, baseline_name)

    print("\n===== BD-Rate (baseline = {}) =====".format(baseline_name))
    print("Negative = bitrate reduction (better)\n")
    print(f"{'Method':30s}  {'PSNR[%]':>10s}  {'SSIM[%]':>10s}  {'LPIPS[%]':>10s}")

    for name, bdp, bds, bdl in bd_results:
        print(f"{name:30s}  {bdp:10.2f}  {bds:10.2f}  {bdl:10.2f}")

    # Save to CSV
    csv_path = os.path.join(args.outdir, "bd_rate.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "bd_rate_psnr", "bd_rate_ssim", "bd_rate_lpips"])
        for name, bdp, bds, bdl in bd_results:
            writer.writerow([name, bdp, bds, bdl])

    print(f"\n[OK] BD-Rate saved to {csv_path}")



if __name__ == "__main__":
    main()
