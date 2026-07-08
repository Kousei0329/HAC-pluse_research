"""
Mutual information analysis between scaling, offset, and anchor_feat.

Usage:
    python analyze_mutual_info.py <ply_path>
"""

import sys
import numpy as np
from plyfile import PlyData
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

PLY_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/tandt_test/truck/_Q=0.1_afterTanh_PointNet2026-06-10_$18-19-40_0.004/point_cloud/iteration_30000/point_cloud.ply"

print(f"Loading: {PLY_PATH}")
ply = PlyData.read(PLY_PATH)
v = ply['vertex']
N = len(v['x'])
print(f"Anchors: {N:,}\n")

# --- Load groups ---
scaling = np.stack([np.array(v[f'scale_{i}']) for i in range(6)], axis=1)       # (N, 6)
offset  = np.stack([np.array(v[f'f_offset_{i}']) for i in range(30)], axis=1)   # (N, 30)
feat    = np.stack([np.array(v[f'f_anchor_feat_{i}']) for i in range(50)], axis=1)  # (N, 50)

print(f"scaling : {scaling.shape}  range [{scaling.min():.3f}, {scaling.max():.3f}]")
print(f"offset  : {offset.shape}   range [{offset.min():.3f}, {offset.max():.3f}]")
print(f"feat    : {feat.shape}   range [{feat.min():.3f}, {feat.max():.3f}]")

# Normalize (MI is scale-invariant for KNN estimator, but helps numerical stability)
scaler = StandardScaler()
scaling_n = scaler.fit_transform(scaling)
offset_n  = scaler.fit_transform(offset)
feat_n    = scaler.fit_transform(feat)

# Subsample for speed (MI estimation is O(N log N))
np.random.seed(42)
MAX_N = 20_000
if N > MAX_N:
    idx = np.random.choice(N, MAX_N, replace=False)
    scaling_n = scaling_n[idx]
    offset_n  = offset_n[idx]
    feat_n    = feat_n[idx]
    print(f"\nSubsampled to {MAX_N:,} anchors for MI estimation")


def mean_mi(X, Y, n_neighbors=5):
    """
    Average MI across all (X_dim, Y_dim) pairs.
    For each dimension of Y, regress against all X columns.
    Returns mean and per-dim array.
    """
    mi_matrix = np.zeros((X.shape[1], Y.shape[1]))
    for j in range(Y.shape[1]):
        mi_vals = mutual_info_regression(X, Y[:, j], n_neighbors=n_neighbors, random_state=42)
        mi_matrix[:, j] = mi_vals
    return mi_matrix


print("\nComputing MI matrices (this may take a moment)...")

print("  [1/3] scaling ↔ anchor_feat ...")
mi_sf = mean_mi(scaling_n, feat_n)      # (6, 50)

print("  [2/3] offset  ↔ anchor_feat ...")
mi_of = mean_mi(offset_n, feat_n)       # (30, 50)

print("  [3/3] scaling ↔ offset ...")
mi_so = mean_mi(scaling_n, offset_n)    # (6, 30)

# --- Summary ---
print("\n" + "="*55)
print("  Mutual Information Summary  (unit: nats, KNN k=5)")
print("="*55)

pairs = [
    ("scaling → anchor_feat", mi_sf),
    ("offset  → anchor_feat", mi_of),
    ("scaling → offset",      mi_so),
]

for name, mi in pairs:
    print(f"\n  {name}")
    print(f"    mean  : {mi.mean():.4f}")
    print(f"    max   : {mi.max():.4f}  (dim pair {np.unravel_index(mi.argmax(), mi.shape)})")
    print(f"    median: {np.median(mi):.4f}")

# --- Per source-dim breakdown ---
print("\n" + "="*55)
print("  Per source-dim: mean MI across all target dims")
print("="*55)

print("\n  scaling dim  |  →feat  |  →offset")
print("  " + "-"*35)
for i in range(6):
    print(f"  scale_{i}      |  {mi_sf[i].mean():.4f}  |  {mi_so[i].mean():.4f}")

print(f"\n  offset dim average (over 30 dims):")
print(f"    offset → feat   : {mi_of.mean(axis=1).mean():.4f}  "
      f"(max dim {mi_of.mean(axis=1).argmax()}: {mi_of.mean(axis=1).max():.4f})")

print(f"\n  anchor_feat dim average (over 50 dims):")
print(f"    feat ← scaling  : {mi_sf.mean(axis=0).mean():.4f}  "
      f"(max dim {mi_sf.mean(axis=0).argmax()}: {mi_sf.mean(axis=0).max():.4f})")
print(f"    feat ← offset   : {mi_of.mean(axis=0).mean():.4f}  "
      f"(max dim {mi_of.mean(axis=0).argmax()}: {mi_of.mean(axis=0).max():.4f})")
