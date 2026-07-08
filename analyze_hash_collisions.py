"""
Hash grid collision analyzer for HAC-plus.

Usage:
    python analyze_hash_collisions.py <ply_path> [--voxel_size 0.01]

Loads anchor positions from a trained point_cloud.ply and reports collision
statistics for each level of the 3D/2D hash grid.
"""

import argparse
import numpy as np
from plyfile import PlyData
from collections import defaultdict

# --- Hash grid config (must match train.py defaults) ---
RESOLUTIONS_3D = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514)
RESOLUTIONS_2D = (130, 258, 514, 1026)
LOG2_HASHMAP_SIZE_3D = 19
LOG2_HASHMAP_SIZE_2D = 17
N_FEATURES = 2  # per level

# Instant-NGP prime constants (from gridencoder.cu)
PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
UINT32_MOD = 2**32


def fast_hash_3d(x, y, z):
    result = (int(x) * PRIMES[0] ^ int(y) * PRIMES[1] ^ int(z) * PRIMES[2]) % UINT32_MOD
    return result


def fast_hash_2d(x, y):
    result = (int(x) * PRIMES[0] ^ int(y) * PRIMES[1]) % UINT32_MOD
    return result


def params_in_level(resolution, num_dim, log2_hashmap_size):
    max_params = 2 ** log2_hashmap_size
    n = min(max_params, resolution ** num_dim)
    return int(np.ceil(n / 8) * 8)


def analyze_level_3d(anchor_grid_coords, resolution, log2_hashmap_size):
    """
    For each anchor, compute its 8 surrounding grid corners at this resolution,
    apply the hash function, count collisions.
    """
    max_params = 2 ** log2_hashmap_size
    table_size = params_in_level(resolution, 3, log2_hashmap_size)
    is_dense = (resolution ** 3) <= max_params

    # Normalize anchor positions to [0, resolution]
    # anchor_grid_coords: (N, 3) integer coordinates in original voxel grid
    # At this hash resolution, we map them proportionally.
    # The CUDA code uses: pos = (input + 0.5) * resolution - 0.5 (align_corners=False)
    # Corner = floor(pos), floor(pos)+1
    # We work directly in hash-grid space.

    # For each anchor, 8 corners
    corners = set()
    for (ax, ay, az) in anchor_grid_coords:
        # Proportional mapping: assume anchor_grid_coords are already in resolution space
        cx, cy, cz = int(ax), int(ay), int(az)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    corners.add((cx + dx, cy + dy, cz + dz))

    corners = list(corners)
    n_unique_corners = len(corners)

    if n_unique_corners == 0:
        return {"resolution": resolution, "is_dense": is_dense,
                "table_size": table_size, "unique_corners": 0,
                "collision_rate": 0.0, "colliding_corners": 0, "n_anchors": 0}

    # Compute hash index for each corner
    hash_to_corners = defaultdict(list)
    for (cx, cy, cz) in corners:
        if is_dense:
            # Dense indexing: no hash needed
            idx = (int(cx) + int(cy) * resolution + int(cz) * resolution * resolution) % table_size
        else:
            idx = fast_hash_3d(cx, cy, cz) % table_size
        hash_to_corners[idx].append((cx, cy, cz))

    # Count how many corners are in a bucket with >1 entry (colliding)
    colliding = sum(len(v) - 1 for v in hash_to_corners.values() if len(v) > 1)
    collision_rate = colliding / n_unique_corners if n_unique_corners > 0 else 0.0

    return {
        "resolution": resolution,
        "is_dense": is_dense,
        "table_size": table_size,
        "unique_corners": n_unique_corners,
        "colliding_corners": colliding,
        "collision_rate": collision_rate,
        "n_anchors": len(anchor_grid_coords),
        "n_buckets_used": len(hash_to_corners),
    }


def analyze_level_2d(anchor_grid_coords_2d, resolution, log2_hashmap_size):
    """Same as 3D but for 2D projection planes."""
    max_params = 2 ** log2_hashmap_size
    table_size = params_in_level(resolution, 2, log2_hashmap_size)
    is_dense = (resolution ** 2) <= max_params

    corners = set()
    for (ax, ay) in anchor_grid_coords_2d:
        cx, cy = int(ax), int(ay)
        for dx in range(2):
            for dy in range(2):
                corners.add((cx + dx, cy + dy))

    corners = list(corners)
    n_unique_corners = len(corners)
    if n_unique_corners == 0:
        return {"resolution": resolution, "is_dense": is_dense,
                "table_size": table_size, "unique_corners": 0,
                "collision_rate": 0.0, "colliding_corners": 0}

    hash_to_corners = defaultdict(list)
    for (cx, cy) in corners:
        if is_dense:
            idx = (int(cx) + int(cy) * resolution) % table_size
        else:
            idx = fast_hash_2d(cx, cy) % table_size
        hash_to_corners[idx].append((cx, cy))

    colliding = sum(len(v) - 1 for v in hash_to_corners.values() if len(v) > 1)
    collision_rate = colliding / n_unique_corners if n_unique_corners > 0 else 0.0

    return {
        "resolution": resolution,
        "is_dense": is_dense,
        "table_size": table_size,
        "unique_corners": n_unique_corners,
        "colliding_corners": colliding,
        "collision_rate": collision_rate,
        "n_buckets_used": len(hash_to_corners),
    }


def load_anchors(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    x = np.array(v['x'])
    y = np.array(v['y'])
    z = np.array(v['z'])
    return np.stack([x, y, z], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", help="Path to point_cloud.ply")
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--log2_hash_3d", type=int, default=LOG2_HASHMAP_SIZE_3D)
    parser.add_argument("--log2_hash_2d", type=int, default=LOG2_HASHMAP_SIZE_2D)
    args = parser.parse_args()

    print(f"Loading: {args.ply_path}")
    anchors = load_anchors(args.ply_path)
    print(f"Anchors: {len(anchors):,}")

    # Convert to integer voxel coords
    anchor_int = np.round(anchors / args.voxel_size).astype(np.int64)
    # Shift to non-negative
    anchor_int -= anchor_int.min(axis=0)

    print(f"\n{'='*70}")
    print(f"3D Hash Grid  (log2_hashmap_size={args.log2_hash_3d}, max_params={2**args.log2_hash_3d:,})")
    print(f"{'='*70}")
    print(f"{'Lv':>3}  {'Res':>5}  {'Type':>6}  {'TableSz':>9}  {'Corners':>9}  {'Collide':>9}  {'Rate':>8}")
    print(f"{'-'*70}")

    total_collide_3d = 0
    total_corners_3d = 0
    aabb_size = anchor_int.max(axis=0) - anchor_int.min(axis=0) + 1
    for lv, res in enumerate(RESOLUTIONS_3D):
        # Per-dimension scaling (matches model: each axis independently normalized to [0,1])
        scale_per_dim = res / aabb_size  # shape (3,)
        scaled = (anchor_int * scale_per_dim).astype(np.int64)
        coords = [tuple(r) for r in scaled]

        stat = analyze_level_3d(coords, res, args.log2_hash_3d)
        tag = "dense" if stat["is_dense"] else "hash"
        print(f"{lv:>3}  {res:>5}  {tag:>6}  {stat['table_size']:>9,}  "
              f"{stat['unique_corners']:>9,}  {stat['colliding_corners']:>9,}  "
              f"{stat['collision_rate']:>7.2%}")
        total_collide_3d += stat["colliding_corners"]
        total_corners_3d += stat["unique_corners"]

    overall_3d = total_collide_3d / total_corners_3d if total_corners_3d > 0 else 0
    print(f"{'':>3}  {'ALL':>5}  {'':>6}  {'':>9}  {total_corners_3d:>9,}  "
          f"{total_collide_3d:>9,}  {overall_3d:>7.2%}")

    print(f"\n{'='*70}")
    print(f"2D Hash Grids  (log2_hashmap_size={args.log2_hash_2d}, max_params={2**args.log2_hash_2d:,})")
    print(f"{'='*70}")
    plane_names = ["XY", "XZ", "YZ"]
    plane_dims  = [(0, 1), (0, 2), (1, 2)]

    for plane_name, (d0, d1) in zip(plane_names, plane_dims):
        print(f"\n  Plane {plane_name}:")
        print(f"  {'Lv':>3}  {'Res':>5}  {'Type':>6}  {'TableSz':>9}  {'Corners':>9}  {'Collide':>9}  {'Rate':>8}")
        print(f"  {'-'*65}")
        coords2d = [(r[d0], r[d1]) for r in anchor_int.tolist()]
        # Per-dimension scaling for 2D planes
        aabb_d0 = anchor_int[:, d0].max() - anchor_int[:, d0].min() + 1
        aabb_d1 = anchor_int[:, d1].max() - anchor_int[:, d1].min() + 1

        total_collide_2d = 0
        total_corners_2d = 0
        for lv, res in enumerate(RESOLUTIONS_2D):
            scale_d0 = res / aabb_d0
            scale_d1 = res / aabb_d1
            scaled_2d = [(int(x * scale_d0), int(y * scale_d1)) for x, y in coords2d]
            stat = analyze_level_2d(scaled_2d, res, args.log2_hash_2d)
            tag = "dense" if stat["is_dense"] else "hash"
            print(f"  {lv:>3}  {res:>5}  {tag:>6}  {stat['table_size']:>9,}  "
                  f"{stat['unique_corners']:>9,}  {stat['colliding_corners']:>9,}  "
                  f"{stat['collision_rate']:>7.2%}")
            total_collide_2d += stat["colliding_corners"]
            total_corners_2d += stat["unique_corners"]

        overall_2d = total_collide_2d / total_corners_2d if total_corners_2d > 0 else 0
        print(f"  {'':>3}  {'ALL':>5}  {'':>6}  {'':>9}  {total_corners_2d:>9,}  "
              f"{total_collide_2d:>9,}  {overall_2d:>7.2%}")


if __name__ == "__main__":
    main()
