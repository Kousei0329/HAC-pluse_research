"""
octree_encoder.py
-----------------
hash-grid (GridEncoder) の代替として使える、オクツリー近傍探索 + MLP による
コンテキスト特徴抽出モジュール。

パイプライン:
    入力座標 [N, 3]
        ↓ オクツリー (CPU) で K 近傍インデックスを取得
    相対座標 [N, K, 3]
        ↓ PointNet 風 MLP で集約
    出力特徴 [N, output_dim]

GridEncoder と同じ呼び出しインターフェース:
    encoder = OctreeNeighborEncoder(output_dim=24)
    feats = encoder(x)   # x: [N, 3] (正規化済みまたは生座標どちらでも可)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Octree (CPU / numpy)
# ---------------------------------------------------------------------------

class _OctreeNode:
    """オクツリーの 1 ノード。"""
    __slots__ = ("center", "half", "indices", "children")

    def __init__(self, center: np.ndarray, half: float):
        self.center: np.ndarray = center          # (3,) float64
        self.half: float = half                   # 辺の半長
        self.indices: Optional[np.ndarray] = None # 葉ノードのみ: 点インデックス
        self.children: Optional[list] = None      # 内部ノードのみ: 最大 8 子


_MAX_LEAF_PTS = 32   # 葉に収める最大点数
_MIN_HALF     = 1e-7 # これ以上細かくしない


def _build_node(
    pts: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    half: float,
) -> _OctreeNode:
    node = _OctreeNode(center, half)

    if len(indices) <= _MAX_LEAF_PTS or half < _MIN_HALF:
        node.indices = indices
        return node

    # 8 分割
    node.children = []
    signs = pts[indices] > center  # [M, 3] bool
    for ox in range(2):
        for oy in range(2):
            for oz in range(2):
                mask = (
                    (signs[:, 0] == ox)
                    & (signs[:, 1] == oy)
                    & (signs[:, 2] == oz)
                )
                child_idx = indices[mask]
                if len(child_idx) == 0:
                    continue
                child_center = center + np.array(
                    [
                        (2 * ox - 1) * half * 0.5,
                        (2 * oy - 1) * half * 0.5,
                        (2 * oz - 1) * half * 0.5,
                    ],
                    dtype=np.float64,
                )
                node.children.append(
                    _build_node(pts, child_idx, child_center, half * 0.5)
                )
    return node


class Octree:
    """
    点群 [N, 3] に対するオクツリー。
    build() で構築し、knn_query() で近傍インデックスを返す。
    """

    def __init__(self):
        self._root: Optional[_OctreeNode] = None
        self._pts: Optional[np.ndarray] = None

    def build(self, pts: np.ndarray) -> "Octree":
        """
        pts: [N, 3] float (CPU numpy)
        """
        pts = pts.astype(np.float64)
        self._pts = pts
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) * 0.5
        half = float(np.max(hi - lo)) * 0.5 + 1e-6
        indices = np.arange(len(pts), dtype=np.int64)
        self._root = _build_node(pts, indices, center, half)
        return self

    # ------------------------------------------------------------------
    # 内部: 1 クエリ点に対する K 近傍探索
    # ------------------------------------------------------------------

    def _search_one(
        self,
        node: _OctreeNode,
        query: np.ndarray,
        k: int,
        heap: list,
    ) -> None:
        """heap は (−距離², インデックス) の max-heap（Python の heapq は min-heap なので負にする）。"""
        import heapq

        # このノードの AABB と query の最小距離² を計算 (枝刈り用)
        lo = node.center - node.half
        hi = node.center + node.half
        closest = np.clip(query, lo, hi)
        min_dist2 = float(np.sum((query - closest) ** 2))

        # heap が k 個以上あり、このノードが現在の最遠点より遠ければスキップ
        if len(heap) >= k and min_dist2 >= -heap[0][0]:
            return

        if node.children is None:
            # 葉: 正確な距離を計算
            diffs = self._pts[node.indices] - query  # [M, 3]
            dists2 = np.sum(diffs ** 2, axis=1)       # [M]
            for idx, d2 in zip(node.indices, dists2):
                d2 = float(d2)
                if len(heap) < k:
                    heapq.heappush(heap, (-d2, int(idx)))
                elif d2 < -heap[0][0]:
                    heapq.heapreplace(heap, (-d2, int(idx)))
        else:
            # 内部ノード: 近い子から優先して探索
            child_min_dists = []
            for child in node.children:
                lo_c = child.center - child.half
                hi_c = child.center + child.half
                clamped = np.clip(query, lo_c, hi_c)
                d = float(np.sum((query - clamped) ** 2))
                child_min_dists.append((d, child))
            child_min_dists.sort(key=lambda t: t[0])
            for _, child in child_min_dists:
                self._search_one(child, query, k, heap)

    def knn_query(self, queries: np.ndarray, k: int) -> np.ndarray:
        """
        queries: [Q, 3]
        k: 近傍数 (自分自身を含む)
        return: [Q, k] int64  (不足分は最後のインデックスで埋める)
        """
        import heapq

        assert self._root is not None, "call build() first"
        Q = len(queries)
        result = np.zeros((Q, k), dtype=np.int64)

        for i, q in enumerate(queries):
            heap: list = []
            self._search_one(self._root, q, k, heap)
            # heap から距離昇順に並べ直す
            heap_sorted = sorted(heap, key=lambda t: -t[0])  # 距離昇順
            n_found = len(heap_sorted)
            for j, (_, idx) in enumerate(heap_sorted):
                result[i, j] = idx
            # 点数が k に満たない場合は最後のインデックスで埋める
            if n_found < k:
                pad_idx = result[i, n_found - 1] if n_found > 0 else 0
                result[i, n_found:] = pad_idx

        return result


# ---------------------------------------------------------------------------
# NeighborMLP: [N, K, 3] → [N, C]
# ---------------------------------------------------------------------------

class NeighborMLP(nn.Module):
    """
    相対座標 [N, K, 3] から per-point 特徴 [N, out_dim] を生成する。

    構造:
        per-neighbor MLP: 3 → hidden → hidden
        max-pool over K
        output MLP: hidden → out_dim
    """

    def __init__(self, out_dim: int = 24, hidden: int = 64, k: int = 16):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

        # Per-neighbor 特徴抽出 (shared MLP)
        self.mlp_neighbor = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

        # max-pool 後の出力 MLP
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        rel_pos: [N, K, 3]  各点の K 近傍への相対座標
        return:  [N, out_dim]
        """
        N, K, _ = rel_pos.shape

        # Shared MLP: [N, K, 3] → [N, K, hidden]
        h = self.mlp_neighbor(rel_pos)

        # Max-pooling over K: [N, K, hidden] → [N, hidden]
        h, _ = h.max(dim=1)

        # 出力 MLP: [N, hidden] → [N, out_dim]
        out = self.mlp_out(h)

        return out


# ---------------------------------------------------------------------------
# OctreeNeighborEncoder  (メインモジュール)
# ---------------------------------------------------------------------------

class OctreeNeighborEncoder(nn.Module):
    """
    オクツリー近傍探索 + PointNet 風 MLP によるコンテキスト特徴抽出器。

    GridEncoder の代替として calc_interp_feat() から呼び出せる。

    Args:
        output_dim : 出力特徴次元 (GridEncoder の output_dim に合わせる。デフォルト 24)
        k_neighbors: 近傍数 (自分自身を含む)
        hidden_dim : 内部 MLP の隠れ次元
        exclude_self: True のとき自分自身 (距離 0) を近傍から除外する
                      (アンカーが既知の点群のとき True が自然)
        fallback_cdist: True のとき N が小さい (< cdist_threshold) 場合は
                        torch.cdist でそのまま KNN を計算する (オクツリー構築
                        オーバーヘッドを避ける)
        cdist_threshold: fallback_cdist を使う N の上限
    """

    def __init__(
        self,
        output_dim: int = 24,
        k_neighbors: int = 16,
        hidden_dim: int = 64,
        exclude_self: bool = True,
        fallback_cdist: bool = True,
        cdist_threshold: int = 512,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.n_output_dims = output_dim  # GridEncoder との互換
        self.k_neighbors = k_neighbors
        self.exclude_self = exclude_self
        self.fallback_cdist = fallback_cdist
        self.cdist_threshold = cdist_threshold

        # 自分自身を除外するとき +1 余分に取って後で切り捨てる
        self._k_query = k_neighbors + 1 if exclude_self else k_neighbors

        self.neighbor_mlp = NeighborMLP(
            out_dim=output_dim,
            hidden=hidden_dim,
            k=k_neighbors,
        )

    # ------------------------------------------------------------------
    # KNN ヘルパー
    # ------------------------------------------------------------------

    def _knn_cdist(
        self, pts: torch.Tensor, queries: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        torch.cdist を使った KNN (小 N 向け)。
        return: [Q, k] LongTensor
        """
        # [Q, N]
        dist = torch.cdist(queries, pts)
        _, indices = dist.topk(k, largest=False, dim=-1)
        return indices  # [Q, k]

    def _knn_octree(
        self, pts: torch.Tensor, queries: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        scipy.spatial.KDTree による KNN (大 N 向け)。C 実装で高速。
        return: [Q, k] LongTensor
        """
        from scipy.spatial import KDTree

        pts_np = pts.detach().cpu().numpy().astype(np.float64)
        queries_np = queries.detach().cpu().numpy().astype(np.float64)

        tree = KDTree(pts_np)
        _, indices_np = tree.query(queries_np, k=k, workers=-1)  # [Q, k] int

        return torch.from_numpy(indices_np.astype(np.int64)).long().to(pts.device)

    def _get_knn_indices(
        self, pts: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """
        KNN インデックス [Q, k_query] を返す (exclude_self 前)。
        N が小さければ cdist, 大きければ octree を使用。
        """
        N = pts.shape[0]
        k = self._k_query

        if self.fallback_cdist and N <= self.cdist_threshold:
            return self._knn_cdist(pts, queries, k)
        else:
            return self._knn_octree(pts, queries, k)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        min_level_id=None,   # GridEncoder 互換 (未使用)
        max_level_id=None,   # GridEncoder 互換 (未使用)
        test_phase=False,    # GridEncoder 互換 (未使用)
        **kwargs,
    ) -> torch.Tensor:
        """
        x: [N, 3]  アンカー座標 (正規化済み [0,1] でも生座標でも可)
        return: [N, output_dim]
        """
        assert x.dim() == 2 and x.shape[1] == 3, \
            f"Expected [N, 3], got {tuple(x.shape)}"

        N = x.shape[0]

        # N が 1 以下の場合はゼロ特徴を返す (デgenerate)
        if N <= 1:
            return x.new_zeros(N, self.output_dim)

        # --- 1. KNN インデックス取得 [N, k_query] ---
        indices = self._get_knn_indices(x, x)  # クエリ = 点群自身

        # --- 2. 自分自身を除外 ---
        if self.exclude_self:
            # distance=0 の要素 (= 自分自身) は先頭にあるはずなので切り捨て
            indices = indices[:, 1:]   # [N, k_neighbors]
        # k_neighbors に満たない場合はクリップ
        k = min(self.k_neighbors, indices.shape[1])
        indices = indices[:, :k]       # [N, k]

        # --- 3. 相対座標を計算 [N, k, 3] ---
        neighbor_pos = x[indices]                     # [N, k, 3]
        rel_pos = neighbor_pos - x.unsqueeze(1)       # [N, k, 3]

        # k が k_neighbors に満たない場合 (点が少ない) はゼロパディング
        if k < self.k_neighbors:
            pad = x.new_zeros(N, self.k_neighbors - k, 3)
            rel_pos = torch.cat([rel_pos, pad], dim=1)  # [N, k_neighbors, 3]

        # --- 4. NeighborMLP で特徴集約 [N, output_dim] ---
        features = self.neighbor_mlp(rel_pos)

        return features

    def __repr__(self) -> str:
        return (
            f"OctreeNeighborEncoder("
            f"output_dim={self.output_dim}, "
            f"k_neighbors={self.k_neighbors}, "
            f"exclude_self={self.exclude_self})"
        )
