from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STE_binary(torch.autograd.Function):
    """
    Straight-Through Estimator for binary quantization
    Forward: quantize to {-1, +1}
    Backward: pass gradient through (with clamping)
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # Binarize: sign function
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient mask: only pass gradient if input is in valid range
        (input,) = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


def farthest_point_sampling(points: torch.Tensor,
                            num_samples: int):
    """
    単純な FPS (Farthest Point Sampling) 実装 (バッチなし版)
    points: (N, 3)
    num_samples: 取得したい点数 M (M <= N)

    return:
        sampled_points: (M, 3)
        sampled_idx   : (M,)  元の点群に対するインデックス
    """
    assert points.dim() == 2 and points.size(1) == 3, \
        f"Expected (N, 3), got {tuple(points.shape)}"
    N = points.size(0)
    device = points.device

    if num_samples >= N:
        # ダウンサンプル不要
        idx = torch.arange(N, device=device, dtype=torch.long)
        return points, idx

    # 入力の異常値チェック
    if torch.isnan(points).any() or torch.isinf(points).any():
        print(f"[WARNING] FPS input contains NaN or Inf: NaN={torch.isnan(points).sum()}, Inf={torch.isinf(points).sum()}")
        # 異常値を除去またはクリップ
        points = torch.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)

    # (N,) 各点の「最近の採用点までの距離」を管理
    distances = torch.full((N,), 1e10, device=device, dtype=points.dtype)  # float("inf")の代わりに大きな有限値
    # サンプルされたインデックス
    sampled_idx = torch.zeros(num_samples, dtype=torch.long, device=device)

    # 初期点をランダムに選択（or 0 固定でも OK）
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_idx[i] = farthest
        centroid = points[farthest, :].unsqueeze(0)  # (1, 3)
        # 各点との距離 (L2)
        dist = torch.sum((points - centroid) ** 2, dim=1)  # (N,)

        # 距離計算後の異常値チェック
        if torch.isnan(dist).any() or torch.isinf(dist).any():
            print(f"[WARNING] FPS iteration {i}: dist contains NaN or Inf")
            dist = torch.nan_to_num(dist, nan=1e10, posinf=1e10, neginf=0.0)

        # これまでの最近距離より近ければ更新
        closer = dist < distances
        distances[closer] = dist[closer]
        # 一番遠い点を次の「farthest」とする
        farthest = torch.argmax(distances).item()

    sampled_points = points[sampled_idx, :]
    return sampled_points, sampled_idx


class PointNet(nn.Module):
    """
    バッチ次元なし版 PointNet 風特徴抽出器
    入力 : (N, 3)  - N個の点 (x, y, z)
    出力 : (M, out_dim) - FPS後 M個の点ごとの特徴
           （num_samples=None のときは M = N）
    """
    def __init__(self, out_dim: int = 48, quantize: bool = True, num_samples: int | None = None):
        super().__init__()

        self.quantize = quantize
        self.num_samples = num_samples

        # 1段目: per-point local feature MLP
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        # 2段目: local(256) + global(256) = 512 → out_dim へ
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        test_phase: bool = False,
        num_samples: int | None = None,
    ):
        """
        x          : (N, 3)
        test_phase : いまは未使用（STEを常に使用）
        num_samples: FPS で残したい点数 M（None のときはダウンサンプルしない）

        return:
            feat: (N, out_dim)  # 最近傍補間で元のサイズに戻す
            fps_idx: (M,) or None
                    元の点群 x に対するインデックス
        """
        assert x.dim() == 2 and x.size(1) == 3, \
            f"Expected input shape (N, 3), got {tuple(x.shape)}"

        N = x.size(0)

        # # --- 0. FPS によるダウンサンプル ---
        if num_samples is None:
            num_samples = self.num_samples
        if num_samples is not None and N > num_samples:
            x_down, fps_idx = farthest_point_sampling(x, num_samples)
        else:
            x_down = x
            fps_idx = None

        # --- 1. per-point local feature ---
        # x_down: (M, 3) → (M, 64) → (M, 128) → (M, 256)
        x_local = F.relu(self.fc1(x))
        x_local = F.relu(self.fc2(x_local))
        x_local = F.relu(self.fc3(x_local))   # (M, 256)

        # --- 2. global feature (max over M) ---
        # global_feat: (256,)
        global_feat, _ = torch.max(x_local, dim=0)

        # 各点に複製: (256,) → (M, 256)
        global_feat_expanded = global_feat.unsqueeze(0).expand(x_local.size(0), -1)

        # --- 3. local + global を結合 ---
        # (M, 256) + (M, 256) → (M, 512)
        feat_cat = torch.cat([x_local, global_feat_expanded], dim=1)

        # --- 4. 最終 MLP で per-point 特徴へ ---
        h = F.relu(self.fc4(feat_cat))   # (M, 256)
        out = self.fc5(h)                # (M, out_dim)

        out = torch.tanh(out)  # 出力を[-1, 1]に制限

        # --- 5. Quantization (optional) ---
        if self.quantize:
            # Always use STE_binary for consistency with GridEncoder
            out = STE_binary.apply(out)

        # # --- 6. 最近傍補間で元のサイズに戻す ---
        if fps_idx is not None:
            # ダウンサンプルした場合、最近傍補間で元のサイズに復元
            # 各元の点について最も近いサンプリング点の特徴を使用
            dist = torch.cdist(x, x_down)  # [N, M]
            nearest_idx = torch.argmin(dist, dim=1)  # [N]
            out = out[nearest_idx]  # [N, out_dim]

        return out
