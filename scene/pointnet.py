import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """
    バッチ次元なし版 PointNet 風特徴抽出器
    入力 : (N, 3)  - N個の点 (x, y, z)
    出力 : (N, out_dim) - 各点ごとの特徴
    """
    def __init__(self, out_dim: int = 48):
        super().__init__()

        # 1段目: per-point local feature MLP
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        # 2段目: local(256) + global(256) = 512 → out_dim へ
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3)
        return: (N, out_dim)
        """
        assert x.dim() == 2 and x.size(1) == 3, \
            f"Expected input shape (N, 3), got {tuple(x.shape)}"

        # --- 1. per-point local feature ---
        # x: (N, 3) → (N, 64) → (N, 128) → (N, 256)
        x_local = F.relu(self.fc1(x))
        x_local = F.relu(self.fc2(x_local))
        x_local = F.relu(self.fc3(x_local))   # (N, 256)

        # --- 2. global feature (max over N) ---
        # global_feat: (256,)
        global_feat, _ = torch.max(x_local, dim=0)

        # 各点に複製: (256,) → (N, 256)
        global_feat_expanded = global_feat.unsqueeze(0).expand(x_local.size(0), -1)

        # --- 3. local + global を結合 ---
        # (N, 256) + (N, 256) → (N, 512)
        feat_cat = torch.cat([x_local, global_feat_expanded], dim=1)

        # --- 4. 最終 MLP で per-point 特徴へ ---
        h = F.relu(self.fc4(feat_cat))   # (N, 256)
        out = self.fc5(h)                # (N, out_dim)

        return out
