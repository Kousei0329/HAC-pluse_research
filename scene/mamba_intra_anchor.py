"""
Mamba-based Intra-Anchor Module
MambaをIntra-Anchor生成に使用するモジュール
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")


class MambaIntraAnchor(nn.Module):
    """
    MambaベースのIntra-Anchorモジュール
    Level 1アンカーからLevel 2アンカーを生成
    """
    def __init__(self,
                 input_dim=89,           # Level 1特徴次元 (3+50+30+6)
                 hidden_dim=256,         # Mamba隠れ層次元
                 level2_per_level1=64,   # Level 1あたりのLevel 2生成数
                 d_state=16,             # SSMの状態次元
                 d_conv=4,               # 畳み込みカーネルサイズ
                 n_layers=2,             # Mambaレイヤー数
                 expand=2):              # 拡張係数
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required for MambaIntraAnchor. Install with: pip install mamba-ssm")

        self.level2_per_level1 = level2_per_level1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 入力投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Mambaブロック（残差接続付き）
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layers)
        ])

        # 層正規化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # 出力投影
        self.output_proj = nn.Linear(
            hidden_dim,
            level2_per_level1 * input_dim
        )

        print(f'MambaIntraAnchor initialized:')
        print(f'  Input dim: {input_dim}')
        print(f'  Hidden dim: {hidden_dim}')
        print(f'  N layers: {n_layers}')
        print(f'  d_state: {d_state}')
        print(f'  Level2 per Level1: {level2_per_level1}')

    def forward(self, level1_features):
        """
        Args:
            level1_features: [N1, 89] Level 1アンカーの特徴
                - [:, :3]: 座標
                - [:, 3:53]: 特徴量
                - [:, 53:83]: オフセット
                - [:, 83:89]: スケーリング
        Returns:
            level2_params: [N1, 64, 89] Level 2アンカーのパラメータ
        """
        N1 = level1_features.shape[0]

        # 入力投影
        x = self.input_proj(level1_features)  # [N1, hidden_dim]

        # Mambaブロックの適用（残差接続 + LayerNorm）
        for mamba, ln in zip(self.mamba_layers, self.layer_norms):
            residual = x
            # Mambaは[B, L, D]を期待するのでunsqueeze/squeeze
            x = mamba(x.unsqueeze(0))  # [1, N1, hidden_dim]
            x = x.squeeze(0)            # [N1, hidden_dim]
            x = ln(x + residual)        # 残差接続 + LayerNorm

        # 出力投影とリシェイプ
        x = self.output_proj(x)  # [N1, level2_per_level1 * input_dim]
        level2_params = x.view(N1, self.level2_per_level1, self.input_dim)

        return level2_params  # [N1, 64, 89]


class SpatialMambaIntraAnchor(nn.Module):
    """
    空間認識版Mamba Intra-Anchor
    Z-order curveでソートしてBidirectional Mambaを適用
    """
    def __init__(self,
                 input_dim=89,
                 hidden_dim=256,
                 level2_per_level1=64,
                 d_state=16,
                 d_conv=4,
                 use_bidirectional=True):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")

        self.level2_per_level1 = level2_per_level1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bidirectional = use_bidirectional

        # 3D位置エンコーディング（Sinusoidal）
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding_dim = hidden_dim // 4
            # 位置エンコーディング用の周波数
            freqs = torch.exp(torch.linspace(0, -4, self.pos_encoding_dim // 6))
            self.register_buffer('freqs', freqs)

        # 入力投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Bidirectional Mamba
        if use_bidirectional:
            self.mamba_forward = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
            )
            self.mamba_backward = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
            )
            # 両方向の特徴を融合
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 出力MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, level2_per_level1 * input_dim)
        )

        print(f'SpatialMambaIntraAnchor initialized:')
        print(f'  Bidirectional: {use_bidirectional}')
        print(f'  Positional encoding: {self.use_pos_encoding}')

    def compute_morton_code(self, xyz):
        """
        3D座標をMorton code (Z-order curve)に変換

        Args:
            xyz: [N, 3] 3D座標
        Returns:
            morton_codes: [N] Morton codes
        """
        # 座標を正の整数に変換（シフト + スケール）
        xyz_min = xyz.min(dim=0, keepdim=True)[0]
        xyz_max = xyz.max(dim=0, keepdim=True)[0]
        xyz_range = xyz_max - xyz_min + 1e-6

        xyz_norm = (xyz - xyz_min) / xyz_range  # [0, 1]に正規化
        xyz_int = (xyz_norm * 1023).long()  # 10ビット整数に変換

        # Morton codeを計算（bit interleaving）
        def part1by2(n):
            """1ビットを3ビットに展開（0と0を挿入）"""
            n = n & 0x000003ff  # 下位10ビットのみ保持
            n = (n ^ (n << 16)) & 0xff0000ff
            n = (n ^ (n << 8)) & 0x0300f00f
            n = (n ^ (n << 4)) & 0x030c30c3
            n = (n ^ (n << 2)) & 0x09249249
            return n

        x = xyz_int[:, 0]
        y = xyz_int[:, 1]
        z = xyz_int[:, 2]

        morton_codes = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        return morton_codes

    def positional_encoding_3d(self, xyz):
        """
        3D座標の位置エンコーディング

        Args:
            xyz: [N, 3]
        Returns:
            pos_enc: [N, pos_encoding_dim]
        """
        N = xyz.shape[0]
        # 各軸ごとにsin/cosエンコーディング
        pos_enc_list = []
        for i in range(3):
            coord = xyz[:, i:i+1]  # [N, 1]
            # sin/cosを交互に適用
            freqs = self.freqs.view(1, -1)  # [1, freq_dim]
            angles = coord * freqs  # [N, freq_dim]
            pos_enc_list.append(torch.sin(angles))
            pos_enc_list.append(torch.cos(angles))

        pos_enc = torch.cat(pos_enc_list, dim=-1)  # [N, pos_encoding_dim]
        return pos_enc

    def forward(self, level1_combined):
        """
        Args:
            level1_combined: [N1, 89]
                - [:, :3]: 座標
                - [:, 3:]: その他の特徴
        Returns:
            level2_params: [N1, 64, 89]
        """
        N1 = level1_combined.shape[0]

        # 座標を抽出
        xyz = level1_combined[:, :3]  # [N1, 3]

        # Morton codeで空間ソート
        morton_codes = self.compute_morton_code(xyz)
        sorted_indices = torch.argsort(morton_codes)
        unsort_indices = torch.argsort(sorted_indices)

        sorted_features = level1_combined[sorted_indices]
        sorted_xyz = xyz[sorted_indices]

        # 位置エンコーディング
        if self.use_pos_encoding:
            pos_enc = self.positional_encoding_3d(sorted_xyz)
            # 入力投影後に位置エンコーディングを追加
            x = self.input_proj(sorted_features)  # [N1, hidden_dim]
            # 位置エンコーディングの次元調整
            if pos_enc.shape[1] < self.hidden_dim:
                pos_enc = torch.cat([
                    pos_enc,
                    torch.zeros(N1, self.hidden_dim - pos_enc.shape[1], device=pos_enc.device)
                ], dim=-1)
            elif pos_enc.shape[1] > self.hidden_dim:
                pos_enc = pos_enc[:, :self.hidden_dim]
            x = x + pos_enc
        else:
            x = self.input_proj(sorted_features)

        # Bidirectional Mamba
        if self.use_bidirectional:
            # Forward pass
            x_fwd = self.mamba_forward(x.unsqueeze(0)).squeeze(0)  # [N1, hidden_dim]

            # Backward pass
            x_flipped = torch.flip(x, [0])
            x_bwd = self.mamba_backward(x_flipped.unsqueeze(0)).squeeze(0)
            x_bwd = torch.flip(x_bwd, [0])  # 元の順序に戻す

            # 両方向を融合
            x = self.fusion(torch.cat([x_fwd, x_bwd], dim=-1))
        else:
            x = self.mamba(x.unsqueeze(0)).squeeze(0)

        x = self.layer_norm(x)

        # 出力生成
        output = self.output_mlp(x)  # [N1, level2_per_level1 * input_dim]
        level2_params = output.view(N1, self.level2_per_level1, self.input_dim)

        # 元の順序に戻す
        level2_params = level2_params[unsort_indices]

        return level2_params


def create_intra_anchor_module(module_type='mlp', **kwargs):
    """
    Intra-Anchorモジュールのファクトリ関数

    Args:
        module_type: 'mlp', 'mamba', 'spatial_mamba'
        **kwargs: モジュール固有のパラメータ
    """
    if module_type == 'mlp':
        # 従来のMLPモジュールを返す
        input_dim = kwargs.get('input_dim', 89)
        hidden_dim = kwargs.get('hidden_dim', 256)
        output_dim = kwargs.get('output_dim', 64 * 89)

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )

    elif module_type == 'mamba':
        return MambaIntraAnchor(**kwargs)

    elif module_type == 'spatial_mamba':
        return SpatialMambaIntraAnchor(**kwargs)

    else:
        raise ValueError(f"Unknown module_type: {module_type}. Choose from ['mlp', 'mamba', 'spatial_mamba']")
