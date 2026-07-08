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
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class PointTransformerLayer(nn.Module):
    """
    単発の PointTransformer layer（バッチなし）。
    x:   (N, C)  - 特徴
    pos: (N, 3)  - 座標
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.dim = dim
        self.k = k

        # Q, K, V
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # 位置エンコーディング φ(Δp)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # ψ(・) → スカラー attention logit
        self.attn_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, C)
        pos: (N, 3)
        return: (N, C)
        """
        N, C = x.shape
        assert pos.shape == (N, 3)

        # Q, K, V
        q = self.to_q(x)            # (N, C)
        k = self.to_k(x)            # (N, C)
        v = self.to_v(x)            # (N, C)

        # --- kNN 近傍取得 ---
        # dist: (N, N)
        dist = torch.cdist(pos, pos, p=2)
        k_neighbor = min(self.k, N)
        # idx: (N, k_neighbor)
        idx = dist.topk(k_neighbor, largest=False).indices

        # 近傍 features / 座標
        # 形状は (N, k_neighbor, C) / (N, k_neighbor, 3)
        k_j = k[idx]
        v_j = v[idx]
        pos_j = pos[idx]

        # 自分の座標・特徴を (N, 1, *) に拡張
        pos_i = pos.unsqueeze(1)          # (N, 1, 3)
        q_i = q.unsqueeze(1)              # (N, 1, C)

        # 相対座標 Δp_ij
        delta = pos_i - pos_j             # (N, k_neighbor, 3)
        r_ij = self.pos_mlp(delta)        # (N, k_neighbor, C)

        # PointTransformer の attention: ψ(q_i - k_j + r_ij)
        attn_feat = q_i - k_j + r_ij      # (N, k_neighbor, C)
        attn_logits = self.attn_mlp(attn_feat).squeeze(-1)  # (N, k_neighbor)

        attn = F.softmax(attn_logits, dim=1)               # (N, k_neighbor)

        # 出力: Σ_j α_ij (v_j + r_ij)
        out = (attn.unsqueeze(-1) * (v_j + r_ij)).sum(dim=1)  # (N, C)

        return out


class PointTransformerBlock(nn.Module):
    """
    PointTransformer layer + FFN + 残差
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.pt_layer = PointTransformerLayer(dim, k)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        x = x + self.pt_layer(x, pos)
        # FFN + residual
        x = x + self.ffn(x)
        return x


def nearest_neighbor_interpolate(
    src_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_feat: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    src_pos: (N, 3) 元の点
    ref_pos: (M, 3) サンプリングした点
    ref_feat:(M, C) サンプリング点上の特徴
    return:  (N, C) 各元点に対して最近傍の ref_feat を割り当て
    """
    device = ref_pos.device
    N = src_pos.size(0)
    C = ref_feat.size(1)

    out = torch.empty(N, C, device=device, dtype=ref_feat.dtype)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        src_chunk = src_pos[start:end]  # (chunk, 3)
        # (chunk, M)
        dist = torch.cdist(src_chunk, ref_pos)
        nn_idx = dist.argmin(dim=1)     # (chunk,)
        out[start:end] = ref_feat[nn_idx]

    return out


class PointTransformer(nn.Module):
    """
    ダウンサンプル + PointTransformer 版の PointNet 風特徴抽出器
    入力 : (N, 3)  - N個の点 (x, y, z)
    出力 : (N, out_dim) - 各点ごとの特徴（tanh → STE_binary まで含む）
    """
    def __init__(
        self,
        out_dim: int = 48,
        quantize: bool = True,
        embed_dim: int = 128,
        k: int = 16,
        num_layers: int = 2,
        max_points: int = 4096,      # PointTransformer をかける最大点数
        interp_chunk_size: int = 4096,
    ):
        super().__init__()

        self.quantize = quantize
        self.embed_dim = embed_dim
        self.k = k
        self.max_points = max_points
        self.interp_chunk_size = interp_chunk_size

        # 座標 → 初期埋め込み
        self.input_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # PointTransformer blocks（ダウンサンプル後の点数に対して適用）
        self.blocks = nn.ModuleList(
            [PointTransformerBlock(embed_dim, k=k) for _ in range(num_layers)]
        )

        # 出力 MLP（元の PointNet 同様、最後は out_dim）
        self.fc_out1 = nn.Linear(embed_dim, 256)
        self.fc_out2 = nn.Linear(256, out_dim)

    def _encode_with_transformer(self, pos: torch.Tensor) -> torch.Tensor:
        """
        ダウンサンプル済みの点群 pos に対して PointTransformer を適用し、
        各点の埋め込み特徴 (M, embed_dim) を返す。
        """
        # 1. 座標から初期特徴へ
        feat = self.input_mlp(pos)  # (M, embed_dim)

        # 2. 複数段の PointTransformer Block
        for blk in self.blocks:
            feat = blk(feat, pos)  # (M, embed_dim)

        return feat

    def forward(self, x: torch.Tensor, test_phase: bool = False) -> torch.Tensor:
        """
        x: (N, 3)
        test_phase: ダミー（シグネチャ互換用）。STE は常に使用。
        return: (N, out_dim)
        """
        assert x.dim() == 2 and x.size(1) == 3, \
            f"Expected input shape (N, 3), got {tuple(x.shape)}"

        pos_full = x  # (N, 3)
        N = pos_full.size(0)

        # --- 1. ダウンサンプリング ---
        if N <= self.max_points:
            # そのまま PointTransformer を適用
            pos_ds = pos_full
            idx_ds = None  # 全点
        else:
            # ランダムサンプリング（必要なら FPS に差し替え可）
            idx_ds = torch.randperm(N, device=pos_full.device)[:self.max_points]
            pos_ds = pos_full[idx_ds]  # (M, 3)

        # --- 2. ダウンサンプル点群に対して PointTransformer ---
        feat_ds = self._encode_with_transformer(pos_ds)  # (M, embed_dim)

        # --- 3. 出力 MLP（まずはダウンサンプル点上の out_dim 特徴） ---
        h_ds = F.relu(self.fc_out1(feat_ds))  # (M, 256)
        out_ds = self.fc_out2(h_ds)           # (M, out_dim)

        # --- 4. 元の N 点へ最近傍補間 ---
        if idx_ds is None:
            # ダウンサンプルしていない場合はそのまま
            out_full = out_ds  # (N, out_dim)
        else:
            # 最近傍補間（1-NN）
            out_full = nearest_neighbor_interpolate(
                src_pos=pos_full,
                ref_pos=pos_ds,
                ref_feat=out_ds,
                chunk_size=self.interp_chunk_size,
            )  # (N, out_dim)

        # --- 5. [-1, 1] に制限 ---
        out_full = torch.tanh(out_full)

        # --- 6. 量子化（GridEncoder と同様に STE を常に適用） ---
        if self.quantize:
            out_full = STE_binary.apply(out_full)

        return out_full
