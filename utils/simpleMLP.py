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
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class PointMLP(nn.Module):
    """
    位置埋め込み無し・完全 point-wise の軽量 MLP
    入力 : (N, 3)
    出力 : (N, out_dim)
    """
    def __init__(
        self,
        out_dim: int = 48,
        quantize: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 2,
        **kwargs,  # 互換のため unused 引数を許可
    ):
        super().__init__()

        self.quantize = quantize

        layers = []

        # --- 最初の層: (3 → hidden_dim) ---
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # --- 中間 hidden 層 ---
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        # --- 出力層 ---
        self.fc_out1 = nn.Linear(hidden_dim, 256)
        self.fc_out2 = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor, test_phase: bool = False) -> torch.Tensor:
        """
        x: (N, 3)
        """
        assert x.dim() == 2 and x.size(1) == 3, \
            f"Expected input shape (N, 3), got {tuple(x.shape)}"

        feat = self.mlp(x)               # (N, hidden_dim)

        h = F.relu(self.fc_out1(feat))   # (N, 256)
        out = self.fc_out2(h)            # (N, out_dim)

        out = torch.tanh(out)

        if self.quantize:
            out = STE_binary.apply(out)

        return out
