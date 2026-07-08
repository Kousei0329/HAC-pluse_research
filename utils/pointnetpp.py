from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# STE (binary) quantization
# ---------------------------
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
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        return p + n

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


# ---------------------------
# FPS (batchless)
# ---------------------------
def farthest_point_sampling(points: torch.Tensor, num_samples: int):
    """
    points: (N, 3)
    return: sampled_points (M,3), sampled_idx (M,)
    """
    assert points.dim() == 2 and points.size(1) == 3
    N = points.size(0)
    device = points.device

    if num_samples >= N:
        idx = torch.arange(N, device=device, dtype=torch.long)
        return points, idx

    if torch.isnan(points).any() or torch.isinf(points).any():
        points = torch.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)

    distances = torch.full((N,), 1e10, device=device, dtype=points.dtype)
    sampled_idx = torch.zeros(num_samples, dtype=torch.long, device=device)

    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_idx[i] = farthest
        centroid = points[farthest].unsqueeze(0)  # (1,3)
        dist = torch.sum((points - centroid) ** 2, dim=1)  # (N,)

        if torch.isnan(dist).any() or torch.isinf(dist).any():
            dist = torch.nan_to_num(dist, nan=1e10, posinf=1e10, neginf=0.0)

        closer = dist < distances
        distances[closer] = dist[closer]
        farthest = torch.argmax(distances).item()

    return points[sampled_idx], sampled_idx


# ---------------------------
# Geometry utils (batchless)
# ---------------------------
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (N,3), dst: (M,3) -> (N,M) squared distance
    """
    diff = src[:, None, :] - dst[None, :, :]
    return (diff * diff).sum(dim=-1)


def ball_query(xyz: torch.Tensor, centroids: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """
    xyz: (N,3), centroids: (M,3)
    return: group_idx (M, nsample)
    - radius 内の点が少ない場合は最近傍で埋める（繰り返し）
    """
    N = xyz.size(0)
    M = centroids.size(0)
    device = xyz.device

    d2 = square_distance(centroids, xyz)  # (M,N)
    within = d2 <= (radius * radius)

    group_idx = torch.empty((M, nsample), device=device, dtype=torch.long)

    for i in range(M):
        idx = torch.nonzero(within[i], as_tuple=False).squeeze(1)  # (K,)
        if idx.numel() == 0:
            _, nn = torch.topk(d2[i], k=nsample, largest=False)
            group_idx[i] = nn
        elif idx.numel() >= nsample:
            d2_i = d2[i, idx]
            _, order = torch.topk(d2_i, k=nsample, largest=False)
            group_idx[i] = idx[order]
        else:
            d2_i = d2[i, idx]
            _, order = torch.topk(d2_i, k=idx.numel(), largest=False)
            idx_sorted = idx[order]
            pad = idx_sorted[0].repeat(nsample - idx_sorted.numel())
            group_idx[i] = torch.cat([idx_sorted, pad], dim=0)

    return group_idx


def three_nn_interpolate(src_xyz: torch.Tensor, src_feat: torch.Tensor, tgt_xyz: torch.Tensor, eps: float = 1e-10):
    """
    3-NN 距離重み付き補間 (PointNet++のFP)
    src_xyz: (M,3)  coarse
    src_feat:(M,C)  coarse feat
    tgt_xyz: (N,3)  fine
    return:  (N,C) interpolated feat
    """
    d2 = square_distance(tgt_xyz, src_xyz)  # (N,M)
    # 3 nearest (or less if not enough points available)
    k = min(3, d2.shape[1])  # Ensure k doesn't exceed available points

    if k == 0:
        # If no source points available, return zeros
        return torch.zeros(tgt_xyz.shape[0], src_feat.shape[1], device=src_feat.device, dtype=src_feat.dtype)

    d2_3, idx_3 = torch.topk(d2, k=k, largest=False)  # (N,k)
    dist = torch.sqrt(d2_3 + eps)                    # (N,k)
    weight = 1.0 / (dist + eps)                      # (N,k)
    weight = weight / torch.sum(weight, dim=1, keepdim=True)  # normalize

    gathered = src_feat[idx_3]  # (N,k,C)
    out = torch.sum(gathered * weight.unsqueeze(-1), dim=1)  # (N,C)
    return out


# ---------------------------
# Small building blocks
# ---------------------------
class PointMLP(nn.Module):
    """
    shared MLP for point features: (..., C_in) -> (..., C_out)
    BN は batchless だと扱いにくいので、必要なら LayerNorm を使う（デフォルト無し）
    """
    def __init__(self, in_ch: int, mlp: list[int], use_ln: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for oc in mlp:
            layers.append(nn.Linear(c, oc))
            if use_ln:
                layers.append(nn.LayerNorm(oc))
            layers.append(nn.ReLU(True))
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., Cin)
        return self.net(x)


class SetAbstractionMSG(nn.Module):
    """
    PointNet++ Set Abstraction (MSG), batchless
      FPS -> (multi-radius ball query) -> shared MLP -> max pool -> concat
    """
    def __init__(
        self,
        npoint: int,
        radii: list[float],
        nsamples: list[int],
        in_channels: int,
        mlps: list[list[int]],   # per scale
        use_ln: bool = False,
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.mlps = nn.ModuleList([
            PointMLP(3 + in_channels, mlp, use_ln=use_ln) for mlp in mlps
        ])

        self.out_channels = sum(m[-1] for m in mlps)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """
        xyz: (N,3)
        points: (N,Cin) or None
        return:
          new_xyz: (M,3)
          new_points: (M,Cout)
        """
        N = xyz.size(0)
        if self.npoint >= N:
            new_xyz = xyz
            fps_idx = torch.arange(N, device=xyz.device, dtype=torch.long)
        else:
            new_xyz, fps_idx = farthest_point_sampling(xyz, self.npoint)

        new_points_list: list[torch.Tensor] = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples, self.mlps):
            group_idx = ball_query(xyz, new_xyz, radius, nsample)   # (M,nsample)
            grouped_xyz = xyz[group_idx]                            # (M,nsample,3)
            grouped_xyz = grouped_xyz - new_xyz[:, None, :]         # relative xyz

            if points is None:
                grouped = grouped_xyz                               # (M,nsample,3)
            else:
                grouped_points = points[group_idx]                  # (M,nsample,Cin)
                grouped = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (M,nsample,3+Cin)

            feat = mlp(grouped)                                     # (M,nsample,C')
            feat = torch.max(feat, dim=1).values                    # (M,C')
            new_points_list.append(feat)

        new_points = torch.cat(new_points_list, dim=-1)             # (M,sum(C'))
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation (FP), batchless
      3-NN interpolation + skip concat + shared MLP
    """
    def __init__(self, in_channels: int, mlp: list[int], use_ln: bool = False):
        super().__init__()
        self.mlp = PointMLP(in_channels, mlp, use_ln=use_ln)
        self.out_channels = mlp[-1]

    def forward(
        self,
        xyz_fine: torch.Tensor,          # (N,3)
        xyz_coarse: torch.Tensor,        # (M,3)
        feat_fine: torch.Tensor | None,  # (N,C1) skip (can be None)
        feat_coarse: torch.Tensor,       # (M,C2)
    ):
        """
        return: new_feat_fine (N, Cout)
        """
        # interpolate coarse -> fine
        interp = three_nn_interpolate(xyz_coarse, feat_coarse, xyz_fine)  # (N,C2)

        if feat_fine is None:
            cat = interp
        else:
            cat = torch.cat([feat_fine, interp], dim=-1)                 # (N,C1+C2)

        out = self.mlp(cat)                                              # (N,Cout)
        return out


# ---------------------------
# "Correct" PointNet++ (SA + FP)
# ---------------------------
class PointNetPP(nn.Module):
    """
    正しい PointNet++ (batchless):
      SA1(MSG) -> SA2(MSG) -> SA3(global/SSG) -> FP3 -> FP2 -> FP1 -> head
    入力 : (N,3)
    出力 : (N,out_dim)
    """
    def __init__(
        self,
        out_dim: int = 48,
        quantize: bool = True,
        use_ln: bool = False,
        npoint1: int = 512,
        npoint2: int = 128,
    ):
        super().__init__()
        self.quantize = quantize

        # ---- SA layers ----
        # SA1: xyz only
        self.sa1 = SetAbstractionMSG(
            npoint=npoint1,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 64],
            in_channels=0,
            mlps=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ],
            use_ln=use_ln,
        )
        c1 = self.sa1.out_channels  # 64+128+128 = 320

        # SA2: from SA1 features
        self.sa2 = SetAbstractionMSG(
            npoint=npoint2,
            radii=[0.2, 0.4, 0.8],
            nsamples=[16, 32, 64],
            in_channels=c1,
            mlps=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ],
            use_ln=use_ln,
        )
        c2 = self.sa2.out_channels  # 128+256+256 = 640

        # SA3: global (npoint=1) っぽくする（全点を1つの集合にして maxpool）
        # ここは MSG ではなく簡易 global abstraction にする（PointNet++標準の最終層の形）
        self.global_mlp = PointMLP(3 + c2, [256, 512, 1024], use_ln=use_ln)  # per-point then maxpool
        c3 = 1024

        # ---- FP layers ----
        # FP3: (xyz2,feat2) + (global) -> feat2'
        #   interp global(1点) -> xyz2 へ (実質同じ特徴が複製される)
        self.fp3 = FeaturePropagation(in_channels=c2 + c3, mlp=[512, 512], use_ln=use_ln)
        c2p = self.fp3.out_channels  # 512

        # FP2: xyz1 へ
        self.fp2 = FeaturePropagation(in_channels=c1 + c2p, mlp=[256, 256], use_ln=use_ln)
        c1p = self.fp2.out_channels  # 256

        # FP1: xyz0(元のN点) へ（元は特徴なしなので skip は None）
        self.fp1 = FeaturePropagation(in_channels=c1p, mlp=[256, 128], use_ln=use_ln)
        c0p = self.fp1.out_channels  # 128

        # ---- Head ----
        self.head = nn.Sequential(
            nn.Linear(c0p, 128),
            nn.ReLU(True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,3)
        return: (N,out_dim)
        """
        assert x.dim() == 2 and x.size(1) == 3, f"Expected (N,3), got {tuple(x.shape)}"
        xyz0 = x
        feat0 = None

        # SA1
        xyz1, feat1 = self.sa1(xyz0, feat0)      # (M1,3), (M1,c1)
        # SA2
        xyz2, feat2 = self.sa2(xyz1, feat1)      # (M2,3), (M2,c2)

        # SA3 (global): per-point MLP on (relative xyz? ここでは xyz2 をそのまま入れて良い) then max
        # grouped = [xyz2, feat2]
        g_in = torch.cat([xyz2, feat2], dim=-1)  # (M2, 3+c2)
        g_feat = self.global_mlp(g_in)           # (M2,1024)
        global_feat = torch.max(g_feat, dim=0).values  # (1024,)

        # global point as xyz3 = (1,3) (ダミーで原点)、feat3=(1,1024)
        xyz3 = torch.zeros((1, 3), device=x.device, dtype=x.dtype)
        feat3 = global_feat.unsqueeze(0)         # (1,1024)

        # FP3: propagate global -> xyz2, with skip feat2
        feat2p = self.fp3(xyz_fine=xyz2, xyz_coarse=xyz3, feat_fine=feat2, feat_coarse=feat3)  # (M2,512)
        # FP2: xyz2 -> xyz1
        feat1p = self.fp2(xyz_fine=xyz1, xyz_coarse=xyz2, feat_fine=feat1, feat_coarse=feat2p) # (M1,256)
        # FP1: xyz1 -> xyz0
        feat0p = self.fp1(xyz_fine=xyz0, xyz_coarse=xyz1, feat_fine=None, feat_coarse=feat1p)  # (N,128)

        out = torch.tanh(self.head(feat0p))  # (N,out_dim)

        if self.quantize:
            out = STE_binary.apply(out)

        return out


# ---------------------------
# quick test
# # ---------------------------
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     N = 3000
#     x = torch.randn(N, 3, device=device)

#     net = PointNetPP_Correct(out_dim=48, quantize=True, use_ln=False, npoint1=512, npoint2=128).to(device)
#     y = net(x)
#     print("out:", y.shape, y.min().item(), y.max().item())
