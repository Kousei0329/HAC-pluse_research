"""
RENO-based anchor position compression utilities.
Drop-in replacement for compress_gpcc / decompress_gpcc.

RENO (network.py, kit/, model/) is vendored under submodules/reno/, so
this module is self-contained within HAC-plus.
Original project: https://github.com/NJUVISION/RENO (MIT License).

Dependencies (must be installed in the active environment):
    pip install torchac
    # torchsparse: build from source, see submodules/reno/README.md
"""

import io as _io
import os as _os
import sys
import time as _time
from collections import defaultdict

import numpy as np
import torch

RENO_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                          'submodules', 'reno')

# Module-level cache: (ckpt_path, channels, kernel_size) -> Network
_net_cache: dict = {}

# Populated by the most recent compress_reno()/decompress_reno() call made with
# verbose=True. {stage_name: seconds}
_last_profile: dict = {}


def get_last_profile() -> dict:
    """Return the per-stage timing breakdown from the last verbose=True call."""
    return dict(_last_profile)


class _StageTimer:
    """Accumulates wall-clock time per named stage, synchronizing CUDA at each boundary."""

    def __init__(self):
        self.totals: dict = defaultdict(float)
        self._t0 = None

    def tic(self) -> None:
        torch.cuda.synchronize()
        self._t0 = _time.time()

    def toc(self, stage: str) -> None:
        torch.cuda.synchronize()
        self.totals[stage] += _time.time() - self._t0
        self._t0 = _time.time()

    def report(self, label: str) -> None:
        global _last_profile
        _last_profile = dict(self.totals)
        total = sum(self.totals.values())
        parts = ', '.join(f'{k}={v:.4f}' for k, v in self.totals.items())
        print(f'[RENO] {label} breakdown (s): {parts}, total={total:.4f}')


def ensure_path() -> None:
    """Put the vendored RENO root on sys.path so `import network` / `import kit.op` work."""
    if RENO_ROOT not in sys.path:
        sys.path.insert(0, RENO_ROOT)


def _setup_torchsparse() -> None:
    ensure_path()
    from torchsparse.nn import functional as F
    cfg = F.conv_config.get_default_conv_config()
    cfg.kmap_mode = "hashmap"
    F.conv_config.set_global_conv_config(cfg)


def get_reno_net(ckpt_path: str = f'{RENO_ROOT}/model/Ford/ckpt.pt',
                 channels: int = 32, kernel_size: int = 3):
    """Return the cached RENO network (initialised + warmed up) for external use."""
    return _get_net(ckpt_path, channels, kernel_size)


def _get_net(ckpt_path: str, channels: int = 32, kernel_size: int = 3):
    key = (ckpt_path, channels, kernel_size)
    if key not in _net_cache:
        _setup_torchsparse()
        from network import Network
        from torchsparse import SparseTensor
        net = Network(channels=channels, kernel_size=kernel_size)
        net.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
        net.cuda().eval()
        # Warm up: trigger JIT compilation of CUDA kernels before first timed call.
        # Without this, the first compress_reno() call pays ~10s compilation overhead
        # while decompress_reno() (which runs after) gets compiled kernels for free.
        with torch.no_grad():
            _wc = torch.randint(0, 2048, (2048, 3), dtype=torch.int32, device='cuda')
            _wc = torch.cat((_wc[:, :1] * 0, _wc), dim=-1)
            _wf = torch.ones((2048, 1), device='cuda')
            net(SparseTensor(coords=_wc, feats=_wf))
        torch.cuda.synchronize()
        _net_cache[key] = net
    return _net_cache[key]


def compress_reno(
    x: torch.Tensor,
    ckpt_path: str = f'{RENO_ROOT}/model/Ford/ckpt.pt',
    channels: int = 32,
    kernel_size: int = 3,
    verbose: bool = False,
) -> bytes:
    """Compress integer grid coordinates [N, 3] with RENO.

    x may contain negative values; coordinates are shifted to non-negative
    internally and the shift is stored in the bitstream header.

    If verbose=True, prints a per-stage timing breakdown (also retrievable
    afterwards via get_last_profile()).

    Returns raw bytes compatible with decompress_reno().
    """
    _setup_torchsparse()
    import torchac
    from torchsparse import SparseTensor
    import kit.op as op

    net = _get_net(ckpt_path, channels, kernel_size)
    net.eval()

    timer = _StageTimer() if verbose else None

    with torch.no_grad():
        if timer: timer.tic()

        x = x.long().cuda()

        # RENO requires non-negative coordinates.
        shift = torch.min(x, dim=0)[0]          # [3]  per-axis minimum
        x_shifted = (x - shift).int()           # [N, 3]  non-negative, posQ=1 (lossless)

        N = x_shifted.shape[0]
        # Prepend batch-index column (all zeros = single batch)
        coords = torch.cat((x_shifted[:, 0:1] * 0, x_shifted), dim=-1).int()
        feats = torch.ones((N, 1), dtype=torch.float, device='cuda')
        sparse_x = SparseTensor(coords=coords, feats=feats)

        if timer: timer.toc('setup')

        # ── Multi-scale downscaling via FOG ──────────────────────────────
        data_ls = []
        while True:
            sparse_x = net.fog(sparse_x)
            data_ls.append((sparse_x.coords.clone(), sparse_x.feats.clone()))
            if sparse_x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]  # index 0 = coarsest level

        if timer: timer.toc('fog')

        # ── Neural encoding with arithmetic coding ───────────────────────
        byte_stream_ls = []
        for depth in range(len(data_ls) - 1):
            x_C, x_O = data_ls[depth]
            gt_C, gt_O = data_ls[depth + 1]
            gt_C, gt_O = op.sort_CF(gt_C, gt_O)

            if timer: timer.tic()

            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels)
            cur = SparseTensor(coords=x_C, feats=x_F)
            cur = net.prior_resnet(cur)

            if timer: timer.toc('prior')

            x_up_C, x_up_F = net.fcg(x_C, x_O, cur.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            if timer: timer.toc('fcg')

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            if timer: timer.toc('target')

            gt_O_s0 = torch.remainder(gt_O, 16)
            gt_O_s1 = torch.div(gt_O, 16, rounding_mode='floor')

            prob_s0 = net.pred_head_s0(x_up.feats)
            prob_s1 = net.pred_head_s1(
                x_up.feats + net.pred_head_s1_emb(gt_O_s0[:, 0].long()))

            if timer: timer.toc('pred_head')

            prob = torch.cat((prob_s0, prob_s1), dim=0)
            gt = torch.cat((gt_O_s0, gt_O_s1), dim=0)

            cdf = torch.cat((prob[:, 0:1] * 0, prob.cumsum(dim=-1)), dim=-1)
            cdf = torch.clamp(cdf, 0.0, 1.0)
            cdf_norm = op._convert_to_int_and_normalize(cdf, True).cpu()
            gt_cpu = gt[:, 0].to(torch.int16).cpu()

            if timer: timer.toc('cdf_prep')

            half = gt_cpu.shape[0] // 2
            byte_stream_ls.append(
                torchac.encode_int16_normalized_cdf(cdf_norm[:half], gt_cpu[:half]))
            byte_stream_ls.append(
                torchac.encode_int16_normalized_cdf(cdf_norm[half:], gt_cpu[half:]))

            if timer: timer.toc('torchac_enc')

        if timer: timer.tic()

        byte_stream = op.pack_byte_stream_ls(byte_stream_ls)

        base_C, base_F = data_ls[0]
        base_len = base_C.shape[0]
        base_C_np = base_C[:, 1:].cpu().numpy().astype(np.int32)   # drop batch dim
        base_F_np = base_F.cpu().numpy().astype(np.uint8)

        # ── Bitstream layout ─────────────────────────────────────────────
        # [shift: 3×int64 = 24 B] [posQ: float16 = 2 B]
        # [base_len: int32 = 4 B] [base_coords: base_len×3×int32]
        # [base_feats: base_len×uint8] [byte_stream]
        buf = _io.BytesIO()
        buf.write(shift.cpu().numpy().astype(np.int64).tobytes())   # 24 B
        buf.write(np.array(1, dtype=np.float16).tobytes())           # 2 B  posQ=1
        buf.write(np.array(base_len, dtype=np.int32).tobytes())      # 4 B
        buf.write(base_C_np.tobytes())                               # base_len*12 B
        buf.write(base_F_np.tobytes())                               # base_len*1 B
        buf.write(byte_stream)
        result = buf.getvalue()

        if timer:
            timer.toc('pack_io')
            timer.report('compress_reno')

        return result


def decompress_reno(
    strings: bytes,
    ckpt_path: str = f'{RENO_ROOT}/model/Ford/ckpt.pt',
    channels: int = 32,
    kernel_size: int = 3,
    verbose: bool = False,
) -> torch.Tensor:
    """Decompress bytes produced by compress_reno().

    If verbose=True, prints a per-stage timing breakdown (also retrievable
    afterwards via get_last_profile()), using the same stage names as
    compress_reno() so the two can be compared directly.

    Returns float32 tensor of shape [N, 3] containing the original
    integer grid coordinates (same dtype/range as compress_reno input).
    """
    _setup_torchsparse()
    import torchac
    from torchsparse import SparseTensor
    import kit.op as op

    net = _get_net(ckpt_path, channels, kernel_size)
    net.eval()

    timer = _StageTimer() if verbose else None

    with torch.no_grad():
        if timer: timer.tic()

        buf = _io.BytesIO(strings)

        # Parse header
        shift = torch.tensor(
            np.frombuffer(buf.read(24), dtype=np.int64), device='cuda')    # [3]
        posQ = int(np.frombuffer(buf.read(2), dtype=np.float16)[0])        # =1
        base_len = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
        base_C = torch.tensor(
            np.frombuffer(buf.read(base_len * 3 * 4), dtype=np.int32).reshape(-1, 3),
            device='cuda')
        base_F = torch.tensor(
            np.frombuffer(buf.read(base_len), dtype=np.uint8).reshape(-1, 1),
            device='cuda')
        byte_stream = buf.read()

        # Reconstruct coarsest SparseTensor
        x = SparseTensor(
            coords=torch.cat((base_F * 0, base_C), dim=-1),
            feats=base_F)
        byte_stream_ls = op.unpack_byte_stream(byte_stream)

        if timer: timer.toc('header_io')

        # ── Multi-stage arithmetic decoding ──────────────────────────────
        for idx in range(0, len(byte_stream_ls), 2):
            bs_s0 = byte_stream_ls[idx]
            bs_s1 = byte_stream_ls[idx + 1]

            if timer: timer.tic()

            x_O = x.feats.int()
            x.feats = net.prior_embedding(x_O).view(-1, net.channels)
            x = net.prior_resnet(x)

            if timer: timer.toc('prior')

            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            if timer: timer.toc('fcg')

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            if timer: timer.toc('target')

            prob_s0 = net.pred_head_s0(x_up.feats)

            if timer: timer.toc('pred_head')

            cdf_s0 = torch.cat((prob_s0[:, 0:1] * 0, prob_s0.cumsum(dim=-1)), dim=-1)
            cdf_s0 = torch.clamp(cdf_s0, 0.0, 1.0)
            cdf_s0_norm = op._convert_to_int_and_normalize(cdf_s0, True).cpu()

            if timer: timer.toc('cdf_prep')

            x_up_O_s0 = torchac.decode_int16_normalized_cdf(cdf_s0_norm, bs_s0).cuda()

            if timer: timer.toc('torchac_dec')

            prob_s1 = net.pred_head_s1(
                x_up.feats + net.pred_head_s1_emb(x_up_O_s0.long()))

            if timer: timer.toc('pred_head')

            cdf_s1 = torch.cat((prob_s1[:, 0:1] * 0, prob_s1.cumsum(dim=-1)), dim=-1)
            cdf_s1 = torch.clamp(cdf_s1, 0.0, 1.0)
            cdf_s1_norm = op._convert_to_int_and_normalize(cdf_s1, True).cpu()

            if timer: timer.toc('cdf_prep')

            x_up_O_s1 = torchac.decode_int16_normalized_cdf(cdf_s1_norm, bs_s1).cuda()

            if timer: timer.toc('torchac_dec')

            x_up_O = x_up_O_s1 * 16 + x_up_O_s0
            x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1))

        if timer: timer.tic()

        # Final coordinate expansion (FCG outputs [N, 4]: batch + xyz)
        scan = net.fcg(x.C, x.F)
        coords_shifted = (scan[:, 1:] * posQ).long()   # [N, 3] non-negative

        # Restore original coordinate range
        result = (coords_shifted + shift).float()

        if timer:
            timer.toc('fcg')
            timer.report('decompress_reno')

        return result
