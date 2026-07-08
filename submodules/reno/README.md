# Vendored RENO (neural anchor-position codec)

This directory contains the parts of [RENO](https://github.com/NJUVISION/RENO)
(MIT License, see `LICENSE`) needed by `utils/reno_utils.py` to compress/decompress
anchor coordinates, so HAC-plus does not depend on an external clone of the RENO repo.

Vendored:
- `network.py`, `kit/nn.py`, `kit/op.py` — model definition and inference-time ops
- `model/{Ford,KITTIDetection,SemanticKITTI}/ckpt.pt` — pretrained checkpoints (~1.1 MB each)

Not vendored (training/eval-only scripts from upstream RENO, unused by HAC-plus):
`compress.py`, `decompress.py`, `dataset.py`, `eval.py`, `train.py`, `kit/io.py`, `third_party/`.

## Required external dependency: torchsparse

`network.py` depends on [torchsparse](https://github.com/mit-han-lab/torchsparse), which
has CUDA kernels and must be built from source (it's not vendored here):

```bash
apt-get install libsparsehash-dev
git clone https://github.com/mit-han-lab/torchsparse.git && cd torchsparse
python setup.py install

pip install torchac
```
