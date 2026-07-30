# wheels

Prebuilt GPU kernel wheels for [imbik-v2](https://github.com/imbik-ai/imbik-v2).

This repo exists only to host binaries. Upstream publishes no wheels for our
torch/CUDA line, and `uv lock` needs a credential-free URL to hash — which a
private repo's release assets cannot provide (they are API-only). Hence a public
host for artifacts that are, in any case, just compiled BSD-3 sources.

## kernels-cu13-torch2.11

Built for: **torch 2.11.0+cu130, CUDA 13.0, CPython 3.12, cxx11abi=TRUE, sm90a (H200)**.

| wheel | source | verified against |
| --- | --- | --- |
| `flash_attn-2.8.3.post1` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) @ `v2.8.3.post1` (BSD-3) | `3.91e-03` max abs diff vs `scaled_dot_product_attention`, bf16 |
| `causal_conv1d-1.6.2.post1` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) @ `v1.6.2.post1` (BSD-3) | exact match vs grouped `conv1d` on a padded input |

These are **arch-specific** (`sm90a` only) to keep build times sane. They will not
run on non-Hopper GPUs; rebuild with a wider `FLASH_ATTN_CUDA_ARCHS` /
`TORCH_CUDA_ARCH_LIST` if that changes.

Rebuild commands are in `build.md`.
