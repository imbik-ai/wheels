# wheels

Prebuilt GPU kernel wheels for [imbik-v2](https://github.com/imbik-ai/imbik-v2).

This repo exists only to host binaries. Upstream publishes no wheels for our
torch/CUDA line, and `uv lock` needs a credential-free URL to hash, which a
private repo's release assets cannot provide (they are API-only). Hence a public
host for artifacts that are, in any case, just compiled BSD-3 sources.

## kernels-cu13-torch2.11-r2

Built for: **torch 2.11.0+cu130, CUDA 13.0, CPython 3.12, cxx11abi=TRUE**.
Runs on Hopper (H100/H200, sm_90) and datacenter Blackwell (B200, sm_100).

| wheel | source | GPU code | verified against |
| --- | --- | --- | --- |
| `flash_attn-2.8.3.post1` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) @ `v2.8.3.post1` (BSD-3) | sm_90, sm_100 | `scaled_dot_product_attention`, bf16, causal, (2, 512, 8, 64): max abs diff `1.95e-03` on H200, `7.81e-03` on B200 |
| `causal_conv1d-1.6.2.post1` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) @ `v1.6.2.post1` (BSD-3) | sm_75 to sm_121 (every arch CUDA 13.0 knows) | exact match vs grouped `conv1d` on a padded input, on H200 and B200 |

The flash-attn wheel is arch-specific to keep its size and build time down. To
add an arch, rebuild with a wider `FLASH_ATTN_CUDA_ARCHS` (see `build.md`).
The causal-conv1d wheel is the same binary as in `kernels-cu13-torch2.11`: its
`setup.py` picks archs from the CUDA toolkit version and ignores
`TORCH_CUDA_ARCH_LIST`, so the first build already covered Blackwell.

Check what a wheel contains with `cuobjdump --list-elf <extension>.so`.

## kernels-cu13-torch2.11 (superseded)

Same sources and toolchain. Its `flash_attn` wheel carries sm_90 only and fails
on B200 with `no kernel image is available for execution on the device`.

Rebuild commands are in `build.md`.
