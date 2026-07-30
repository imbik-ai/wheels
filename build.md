# Rebuilding

Both need a **complete** CUDA toolkit at build time — not the pip
`nvidia-cuda-*` packages, which ship headers but no `libcudart.so` dev symlink,
so linking fails with `cannot find -lcudart`.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

# flash-attn (~4 min at MAX_JOBS=64)
git clone --depth 1 -b v2.8.3.post1 --recurse-submodules --shallow-submodules \
  https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=64 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
  FLASH_ATTN_CUDA_ARCHS=90 \
  python -m pip wheel . --no-build-isolation --no-deps -w ../wheelhouse

# causal-conv1d
git clone --depth 1 -b v1.6.2.post1 https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
MAX_JOBS=32 CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE \
  TORCH_CUDA_ARCH_LIST="9.0a" \
  python -m pip wheel . --no-build-isolation --no-deps -w ../wheelhouse
```

Run both against a venv that already has the target `torch` installed —
`--no-build-isolation` is what makes them compile against *that* torch.
