# Rebuilding

Both need a **complete** CUDA toolkit at build time, not the pip
`nvidia-cuda-*` packages, which ship headers but no `libcudart.so` dev symlink,
so linking fails with `cannot find -lcudart`.

Build in a throwaway venv that has the target `torch` installed. Use
`--no-build-isolation` so the extension compiles against that torch. Put the
venv's `bin` on `PATH`: torch's extension builder only parallelizes when it can
find the `ninja` executable, and without it the flash-attn build runs three
compile jobs at a time (hours instead of minutes).

```bash
export CUDA_HOME=/usr/local/cuda-13.0
uv venv venv --python 3.12
uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
uv pip install --python venv/bin/python pip setuptools wheel packaging ninja psutil
export PATH=$PWD/venv/bin:$CUDA_HOME/bin:$PATH

# flash-attn: sm_90 + sm_100. Measured 5 min at MAX_JOBS=96 on an idle 192-core
# box with 2 TB RAM (peak use 420 GB). setup.py estimates 9 GB per job.
git clone --depth 1 -b v2.8.3.post1 --recurse-submodules --shallow-submodules \
  https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=96 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
  FLASH_ATTN_CUDA_ARCHS="90;100" \
  python -m pip wheel . --no-build-isolation --no-deps -w ../wheelhouse
cd ..

# causal-conv1d. Its setup.py hardcodes the arch list from the toolkit version
# (CUDA 13.0 gives sm_75 through sm_121) and ignores TORCH_CUDA_ARCH_LIST.
git clone --depth 1 -b v1.6.2.post1 https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
MAX_JOBS=32 CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE \
  python -m pip wheel . --no-build-isolation --no-deps -w ../wheelhouse
```

Confirm the archs before publishing:

```bash
unzip -o wheelhouse/flash_attn-*.whl 'flash_attn_2_cuda*.so' -d elf
cuobjdump --list-elf elf/flash_attn_2_cuda*.so | sed -E 's/.*\.(sm_[0-9a-z]+)\..*/\1/' | sort | uniq -c
```
