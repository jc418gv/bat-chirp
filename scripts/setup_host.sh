#!/usr/bin/env bash
set -euo pipefail

# Generic Linux host setup for this repo. Override versions or wheel index with
# environment variables if your host needs a different Python or CUDA stack.

BATPIPE_BATDETECT2_SPEC="${BATPIPE_BATDETECT2_SPEC:-batdetect2==2.0.0b1}"
BATPIPE_TORCH_INDEX_URL="${BATPIPE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
BATPIPE_TORCH_SPEC="${BATPIPE_TORCH_SPEC:-torch==2.5.1}"
BATPIPE_TORCHAUDIO_SPEC="${BATPIPE_TORCHAUDIO_SPEC:-torchaudio==2.5.1}"
BATPIPE_FSSPEC_SPEC="${BATPIPE_FSSPEC_SPEC:-fsspec<2026.0}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

python -m pip install --upgrade pip

# Install the pinned Torch stack first so BatDetect2 resolves against the
# intended CUDA/PyTorch runtime instead of briefly pulling a newer default.
python -m pip install --force-reinstall \
  --index-url "$BATPIPE_TORCH_INDEX_URL" \
  "$BATPIPE_TORCH_SPEC" \
  "$BATPIPE_TORCHAUDIO_SPEC"

python -m pip install "$BATPIPE_FSSPEC_SPEC"
python -m pip install -e . "$BATPIPE_BATDETECT2_SPEC"

if ! command -v batdetect2 >/dev/null 2>&1; then
  echo "batdetect2 was not installed into $(pwd)/.venv/bin" >&2
  echo "Try: python -m pip install \"$BATPIPE_BATDETECT2_SPEC\"" >&2
  exit 1
fi

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

echo "batdetect2 $(command -v batdetect2)"