#!/usr/bin/env bash
set -euo pipefail

# Generic Linux host setup for this repo. Override versions or wheel index with
# environment variables if your host needs a different Python or CUDA stack.

BATPIPE_BATDETECT2_SPEC="${BATPIPE_BATDETECT2_SPEC:-batdetect2==2.0.0b1}"
BATPIPE_TORCH_INDEX_URL="${BATPIPE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
BATPIPE_TORCH_SPEC="${BATPIPE_TORCH_SPEC:-torch==2.5.1}"
BATPIPE_TORCHAUDIO_SPEC="${BATPIPE_TORCHAUDIO_SPEC:-torchaudio==2.5.1}"

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e . "$BATPIPE_BATDETECT2_SPEC"

# If your host is CPU-only or uses a different CUDA runtime, override the wheel
# index and package specs before running this script.
python -m pip install --force-reinstall \
  --index-url "$BATPIPE_TORCH_INDEX_URL" \
  "$BATPIPE_TORCH_SPEC" \
  "$BATPIPE_TORCHAUDIO_SPEC"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY