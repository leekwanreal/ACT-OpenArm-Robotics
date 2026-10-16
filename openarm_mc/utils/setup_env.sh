#!/usr/bin/env bash
#
# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.
#
# Set up onnxruntime for this machine: the CUDA build when an NVIDIA GPU
# answers, the CPU build otherwise. Both install the same `onnxruntime`
# module, so exactly one of them goes in.
#
#   ./utils/setup_env.sh            the uv environment (utils/export_onnx.py)
#   ./utils/setup_env.sh --system   also the system Python the ROS nodes run
#                                   on (openarm_act, and msgpack for
#                                   openarm_smolvla), with "numpy<2" so
#                                   cv_bridge keeps working
#   --cpu | --gpu                   skip the detection
#
# A GPU counts when `nvidia-smi -L` lists one: the driver is loaded and
# talks. onnxruntime-gpu also needs the CUDA 12 and cuDNN 9 runtime
# libraries; without them it falls back to the CPU, and openarm_act says so.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
kind=auto
system=false
for arg in "$@"; do
  case "$arg" in
    --cpu) kind=cpu ;;
    --gpu) kind=gpu ;;
    --system) system=true ;;
    -h|--help) sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [ "$kind" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    kind=gpu
    echo "NVIDIA GPU found: $(nvidia-smi -L | head -1)"
  else
    kind=cpu
    echo "no NVIDIA GPU answering: the CPU build"
  fi
fi
# onnxruntime-gpu ships wheels for x86_64 only (Linux and Windows); on ARM,
# a Jetson among them, there is nothing to install and uv sync would fail.
if [ "$kind" = gpu ] && [ "$(uname -m)" != x86_64 ]; then
  echo "onnxruntime-gpu has no wheel for $(uname -m): the CPU build instead"
  kind=cpu
fi
package=$([ "$kind" = gpu ] && echo onnxruntime-gpu || echo onnxruntime)

uv=$(command -v uv || echo "$HOME/.local/bin/uv")
if [ ! -x "$uv" ]; then
  echo "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
echo "uv environment: uv sync --extra $kind ($package)"
"$uv" sync --project "$ROOT" --extra "$kind"

if $system; then
  # The other build first: two of them in one site-packages overwrite each
  # other's files and leave neither working.
  other=$([ "$kind" = gpu ] && echo onnxruntime || echo onnxruntime-gpu)
  if /usr/bin/python3 -m pip show "$other" >/dev/null 2>&1; then
    echo "system Python: removing $other"
    /usr/bin/python3 -m pip uninstall -y "$other"
  fi
  echo "system Python: $package, msgpack"
  /usr/bin/python3 -m pip install --user "$package" "numpy<2" "msgpack>=1.0"
fi

check() {
  "$1" -c 'import onnxruntime as o; print(f"  {o.__version__}:", ", ".join(o.get_available_providers()))'
}
echo "uv environment:"; check "$ROOT/.venv/bin/python"
if $system; then echo "system Python:"; check /usr/bin/python3; fi
