# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which model ACT starts with when none is named.

The same rule as run_main.sh, for bringup and act.launch.py: the file in
$OPENARM_ACT_MODEL, else the newest .onnx under checkpoint/onnx/ (in a
folder of its own or not), where utils/export_onnx.py writes it; the
training .pth files sit beside it in checkpoint/pth/. Kept free of ROS,
numpy and the runtimes, so a launch file can import it.
"""

import os
from pathlib import Path

MODEL_ENV = "OPENARM_ACT_MODEL"
# Found through this file: `colcon build --symlink-install` links the
# package back to src/, so this is <workspace>/checkpoint/onnx/. A copied
# install has no checkpoint/ above it; $OPENARM_ACT_MODEL names one then.
CHECKPOINT_DIR = Path(os.path.realpath(__file__)).parents[3] / "checkpoint" / "onnx"


def find_model():
    """The model's path, or "" when there is none."""
    named = os.environ.get(MODEL_ENV, "").strip()
    if named:
        return str(Path(named).expanduser())
    models = sorted(CHECKPOINT_DIR.rglob("*.onnx"),
                    key=lambda path: path.stat().st_mtime)
    return str(models[-1]) if models else ""


def list_models():
    """Every .onnx under checkpoint/onnx/, newest first, for the dashboard
    to pick from: [{path, name (relative to checkpoint/onnx/), mtime}]."""
    try:
        found = [(path.stat().st_mtime, path) for path in CHECKPOINT_DIR.rglob("*.onnx")
                 if path.is_file()]
    except OSError:
        return []
    found.sort(reverse=True)
    return [{"path": str(path), "name": str(path.relative_to(CHECKPOINT_DIR)),
             "mtime": mtime} for mtime, path in found]
