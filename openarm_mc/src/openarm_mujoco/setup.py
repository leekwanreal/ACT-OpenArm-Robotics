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

from collections import defaultdict
from pathlib import Path

from setuptools import find_namespace_packages, setup


_PROJECT_ROOT = Path(__file__).resolve().parent
_BUNDLED_VERSION_DIRS = ("v1", "v2")
_EXCLUDED_PARTS = {
    ".venv",
    "__pycache__",
    "openarm_mujoco.egg-info",
    "openarm_mujoco_v2",
}
_ALLOWED_SUFFIXES = {
    ".jpeg",
    ".jpg",
    ".mjcf",
    ".mtl",
    ".obj",
    ".png",
    ".stl",
    ".xml",
}


def _version_dirs() -> list[Path]:
    return [
        path
        for version_dir in _BUNDLED_VERSION_DIRS
        if (path := _PROJECT_ROOT / version_dir).is_dir()
    ]


def _data_files() -> list[tuple[str, list[str]]]:
    files_by_target: dict[str, list[str]] = defaultdict(list)

    for root in _version_dirs():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            relative_path = path.relative_to(_PROJECT_ROOT)
            if any(
                part.startswith(".") or part in _EXCLUDED_PARTS
                for part in relative_path.parts
            ):
                continue

            target = Path("share") / "openarm_mujoco" / relative_path.parent
            files_by_target[str(target)].append(str(relative_path))

    return [
        (target, sorted(files)) for target, files in sorted(files_by_target.items())
    ]


def _ament_data_files() -> list[tuple[str, list[str]]]:
    """What colcon needs to see this as a ROS 2 package.

    The marker file registers the package in the ament index, package.xml
    makes it discoverable, and the launch files let `ros2 launch` find it.
    Building with pip instead simply installs these few extra files.
    """
    share = str(Path("share") / "openarm_mujoco")
    launch_files = sorted(
        str(path.relative_to(_PROJECT_ROOT))
        for path in (_PROJECT_ROOT / "launch").glob("*.launch.py")
    )
    return [
        ("share/ament_index/resource_index/packages", ["resource/openarm_mujoco"]),
        (share, ["package.xml"]),
        (f"{share}/launch", launch_files),
    ]


setup(
    # pyproject.toml carries these for pip, but colcon's ament_python build
    # calls this file directly without reading it; without a name setuptools
    # installs the package as "UNKNOWN" and it never lands on the path.
    name="openarm_mujoco",
    version="2.3.0",
    # Namespace-aware discovery is kept so that a subpackage added without an
    # __init__.py still gets installed rather than silently dropped.
    # "share" is excluded because a --symlink-install build makes colcon write
    # its pythonpath_develop hook into src/share/openarm_mujoco/hook/ inside the
    # build directory, and the next build would otherwise discover that tree as
    # a namespace package.
    packages=find_namespace_packages(
        where="src", exclude=["*.egg-info*", "share", "share.*"]
    ),
    package_dir={"": "src"},
    data_files=_data_files() + _ament_data_files(),
    # Declared here rather than only in pyproject.toml so that colcon's
    # ament_python build installs them into lib/openarm_mujoco, which is
    # where `ros2 run` looks.
    entry_points={
        "console_scripts": [
            "can_bridge = openarm_mujoco.can_bridge:main",
            "launch_viewer = openarm_mujoco.v2.launch:main",
        ],
    },
)
