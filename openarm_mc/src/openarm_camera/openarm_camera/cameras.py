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

"""Which cameras OpenArm has, where their frames go, and which is which.

Two RealSense RGB-D cameras, each by its role:

    chest   on the robot's chest, looking at the table (the policies' camera
            since the start: the datasets and the models know it by this name)
    front   across the table, looking back at the robot

Every camera's frames travel the same way:

    driver        /camera/<name>/color/image_raw
                  /camera/<name>/aligned_depth_to_color/image_raw
    preprocessor  /openarm/camera/<name>/rgb     rgb8, 640x480
                  /openarm/camera/<name>/depth   16UC1 mm, 640x480, paired

and the recorder, the dashboard and the policies read the second pair.

A camera that is not plugged in is stood in for by a virtual one
(virtual_camera.py): the same topics, size and rate, every frame black and
every depth pixel 0 (no reading), stamped with frame_id <name>_virtual so
whoever reads it can tell. The rest of the stack runs as with the real
thing -- in the simulator, say, with no camera at all.

Which physical camera has which role is the serial number, kept in
cameras.yaml at the workspace root -- machine-specific, so not in git --
which ./setup_camera.sh writes:

    cameras:
      chest:
        serial: "123456789012"
      front:
        serial: "234567890123"

No ROS here: setup_camera.sh, the launch files and the tests all read it.
"""

import glob
import os
import subprocess
from pathlib import Path

# The roles, in the order they are listed and recorded; the first is the
# one a single-camera policy reads.
CAMERAS = ("chest", "front")
TOPIC_PREFIX = "/openarm/camera"
KINDS = ("rgb", "depth")
# A virtual camera's frame_id ends with this.
VIRTUAL_SUFFIX = "_virtual"
# Where a camera's frames come from.
SOURCES = ("realsense", "virtual", "external")


def is_virtual(frame_id):
    """Whether a frame came from a virtual (black) camera."""
    return str(frame_id).endswith(VIRTUAL_SUFFIX)


def rgb_topic(name, prefix=TOPIC_PREFIX):
    return f"{prefix.rstrip('/')}/{name}/rgb"


def depth_topic(name, prefix=TOPIC_PREFIX):
    return f"{prefix.rstrip('/')}/{name}/depth"


def raw_topics(name):
    """The RealSense driver's colour and aligned-depth topics for `name`,
    as rs_launch.py names them with camera_namespace camera."""
    return (f"/camera/{name}/color/image_raw",
            f"/camera/{name}/aligned_depth_to_color/image_raw")


def parse_names(text):
    """"chest,front" (or spaces) -> ("chest", "front"); raises on an
    unknown role or a repeated one. Empty -> ()."""
    names = tuple(part for part in str(text).replace(",", " ").split() if part)
    unknown = [name for name in names if name not in CAMERAS]
    if unknown:
        raise ValueError(f"unknown camera {unknown[0]!r}; the cameras are "
                         f"{', '.join(CAMERAS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"a camera is named twice in {text!r}")
    # Always in CAMERAS order, so chest is first whichever way it was typed.
    return tuple(name for name in CAMERAS if name in names)


def workspace():
    """The workspace root, through this file's real path: a symlink-install
    build keeps it in src/openarm_camera/openarm_camera/."""
    return Path(os.path.realpath(__file__)).parents[3]


def config_path():
    """$OPENARM_CAMERAS_FILE, else <workspace>/cameras.yaml."""
    return Path(os.environ.get("OPENARM_CAMERAS_FILE")
                or workspace() / "cameras.yaml")


def load(path=None):
    """{role: {"serial": str}} from cameras.yaml, roles in CAMERAS order;
    {} when there is no file yet. Raises ValueError on a malformed one."""
    path = Path(path) if path else config_path()
    if not path.is_file():
        return {}
    import yaml

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not YAML ({error})") from error
    entries = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: expected a `cameras:` mapping")
    cameras = {}
    for name in CAMERAS:
        entry = entries.get(name)
        if entry is None:
            continue
        if not isinstance(entry, dict) or not str(entry.get("serial") or "").strip():
            raise ValueError(f"{path}: cameras.{name} needs a serial")
        # YAML reads an unquoted serial as a number, and a leading zero
        # would be lost on the way: keep what was written, as text.
        cameras[name] = {"serial": str(entry["serial"]).strip()}
    unknown = sorted(set(entries) - set(CAMERAS))
    if unknown:
        raise ValueError(f"{path}: unknown camera {unknown[0]!r}; the cameras "
                         f"are {', '.join(CAMERAS)}")
    serials = [entry["serial"] for entry in cameras.values()]
    if len(set(serials)) != len(serials):
        raise ValueError(f"{path}: two cameras have the same serial")
    return cameras


HEADER = """\
# OpenArm's cameras: which RealSense is which, by serial number.
# Written by ./setup_camera.sh; machine-specific, not in git.
#   chest  on the robot's chest (the policies' camera)
#   front  across the table, looking back at the robot
"""


def save(config, path=None):
    """Write {role: {"serial": str}} to cameras.yaml, roles in CAMERAS
    order; atomically, so a reader never sees half a file."""
    path = Path(path) if path else config_path()
    entries = [(name, str(config[name]["serial"]).strip()) for name in CAMERAS
               if name in config and str(config[name].get("serial") or "").strip()]
    unknown = sorted(set(config) - set(CAMERAS))
    if unknown:
        raise ValueError(f"unknown camera {unknown[0]!r}")
    if len({serial for _, serial in entries}) != len(entries):
        raise ValueError("two cameras have the same serial")
    lines = ["cameras: {}"] if not entries else ["cameras:"] + [
        line for name, serial in entries
        for line in (f"  {name}:", f'    serial: "{serial}"')]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(HEADER + "\n".join(lines) + "\n")
    os.replace(tmp, path)


def swap(path=None):
    """Chest becomes front and front chest, in cameras.yaml; -> the new
    roles. For when the two cameras were assigned the wrong way round."""
    config = load(path)
    swapped = {}
    if "front" in config:
        swapped["chest"] = config["front"]
    if "chest" in config:
        swapped["front"] = config["chest"]
    save(swapped, path)
    return swapped


def parse_devices(text):
    """rs-enumerate-devices -S output -> [serial], in its order.

    Its "Device info:" blocks hold "    <Field>   : \t<value>" lines and end
    at a blank line; a line before the first block (the summary table, or
    "No device detected") is not one of them.
    """
    serials, block = [], False
    for line in str(text).splitlines():
        if line.startswith("Device info"):
            block = True
        elif not line.strip():
            block = False
        elif block and ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "Serial Number" and value.strip():
                serials.append(value.strip())
    return serials


def plugged_serials(command=None, timeout=10.0):
    """The serial numbers of the RealSense cameras plugged in; None when
    they cannot be listed (no rs-enumerate-devices). $OPENARM_RS_ENUMERATE
    names another command, as for setup_camera.sh."""
    command = command or os.environ.get("OPENARM_RS_ENUMERATE",
                                        "rs-enumerate-devices")
    path = os.environ.get("PATH", "")
    libraries = os.environ.get("LD_LIBRARY_PATH", "")
    # The ROS packages install the tool there (humble, jazzy), and
    # librealsense2.so beside it (Jazzy: lib/<arch>/); ROS need not be
    # sourced -- without the library the tool fails to start and every
    # camera would look unplugged.
    for ros_bin in sorted(glob.glob("/opt/ros/*/bin")):
        path += os.pathsep + ros_bin
    for ros_lib in sorted(glob.glob("/opt/ros/*/lib") + glob.glob("/opt/ros/*/lib/*-linux-gnu")):
        libraries += os.pathsep + ros_lib
    try:
        result = subprocess.run([command, "-S"], capture_output=True, text=True,
                                timeout=timeout,
                                env=dict(os.environ, PATH=path,
                                         LD_LIBRARY_PATH=libraries.strip(os.pathsep)))
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_devices(result.stdout)


def plan(requested="auto", virtual="auto", path=None, plugged=None):
    """What to start: [(role, serial, source)], source one of SOURCES.

    `requested`: auto (both roles) or a list ("chest", "chest,front").
    `virtual`: auto -- a camera whose RealSense is not plugged in is a
    virtual (black) one; none -- start only real ones, a missing camera
    gets nothing; all -- every camera virtual; or a list of the roles to
    make virtual. `plugged`: the serials plugged in (plugged_serials());
    None when they could not be listed, which counts as none.

    A role without a serial in cameras.yaml is the one RealSense plugged in
    when it is the only role asked for a real camera, or when it is chest
    and there is no cameras.yaml yet (the one-camera setup from before);
    otherwise it cannot tell which camera is its, and is virtual.
    """
    config = load(path)
    names = CAMERAS if str(requested).strip().lower() in ("", "auto") \
        else parse_names(requested)
    mode = str(virtual).strip().lower() or "auto"
    forced = set()
    if mode == "all":
        forced = set(names)
    elif mode not in ("auto", "none"):
        forced = set(parse_names(virtual))
    plugged = list(plugged or [])

    chosen = []
    unassigned = [s for s in plugged
                  if s not in {entry["serial"] for entry in config.values()}]
    for name in names:
        serial = config.get(name, {}).get("serial", "")
        if name in forced:
            chosen.append((name, serial, "virtual"))
            continue
        if serial:
            real = serial in plugged
        else:
            # No serial: only when there is no doubt which camera it is,
            # and then by its serial, not "the first one the driver finds".
            alone = not config and name == "chest" and len(plugged) == 1
            only = len(names) == 1 and len(unassigned) == 1
            real = alone or only
            if real:
                serial = plugged[0] if alone else unassigned[0]
        if real or mode == "none":
            if real:
                chosen.append((name, serial, "realsense"))
            continue
        chosen.append((name, serial, "virtual"))
    return chosen
