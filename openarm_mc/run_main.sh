#!/usr/bin/env bash
#
# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.
#
# Run everything: CAN, build if needed, and the whole stack in one launch.
#
#   ./run_main.sh
#
# works out the rest from what is plugged in:
#   robot     can0 and can1 up -> the real arm; otherwise MuJoCo on vcan0
#   cameras   chest and front (./setup_camera.sh says which RealSense is
#             which): a driver and a preprocessor for each one plugged in, a
#             virtual (black) camera for each one that is not
#   ACT       a model at $OPENARM_ACT_MODEL or the newest checkpoint/onnx/**/*.onnx
#             (export one from checkpoint/pth/: uv run utils/export_onnx.py) -> the policy,
#             idle until Generate trajectory or Start on the dashboard
#
# and any of it can be forced:
#   --real | --sim         which robot
#   --no-camera            no cameras at all, not even virtual ones
#   --cameras <list>       only these: chest, front, chest,front
#   --act <model.onnx>     this model;  --no-act  none
#   --cpu                  run the model on the CPU
#   --build                colcon build first
#   --port <n>             dashboard port (8080)
#   -- <name>:=<value>     passed straight to ros2 launch
#
# The motors stay off until Connect on the dashboard, and the arm stays
# still until a command source is chosen there.

set -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The ROS distro sourced, else the one installed (humble on 22.04, jazzy on 24.04).
ROS_SETUP=/opt/ros/${ROS_DISTRO:-$(ls /opt/ros 2>/dev/null | head -n 1)}/setup.bash

red=$'\e[31m'; green=$'\e[32m'; yellow=$'\e[33m'; dim=$'\e[2m'; off=$'\e[0m'
step() { printf '\n%s==>%s %s\n' "$green" "$off" "$1"; }
ok()   { printf '  %s✓%s %s\n' "$green" "$off" "$1"; }
warn() { printf '  %s!%s %s\n' "$yellow" "$off" "$1"; }
die()  { printf '  %s✗%s %s\n' "$red" "$off" "$1" >&2; exit 1; }
note() { printf '    %s%s%s\n' "$dim" "$1" "$off"; }

usage() { sed -n '6,30p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------- arguments

robot=auto          # auto | real | sim
camera=auto         # auto | off
cameras=auto        # auto | chest | front | chest,front
act=auto            # auto | off | <path>
device=cuda
build=false
port=8080
extra=()

while [ $# -gt 0 ]; do
  case "$1" in
    --real)       robot=real ;;
    --sim)        robot=sim ;;
    --no-camera)  camera=off ;;
    --cameras)    cameras=${2:?--cameras needs a list: chest, front, chest,front}; shift ;;
    --act)        act=${2:?--act needs a model file (.onnx)}; shift ;;
    --no-act)     act=off ;;
    --cpu)        device=cpu ;;
    --build)      build=true ;;
    --port)       port=${2:?--port needs a number}; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; extra=("$@"); break ;;
    *)            usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT" || die "cannot enter $ROOT"

# ------------------------------------------------------------ one of each

# Two controllers on one bus, or two dashboards on one port, fail in ways
# that look like something else, and Ctrl-C on a launch does not always take
# its nodes with it.
step "Checking for a stack that is already running"
running=$(pgrep -af "[l]ib/openarm" || true)
if [ -n "$running" ]; then
  warn "OpenArm nodes are still running:"
  printf '%s\n' "$running" | sed 's/^/      /'
  note "stop them first:  pkill -f lib/openarm"
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":$port "; then
  die "port $port is taken -- another dashboard? Try --port <other>"
fi
ok "nothing running"

# ---------------------------------------------------------------------- CAN

# Before ROS is sourced, as setup_can.sh wants. It asks for sudo only when it
# has something to change, and leaves links that are configured alone.
step "CAN"
link_up() { ip -br link show "$1" 2>/dev/null | grep -qw UP; }
vcan_ok() { ip -d link show vcan0 2>/dev/null | grep -q "mtu 72"; }
robot_buses() { link_up can0 && link_up can1; }

if ! vcan_ok || { [ "$robot" != sim ] && ! robot_buses && ip link show can0 >/dev/null 2>&1; }; then
  "$ROOT/setup_can.sh" both || warn "setup_can.sh reported a problem; see above"
fi

if [ "$robot" = auto ]; then
  if robot_buses; then robot=real; else robot=sim; fi
fi
if [ "$robot" = real ]; then
  robot_buses || die "--real, but can0 and can1 are not both up (./setup_can.sh status)"
  ok "robot: the real arm on can0/can1"
else
  vcan_ok || die "the simulator needs vcan0 at MTU 72 (./setup_can.sh sim)"
  ok "robot: MuJoCo on vcan0 (no robot buses found)"
fi

# ------------------------------------------------------------------- camera

step "Cameras"
virtual_cameras=()
if [ "$camera" = off ]; then
  note "off -- the dashboard and robot still work; the policies see a mock frame"
else
  # What camera.launch.py will start, as setup_camera.sh works it out:
  # "<role> realsense <serial>" or "<role> virtual".
  while read -r role source serial; do
    [ -z "$role" ] && continue
    if [ "$cameras" != auto ] && ! [[ ",$cameras," == *",$role,"* ]]; then
      continue
    fi
    if [ "$source" = realsense ]; then
      ok "$role: RealSense $serial"
    else
      warn "$role: virtual -- black frames (no RealSense for it; ./setup_camera.sh status)"
      virtual_cameras+=("$role")
    fi
  done <<< "$("$ROOT/setup_camera.sh" plan 2>/dev/null)"
fi

# ---------------------------------------------------------------------- ROS

step "ROS 2"
[ -f "$ROS_SETUP" ] || die "ROS 2 Humble is not installed ($ROS_SETUP)"
if [ -n "${CONDA_PREFIX:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
  # The nodes need mujoco, h5py and websockets in the interpreter ROS uses.
  warn "a conda/venv environment is active; the nodes run on /usr/bin/python3"
  note "if a node fails to import something, run: conda deactivate"
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

if $build || [ ! -f "$ROOT/install/setup.bash" ]; then
  step "Building"
  colcon build --symlink-install || die "build failed"
fi
# shellcheck disable=SC1091
source "$ROOT/install/setup.bash"
ok "sourced ROS 2 Humble and the workspace"

# ---------------------------------------------------------------------- ACT

step "ACT policy"
if [ "$act" = auto ]; then
  act=${OPENARM_ACT_MODEL:-}
  if [ -z "$act" ]; then
    act=$(find "$ROOT/checkpoint/onnx" -name '*.onnx' -type f -printf '%T@ %p\n' \
            2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  fi
  [ -n "$act" ] || act=off
  auto_act=true
else
  auto_act=false
fi

# Starting without ACT is fine when it was only looked for; asked for by
# name, a problem with it stops the launch.
skip_act() {
  if $auto_act; then warn "$1 -- starting without ACT"; act=off
  else die "$1"; fi
}

if [ "$act" != off ]; then
  if ! act_path=$(realpath -e "$act" 2>/dev/null); then
    skip_act "no such model: $act"
  elif [[ "$act_path" != *.onnx ]]; then
    skip_act "$act is not an .onnx (export one: uv run utils/export_onnx.py)"
  elif ! /usr/bin/python3 -c "import onnxruntime" 2>/dev/null; then
    skip_act "onnxruntime is missing; install the build for this machine (GPU or CPU) with:
      ./utils/setup_env.sh --system"
  else
    act=$act_path
    if [ "$device" = cuda ] && ! /usr/bin/python3 -c \
        "import onnxruntime as o, sys; sys.exit('CUDAExecutionProvider' not in o.get_available_providers())" 2>/dev/null; then
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        warn "a GPU is here but this onnxruntime has no CUDA; the model runs on the CPU
      (./utils/setup_env.sh --system installs onnxruntime-gpu)"
      else
        note "no GPU: the model runs on the CPU"
      fi
      device=cpu
    fi
  fi
fi
if [ "$act" = off ]; then
  note "off -- put a model under checkpoint/onnx/ (uv run utils/export_onnx.py), or set OPENARM_ACT_MODEL"
else
  ok "$act on $device, idle; Generate trajectory or Start on the dashboard"
  if [ "$camera" = off ]; then
    warn "no camera: the policy sees a mock frame (gray, 600 mm), as infer_robot.py does"
  elif [[ " ${virtual_cameras[*]} " == *" chest "* ]]; then
    warn "chest is virtual: the policy sees black frames"
  fi
fi

# ------------------------------------------------------------------- launch

launch_args=("http_port:=$port")
if [ "$robot" = sim ]; then
  launch_args+=("use_sim:=true" "enable_sim:=true")
fi
if [ "$camera" = off ]; then
  launch_args+=("enable_camera:=false")
else
  launch_args+=("enable_camera:=true" "cameras:=$cameras")
fi
if [ "$act" != off ]; then
  launch_args+=("enable_act:=true" "checkpoint:=$act" "device:=$device")
else
  # bringup starts ACT by itself when it finds a model; off means off.
  launch_args+=("enable_act:=false")
fi

step "Launching"
note "ros2 launch bringup bringup.launch.py ${launch_args[*]} ${extra[*]}"
printf '\n  Dashboard: %shttp://localhost:%s%s\n' "$green" "$port" "$off"
note "press Connect to arm the motors, then pick a command source"
printf '\n'

exec ros2 launch bringup bringup.launch.py "${launch_args[@]}" "${extra[@]}"
