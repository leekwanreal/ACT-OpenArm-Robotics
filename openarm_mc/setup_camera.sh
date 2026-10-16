#!/usr/bin/env bash
#
# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.
#
# Set up OpenArm's RealSense cameras: the host, and which camera is which.
#
#   ./setup_camera.sh setup     (default) the host, then the roles: a camera
#                               already known keeps its role, a new one gets
#                               the free one, two new ones are told apart by
#                               unplugging one
#   ./setup_camera.sh status    print the cameras and their roles, change nothing
#   ./setup_camera.sh assign <chest|front> <serial>
#   ./setup_camera.sh switch    chest becomes front and front chest: in
#                               cameras.yaml, and at once in the cameras
#                               running (also: swap; the dashboard's Switch)
#   ./setup_camera.sh forget <chest|front>
#   ./setup_camera.sh plan      what the launch will start, a line per camera:
#                               "<role> realsense <serial>" or "<role> virtual"
#                               (run_main.sh reads it)
#   ./setup_camera.sh roles     the roles whose camera is plugged in, one line
#
# A camera that is not plugged in still runs, as a virtual one: black frames
# on its topics, so the recorder, the dashboard and the policies work without
# it (in the simulator, say). `status` says which.
#
# Two cameras, by role:
#   chest   on the robot's chest -- the policies' camera, and the datasets'
#   front   across the table, looking back at the robot
#
# The roles go to cameras.yaml at the workspace root ($OPENARM_CAMERAS_FILE
# names another), by serial number: the launch files start each camera's
# driver on its own serial (src/openarm_camera/openarm_camera/cameras.py).
#
# The host part, needed once per machine (udev) or once per boot (usbfs):
#   udev rules   librealsense's, so the cameras open without root
#   usbfs        a 1000 MB USB buffer: two RealSense cameras streaming through
#                libusb overrun the kernel's default 16 MB
#
# Run it as yourself; it asks for sudo only when it has something to change.

set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${OPENARM_CAMERAS_FILE:-$ROOT/cameras.yaml}
ROLES=(chest front)
USBFS_MB=${OPENARM_USBFS_MB:-1000}
UDEV_RULES=/etc/udev/rules.d/99-realsense-libusb.rules
USBFS_PARAM=/sys/module/usbcore/parameters/usbfs_memory_mb
# The ROS distro: the one sourced, else the one installed (humble on 22.04,
# jazzy on 24.04).
ROS_DISTRO=${ROS_DISTRO:-$(ls /opt/ros 2>/dev/null | head -n 1)}
ROS_DISTRO=${ROS_DISTRO:-humble}
# The ROS packages install the RealSense tools here; ROS need not be sourced.
[ -d "/opt/ros/$ROS_DISTRO/bin" ] && PATH=$PATH:/opt/ros/$ROS_DISTRO/bin
RS_ENUMERATE=${OPENARM_RS_ENUMERATE:-rs-enumerate-devices}
# ... and librealsense2.so beside them (Jazzy: lib/<arch>/), which only a
# sourced ROS puts on the library path: without it the tool fails to start
# and every camera would look unplugged.
for ros_lib in "/opt/ros/$ROS_DISTRO/lib" "/opt/ros/$ROS_DISTRO/lib/$(uname -m)-linux-gnu"; do
  [ -d "$ros_lib" ] && LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ros_lib
done
export LD_LIBRARY_PATH

red=$'\e[31m'; green=$'\e[32m'; yellow=$'\e[33m'; bold=$'\e[1m'; dim=$'\e[2m'; off=$'\e[0m'
ok()   { printf '  %s✓%s %s\n' "$green" "$off" "$1"; }
warn() { printf '  %s!%s %s\n' "$yellow" "$off" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$red" "$off" "$1"; }
note() { printf '    %s%s%s\n' "$dim" "$1" "$off"; }

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

is_role() { [ "$1" = chest ] || [ "$1" = front ]; }

# ------------------------------------------------------------------ cameras

# One line per RealSense plugged in: serial|name|usb|port|firmware.
# From rs-enumerate-devices -S, whose "Device info:" blocks hold
# "    <Field>   : \t<value>" lines; the port's value has colons of its own.
devices() {
  "$RS_ENUMERATE" -S 2>/dev/null | awk '
    function flush() {
      if (sn != "") print sn "|" name "|" usb "|" port "|" fw
      sn = name = usb = port = fw = ""
    }
    /^Device info/ { flush(); block = 1; next }
    block && /^[[:space:]]*$/ { flush(); block = 0; next }
    block && index($0, ":") {
      key = substr($0, 1, index($0, ":") - 1)
      value = substr($0, index($0, ":") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      if (key == "Name") name = value
      else if (key == "Serial Number") sn = value
      else if (key == "Usb Type Descriptor") usb = value
      else if (key == "Firmware Version") fw = value
      else if (key == "Physical Port") {
        # /sys/devices/.../usb2/2-2/2-2:1.0/video4linux/video0 -> 2-2
        port = value
        if (match(value, /usb[0-9]+\/[0-9][0-9.-]*\//)) {
          port = substr(value, RSTART, RLENGTH - 1)
          sub(/usb[0-9]+\//, "", port)
        }
      }
    }
    END { flush() }'
}

serials() { devices | cut -d'|' -f1; }

# What camera.launch.py will start, through the same cameras.plan():
# "<role> <source> <serial>" lines.
plan() {
  local plugged
  plugged=$(serials | paste -sd, -)
  config_py "
plugged = [s for s in '$plugged'.split(',') if s]
for name, serial, source in cameras.plan('auto', 'auto', plugged=plugged):
    print(name, source, serial if source == 'realsense' else '')" 2>/dev/null
}

plugged() { serials | grep -qx -- "$1"; }

# ------------------------------------------------------------------- config

# cameras.yaml through openarm_camera/cameras.py, the same reader the launch
# files use, so the two can never disagree about it.
config_py() {
  OPENARM_CAMERAS_FILE=$CONFIG python3 -c "
import sys
sys.path.insert(0, '$ROOT/src/openarm_camera')
from openarm_camera import cameras
$1"
}

# The serial cameras.yaml gives a role, or nothing.
serial_of() {
  config_py "print(cameras.load().get('$1', {}).get('serial', ''))" 2>/dev/null
}

# Which role a serial has, or nothing.
role_of() {
  local role
  for role in "${ROLES[@]}"; do
    [ "$(serial_of "$role")" = "$1" ] && { echo "$role"; return; }
  done
}

config_ok() {
  local problem
  problem=$(config_py "cameras.load()" 2>&1 >/dev/null) && return 0
  bad "$CONFIG is broken: ${problem##*: }"
  note "fix it by hand, or remove it and run ./setup_camera.sh again"
  return 1
}

# Write the roles: write_config <chest serial> <front serial> (either empty).
write_config() {
  config_py "cameras.save({k: {'serial': v} for k, v in (('chest', '$1'), ('front', '$2')) if v})"
}

set_role() {      # set_role <role> <serial or empty>
  local chest front
  chest=$(serial_of chest); front=$(serial_of front)
  # A serial has one role: taking it elsewhere frees the old one.
  [ -n "$2" ] && [ "$chest" = "$2" ] && chest=""
  [ -n "$2" ] && [ "$front" = "$2" ] && front=""
  case "$1" in
    chest) chest=$2 ;;
    front) front=$2 ;;
  esac
  write_config "$chest" "$front"
}

# The running cameras swapped too: each preprocessor reads the other
# camera's driver (its `source` parameter), so the switch shows at once,
# with no driver restarted. Nothing to do when the cameras are not running.
switch_running() {
  pgrep -f "__node:=openarm_chest_preprocessor" >/dev/null || return 0
  pgrep -f "__node:=openarm_front_preprocessor" >/dev/null || {
    warn "only chest's preprocessor runs (front is virtual?): restart the cameras to apply"
    return 0
  }
  (
    # ROS's setup scripts read variables that may be unset.
    set +u
    # shellcheck disable=SC1091
    source "/opt/ros/$ROS_DISTRO/setup.bash" >/dev/null 2>&1 || exit 1
    [ -f "$ROOT/install/setup.bash" ] && source "$ROOT/install/setup.bash" >/dev/null 2>&1
    chest=$(ros2 param get /openarm_chest_preprocessor source 2>/dev/null | awk '{print $NF}')
    front=$(ros2 param get /openarm_front_preprocessor source 2>/dev/null | awk '{print $NF}')
    [ -n "$chest" ] && [ -n "$front" ] || exit 1
    ros2 param set /openarm_chest_preprocessor source "$front" >/dev/null &&
      ros2 param set /openarm_front_preprocessor source "$chest" >/dev/null
  ) && ok "switched in the running cameras too" \
    || warn "could not switch the running cameras: restart them to apply"
}

# --------------------------------------------------------------------- host

librealsense_version() {
  dpkg-query -W -f='${Version}' "ros-$ROS_DISTRO-librealsense2" 2>/dev/null | cut -d- -f1
}

setup_udev() {
  if [ -f "$UDEV_RULES" ]; then
    ok "udev rules installed ($UDEV_RULES)"
    return 0
  fi
  local version url tmp
  version=$(librealsense_version)
  if [ -z "$version" ]; then
    bad "no ros-$ROS_DISTRO-librealsense2 to take udev rules from"
    note "sudo apt install ros-$ROS_DISTRO-realsense2-camera"
    return 1
  fi
  # librealsense's own rules, for the version installed.
  url="https://raw.githubusercontent.com/IntelRealSense/librealsense/v$version/config/99-realsense-libusb.rules"
  tmp=$(mktemp)
  if ! curl -fsSL "$url" -o "$tmp" 2>/dev/null || ! grep -q '8086' "$tmp"; then
    rm -f "$tmp"
    bad "cannot download librealsense $version's udev rules"
    note "$url"
    return 1
  fi
  as_root install -m 0644 "$tmp" "$UDEV_RULES" && rm -f "$tmp" || { rm -f "$tmp"; bad "cannot install $UDEV_RULES"; return 1; }
  as_root udevadm control --reload-rules && as_root udevadm trigger
  ok "installed librealsense $version's udev rules"
  note "unplug the cameras and plug them back in for the rules to apply"
}

setup_usbfs() {
  local current
  current=$(cat "$USBFS_PARAM" 2>/dev/null || echo "")
  if [ -z "$current" ]; then
    warn "cannot read $USBFS_PARAM"
    return 0
  fi
  if [ "$current" -ge "$USBFS_MB" ] 2>/dev/null; then
    ok "usbfs buffer ${current} MB"
    return 0
  fi
  if echo "$USBFS_MB" | as_root tee "$USBFS_PARAM" >/dev/null; then
    ok "usbfs buffer ${current} -> ${USBFS_MB} MB, until the next reboot"
    note "to keep it: add usbcore.usbfs_memory_mb=$USBFS_MB to GRUB_CMDLINE_LINUX_DEFAULT"
  else
    bad "cannot raise the usbfs buffer (${current} MB)"
  fi
}

setup_host() {
  printf '\nHost\n'
  if ! command -v "$RS_ENUMERATE" >/dev/null 2>&1; then
    bad "$RS_ENUMERATE not found"
    note "sudo apt install ros-$ROS_DISTRO-realsense2-camera"
    return 1
  fi
  ok "librealsense $(librealsense_version)"
  setup_udev
  setup_usbfs
}

# -------------------------------------------------------------------- roles

# Ask the operator to unplug a camera; -> the serial that went, once the
# list has shrunk by exactly one. Gives up after two minutes.
wait_for_unplug() {
  local before=$1 deadline=$((SECONDS + 120)) now gone
  while [ "$SECONDS" -lt "$deadline" ]; do
    sleep 1
    now=$(serials)
    gone=$(comm -23 <(echo "$before" | sort) <(echo "$now" | sort))
    if [ -n "$gone" ] && [ "$(echo "$gone" | wc -l)" -eq 1 ]; then
      echo "$gone"
      return 0
    fi
  done
  return 1
}

wait_for_plug() {
  local serial=$1 deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    plugged "$serial" && return 0
    sleep 1
  done
  return 1
}

assign_new() {
  printf '\nRoles\n'
  config_ok || return 1
  local all new=() serial role free=()
  all=$(serials)
  for serial in $all; do
    [ -z "$(role_of "$serial")" ] && new+=("$serial")
  done
  for role in "${ROLES[@]}"; do
    serial=$(serial_of "$role")
    if [ -z "$serial" ]; then
      free+=("$role")
    elif ! echo "$all" | grep -qx -- "$serial"; then
      warn "$role ($serial) is not plugged in; its role is kept"
    fi
  done

  if [ "${#new[@]}" -eq 0 ]; then
    [ -n "$all" ] && ok "every camera plugged in has a role"
    return 0
  fi
  if [ "${#free[@]}" -eq 0 ]; then
    for serial in "${new[@]}"; do
      warn "camera $serial has no role, and both are taken"
    done
    note "./setup_camera.sh assign <chest|front> <serial> gives it one"
    return 0
  fi
  if [ "${#new[@]}" -eq 1 ]; then
    # One new camera: it takes the free role, chest first.
    set_role "${free[0]}" "${new[0]}"
    ok "camera ${new[0]} is now ${bold}${free[0]}${off}"
    [ "${#free[@]}" -eq 2 ] && note "a second camera plugged in later becomes front"
    return 0
  fi

  # Two new cameras look the same: tell them apart by unplugging one.
  if [ ! -t 0 ]; then
    warn "two new cameras (${new[*]}): which is which?"
    note "./setup_camera.sh assign chest <serial>, then assign front <serial>"
    return 0
  fi
  printf '    Two new cameras: %s and %s. Which is which?\n' "${new[0]}" "${new[1]}"
  printf '    %sUnplug the FRONT camera%s (the one across the table) now...\n' "$bold" "$off"
  local front chest
  if ! front=$(wait_for_unplug "$all"); then
    bad "no camera was unplugged; nothing changed"
    note "or: ./setup_camera.sh assign chest <serial>"
    return 1
  fi
  chest=$( [ "$front" = "${new[0]}" ] && echo "${new[1]}" || echo "${new[0]}")
  set_role chest "$chest"
  set_role front "$front"
  ok "chest is $chest, front is $front"
  printf '    %sPlug the front camera back in%s...\n' "$bold" "$off"
  if wait_for_plug "$front"; then
    ok "front is back"
  else
    warn "front ($front) is still unplugged; its role is kept"
  fi
}

# ------------------------------------------------------------------- status

status() {
  printf '\nRealSense cameras\n'
  local list line serial name usb port fw role
  list=$(devices)
  if [ -z "$list" ]; then
    warn "none plugged in"
    note "check lsusb | grep 8086, and that the cable is USB 3"
  fi
  while IFS='|' read -r serial name usb port fw; do
    [ -z "$serial" ] && continue
    role=$(role_of "$serial")
    line="$serial  $name  USB $usb  port $port  fw $fw"
    if [ -z "$role" ]; then
      warn "$line  -- no role: ./setup_camera.sh"
    elif [ "${usb%%.*}" -lt 3 ] 2>/dev/null; then
      warn "${bold}$role${off}  $line  -- USB 2: too slow for 640x480 RGB-D at 30 fps"
    else
      ok "${bold}$role${off}  $line"
    fi
  done <<< "$list"

  printf '\nRoles (%s)\n' "$CONFIG"
  if [ ! -f "$CONFIG" ]; then
    warn "no file yet: ./setup_camera.sh"
    note "without one, the chest camera is whichever single RealSense is plugged in"
  elif config_ok; then
    for role in "${ROLES[@]}"; do
      serial=$(serial_of "$role")
      if [ -z "$serial" ]; then
        note "$role: none"
      elif echo "$list" | cut -d'|' -f1 | grep -qx -- "$serial"; then
        ok "$role: $serial"
      else
        warn "$role: $serial, not plugged in"
      fi
    done
  fi

  printf '\nAt launch (camera.launch.py, run_main.sh)\n'
  local source
  while read -r role source serial; do
    [ -z "$role" ] && continue
    if [ "$source" = realsense ]; then
      ok "${bold}$role${off}  RealSense $serial"
    else
      warn "${bold}$role${off}  virtual: black frames on its topics (no RealSense for it)"
    fi
  done <<< "$(plan)"

  printf '\nHost\n'
  [ -f "$UDEV_RULES" ] && ok "udev rules installed" \
    || warn "no udev rules: ./setup_camera.sh setup"
  local mb
  mb=$(cat "$USBFS_PARAM" 2>/dev/null || echo 0)
  if [ "$mb" -ge "$USBFS_MB" ] 2>/dev/null; then
    ok "usbfs buffer $mb MB"
  else
    warn "usbfs buffer $mb MB: enough for one camera, not two (./setup_camera.sh setup)"
  fi

  printf '\nNext\n'
  note "./run_main.sh, or: ros2 launch openarm_camera camera.launch.py"
  [ -n "$list" ] || note "plug the cameras in (USB 3), then ./setup_camera.sh"
}

# The roles whose camera is plugged in, in role order; with no cameras.yaml,
# chest for a single RealSense (the setup from before).
roles() {
  local list role serial out=()
  list=$(serials)
  if [ ! -f "$CONFIG" ]; then
    [ "$(echo "$list" | grep -c .)" -eq 1 ] && echo chest
    return 0
  fi
  for role in "${ROLES[@]}"; do
    serial=$(serial_of "$role")
    [ -n "$serial" ] && echo "$list" | grep -qx -- "$serial" && out+=("$role")
  done
  echo "${out[*]}"
}

# ----------------------------------------------------------------- commands

usage() {
  printf 'usage: %s [setup|status|assign <chest|front> <serial>|switch|forget <chest|front>|plan|roles]\n' \
    "$(basename "$0")" >&2
  exit 2
}

# Sourced (the tests do), it only defines the functions.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0

case "${1:-setup}" in
  setup)
    setup_host
    assign_new
    status ;;
  status)
    status ;;
  assign)
    [ $# -eq 3 ] && is_role "$2" || usage
    config_ok || exit 1
    plugged "$3" || warn "camera $3 is not plugged in; assigning it anyway"
    set_role "$2" "$3"
    ok "$2 is now $3"
    status ;;
  switch|swap)
    config_ok || exit 1
    [ -f "$CONFIG" ] || { bad "no $CONFIG yet: ./setup_camera.sh first"; exit 1; }
    config_py "cameras.swap()" || exit 1
    ok "chest is now $(serial_of chest), front $(serial_of front)"
    switch_running
    status ;;
  forget)
    [ $# -eq 2 ] && is_role "$2" || usage
    config_ok || exit 1
    set_role "$2" ""
    ok "$2 has no camera now"
    status ;;
  plan)
    plan ;;
  roles)
    roles ;;
  -h|--help)
    sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//' ;;
  *)
    usage ;;
esac
