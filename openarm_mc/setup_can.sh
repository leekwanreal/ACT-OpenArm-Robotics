#!/usr/bin/env bash
#
# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.
#
# Bring up the CAN interfaces OpenArm needs, before ROS is sourced.
#
#   ./setup_can.sh sim      a virtual bus for the MuJoCo simulator
#   ./setup_can.sh real     the robot's two buses
#   ./setup_can.sh both     (default) whichever of the two is possible
#   ./setup_can.sh status   print what is up, change nothing
#   ./setup_can.sh down     take the interfaces down again
#
# Interfaces survive until the machine reboots, so this is a once-per-boot
# step rather than something to run before every launch. It is safe to run
# again: an interface that is already configured is left alone.
#
# Only the `ip` calls need root. Run it as yourself and it will ask for sudo
# when it has something to change.

set -u

VCAN=${OPENARM_VCAN:-vcan0}
REAL_BUSES=(${OPENARM_CAN_BUSES:-can0 can1})

# What the robot's adapter runs at, from openarm_can's install_can_autosetup.sh.
BITRATE=${OPENARM_CAN_BITRATE:-1000000}
DBITRATE=${OPENARM_CAN_DBITRATE:-5000000}
RESTART_MS=${OPENARM_CAN_RESTART_MS:-100}
TXQUEUELEN=${OPENARM_CAN_TXQUEUELEN:-1000}

# A CAN-FD frame carries up to 64 data bytes, which is an MTU of 72. A link
# left at the classic 16 silently refuses every FD frame, and openarm_can
# speaks FD, so this is not optional.
FD_MTU=72

red=$'\e[31m'; green=$'\e[32m'; yellow=$'\e[33m'; dim=$'\e[2m'; off=$'\e[0m'
ok()   { printf '  %s✓%s %s\n' "$green" "$off" "$1"; }
warn() { printf '  %s!%s %s\n' "$yellow" "$off" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$red" "$off" "$1"; }
note() { printf '    %s%s%s\n' "$dim" "$1" "$off"; }

exists()  { ip link show "$1" >/dev/null 2>&1; }
is_up()   { ip -br link show "$1" 2>/dev/null | grep -qw UP; }
mtu_of()  { ip -d link show "$1" 2>/dev/null | sed -n 's/.*mtu \([0-9]*\).*/\1/p' | head -1; }

# Run a privileged command, asking for sudo only when we are not already root.
as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

setup_vcan() {
  printf '\nVirtual bus for the simulator\n'

  if ! lsmod | grep -q '^vcan'; then
    as_root modprobe vcan || { bad "cannot load the vcan kernel module"; return 1; }
    ok "loaded the vcan module"
  fi

  if exists "$VCAN"; then
    ok "$VCAN already exists"
  else
    as_root ip link add dev "$VCAN" type vcan || { bad "cannot create $VCAN"; return 1; }
    ok "created $VCAN"
  fi

  local mtu
  mtu=$(mtu_of "$VCAN")
  if [ "$mtu" != "$FD_MTU" ]; then
    # MTU cannot change while the link is up.
    is_up "$VCAN" && as_root ip link set "$VCAN" down
    as_root ip link set "$VCAN" mtu "$FD_MTU" || { bad "cannot set MTU on $VCAN"; return 1; }
    ok "set $VCAN MTU to $FD_MTU, so CAN-FD frames fit"
  fi

  is_up "$VCAN" || as_root ip link set "$VCAN" up
  ok "$VCAN is up"
}

setup_one_real() {
  local bus=$1

  if ! exists "$bus"; then
    warn "$bus is not there"
    note "the adapter creates it when plugged in: check lsusb and dmesg | tail"
    return 1
  fi

  # Bitrate and FD mode can only be set while the link is down.
  is_up "$bus" && as_root ip link set "$bus" down

  if ! as_root ip link set "$bus" type can \
      bitrate "$BITRATE" dbitrate "$DBITRATE" fd on restart-ms "$RESTART_MS"; then
    bad "cannot configure $bus"
    note "is it a real CAN device? vcan has no bitrate"
    return 1
  fi
  as_root ip link set "$bus" txqueuelen "$TXQUEUELEN"
  as_root ip link set "$bus" up

  ok "$bus up at ${BITRATE}/${DBITRATE} FD, restart-ms $RESTART_MS, txqueuelen $TXQUEUELEN"
}

setup_real() {
  printf '\nRobot buses\n'
  local any=1
  for bus in "${REAL_BUSES[@]}"; do
    setup_one_real "$bus" && any=0
  done
  if [ "$any" -ne 0 ]; then
    warn "no robot bus came up; the simulator does not need one"
  fi
  return 0
}

take_down() {
  printf '\nTaking interfaces down\n'
  for bus in "$VCAN" "${REAL_BUSES[@]}"; do
    if exists "$bus" && is_up "$bus"; then
      as_root ip link set "$bus" down && ok "$bus down"
    fi
  done
  if exists "$VCAN"; then
    as_root ip link delete "$VCAN" && ok "removed $VCAN"
  fi
}

status() {
  printf '\nCAN interfaces\n'
  local found=1
  for bus in "$VCAN" "${REAL_BUSES[@]}"; do
    exists "$bus" || continue
    found=0
    local state mtu
    state=$(ip -br link show "$bus" | awk '{print $2}')
    mtu=$(mtu_of "$bus")
    if is_up "$bus" && [ "$mtu" = "$FD_MTU" ]; then
      ok "$bus  $state  mtu $mtu"
    elif is_up "$bus"; then
      warn "$bus  $state  mtu $mtu  (classic CAN only; FD frames will fail)"
    else
      warn "$bus  $state  mtu $mtu"
    fi
  done
  [ "$found" -eq 0 ] || warn "no CAN interface exists yet"

  printf '\nNext\n'
  note "source /opt/ros/${ROS_DISTRO:-$(ls /opt/ros 2>/dev/null | head -n 1)}/setup.bash"
  note "source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install/setup.bash"
  note "ros2 launch bringup bringup.launch.py enable_sim:=true use_sim:=true"
}

case "${1:-both}" in
  sim)    setup_vcan; status ;;
  real)   setup_real; status ;;
  both)   setup_vcan; setup_real; status ;;
  status) status ;;
  down)   take_down; status ;;
  *)
    printf 'usage: %s [sim|real|both|status|down]\n' "$(basename "$0")" >&2
    exit 2
    ;;
esac
