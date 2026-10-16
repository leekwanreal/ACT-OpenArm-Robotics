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

"""SocketCAN transport and Damiao wire format for the OpenArm motors.

The frame layouts, the fixed-point encodings and the motor table mirror
openarm_can (its C++ library and openarm_can/sim/can_bridge.py), so the bytes
on the bus are the ones a real DM8009/DM4340/DM4310 answers.
"""

import socket
import struct
import threading

SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", 101)
CAN_RAW_FD_FRAMES = getattr(socket, "CAN_RAW_FD_FRAMES", 5)

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SZ = struct.calcsize(CAN_FRAME_FMT)
CANFD_FRAME_FMT = "=IBBB1x64s"
CANFD_FRAME_SZ = 72
CAN_EFF_MASK = 0x1FFFFFFF

MGMT_ID = 0x7FF
MGMT_REFRESH = 0xCC
MGMT_READ_PARAM = 0x33
MGMT_WRITE_PARAM = 0x55

CMD_ENABLE = 0xFC
CMD_DISABLE = 0xFD
CMD_SET_ZERO = 0xFE
CMD_CLEAR_ERROR = 0xFB

MODE_MIT = 0x000
MODE_POSVEL = 0x100
MODE_VEL = 0x200
MODE_POSFORCE = 0x300

RID_CTRL_MODE = 10
CTRL_MODE_MIT = 1
CTRL_MODE_POSFORCE = 4

KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

# Gripper: linear finger stroke (m) <-> motor angle (rad), as openarm_can
# exposes it on its ROS topics and in its datasets.
GRIPPER_MAX_STROKE_M = 0.043
GRIPPER_OPEN_RAD = -1.20

JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]
JOINT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}

# The two motors that are grippers rather than arm joints: their state and
# their commands travel on their own topics, in metres of finger stroke.
LEFT_GRIPPER_ID = JOINT_NAME_TO_ID["left_gripper"]
RIGHT_GRIPPER_ID = JOINT_NAME_TO_ID["right_gripper"]


def double_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    val = max(x_min, min(x_max, x))
    return int((val - x_min) * ((1 << bits) - 1) / span)


def uint_to_double(x, x_min, x_max, bits):
    span = x_max - x_min
    return (float(x) / float((1 << bits) - 1)) * span + x_min


def gripper_stroke_to_rad(stroke_m, invert=False):
    """Finger stroke (m) -> gripper motor angle (rad)."""
    ratio = max(0.0, min(1.0, stroke_m / GRIPPER_MAX_STROKE_M))
    if invert:
        ratio = 1.0 - ratio
    return ratio * GRIPPER_OPEN_RAD


def gripper_rad_to_stroke(q_rad, invert=False):
    """Gripper motor angle (rad) -> finger stroke (m)."""
    ratio = max(0.0, min(1.0, q_rad / GRIPPER_OPEN_RAD))
    if invert:
        ratio = 1.0 - ratio
    return ratio * GRIPPER_MAX_STROKE_M


def gripper_rad_rate_to_stroke_rate(dq_rad, invert=False):
    """Gripper motor velocity (rad/s) -> finger stroke velocity (m/s)."""
    rate = dq_rad * GRIPPER_MAX_STROKE_M / GRIPPER_OPEN_RAD
    return -rate if invert else rate


class Motor:
    """Live state of one Damiao motor, plus where to reach it."""

    def __init__(self, motor_id, name, arm, joint_idx, motor_type,
                 send_id, recv_id, interface, p_max, v_max, t_max):
        self.id = motor_id
        self.name = name
        self.arm = arm
        self.joint_idx = joint_idx          # 1..7 arm joints, 8 gripper
        self.motor_type = motor_type
        self.send_id = send_id
        self.recv_id = recv_id
        self.interface = interface
        self.p_max = p_max
        self.v_max = v_max
        self.t_max = t_max

        # Per-joint gains, loaded from parameters. Arm joints use kp/kd in
        # their MIT frames; grippers use speed/effort in position-force.
        self.kp = 0.0
        self.kd = 0.0
        self.speed = 0.0
        self.effort = 0.0

        self.enabled = False
        self.error_code = 0
        self.q = 0.0                        # motor angle, rad
        self.dq = 0.0                       # rad/s
        self.tau = 0.0                      # Nm
        self.t_mos = 0.0
        self.t_rotor = 0.0
        self.last_feedback = 0.0            # monotonic seconds, 0 = never
        self.invert = False

    @property
    def is_gripper(self):
        return self.joint_idx == 8

    def joint_position(self):
        """Position in the unit the ROS topics use: rad, or m for a gripper."""
        return self.motor_to_joint(self.q)

    def joint_velocity(self):
        return self.motor_to_joint_rate(self.dq)

    def motor_to_joint(self, q_rad):
        """A motor angle in the topics' unit: rad, or m for a gripper."""
        return (gripper_rad_to_stroke(q_rad, self.invert)
                if self.is_gripper else q_rad)

    def motor_to_joint_rate(self, dq_rad):
        return (gripper_rad_rate_to_stroke_rate(dq_rad, self.invert)
                if self.is_gripper else dq_rad)

    def joint_to_motor_rad(self, value):
        """Inverse of joint_position, for an incoming command."""
        return (gripper_stroke_to_rad(value, self.invert)
                if self.is_gripper else value)


# Motor types and limits per joint, shared by both arms.
_LAYOUT = [
    ("DM8009", 12.5, 45.0, 54.0, "J1 (Shoulder Pitch)"),
    ("DM8009", 12.5, 45.0, 54.0, "J2 (Shoulder Roll)"),
    ("DM4340", 12.5, 10.0, 28.0, "J3 (Arm Twist)"),
    ("DM4340", 12.5, 10.0, 28.0, "J4 (Elbow Pitch)"),
    ("DM4310", 12.5, 30.0, 10.0, "J5 (Forearm Twist)"),
    ("DM4310", 12.5, 30.0, 10.0, "J6 (Wrist Pitch)"),
    ("DM4310", 12.5, 30.0, 10.0, "J7 (Wrist Roll)"),
    ("DM4310", 12.5, 30.0, 10.0, "Gripper"),
]


def build_motors(left_interface, right_interface, arms="both",
                 right_id_offset=0x00):
    """Build the motor table.

    On the real robot each arm has its own bus and both use send ids
    0x01..0x08, so `right_id_offset` stays 0. Put both arms on one bus (the
    openarm_can simulator does, on vcan0) and the right arm answers on
    0x21..0x28 instead: pass 0x20.
    """
    motors = {}
    sides = []
    if arms in ("both", "left"):
        sides.append(("left", 1, left_interface, 0x00))
    if arms in ("both", "right"):
        sides.append(("right", 9, right_interface, right_id_offset))

    for arm, id_base, interface, offset in sides:
        for idx, (motor_type, p_max, v_max, t_max, label) in enumerate(_LAYOUT):
            motor_id = id_base + idx
            motors[motor_id] = Motor(
                motor_id, f"{arm.capitalize()} {label}", arm, idx + 1,
                motor_type, 0x01 + idx + offset, 0x11 + idx + offset,
                interface, p_max, v_max, t_max)
    return motors


def encode_mit(q_des, dq_des, kp, kd, tau_ff, p_max, v_max, t_max):
    """Build an MIT impedance command frame, sent on the motor's send id."""
    q_uint = double_to_uint(q_des, -p_max, p_max, 16)
    dq_uint = double_to_uint(dq_des, -v_max, v_max, 12)
    kp_uint = double_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_uint = double_to_uint(kd, KD_MIN, KD_MAX, 12)
    tau_uint = double_to_uint(tau_ff, -t_max, t_max, 12)
    return bytes([
        (q_uint >> 8) & 0xFF,
        q_uint & 0xFF,
        (dq_uint >> 4) & 0xFF,
        ((dq_uint & 0x0F) << 4) | ((kp_uint >> 8) & 0x0F),
        kp_uint & 0xFF,
        (kd_uint >> 4) & 0xFF,
        ((kd_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F),
        tau_uint & 0xFF,
    ])


# The position-force frame carries speed and current as fixed-point values:
# speed in hundredths of a rad/s, current as a tenth of a percent of maximum.
POSFORCE_SPEED_SCALE = 100.0
POSFORCE_CURRENT_SCALE = 10000.0
POSFORCE_SPEED_MAX = 655.0        # rad/s, what the 16-bit field holds
POSFORCE_CURRENT_MAX = 1.0        # a fraction, 1.0 being the motor's limit


def encode_posforce(q_des, speed=10.0, effort=0.15):
    """Build a position-force frame, sent on send_id + MODE_POSFORCE.

    This is the mode the grippers run in, so `speed` (rad/s) and `effort`
    (a fraction of the motor's current limit) are their half of the PD set.
    The defaults are openarm_can's: 10 rad/s and a 15% cap, low enough that
    a closing gripper holds rather than crushes.
    """
    speed = max(0.0, min(POSFORCE_SPEED_MAX, float(speed)))
    effort = max(0.0, min(POSFORCE_CURRENT_MAX, float(effort)))
    return struct.pack("<fHH", float(q_des),
                       int(round(speed * POSFORCE_SPEED_SCALE)),
                       int(round(effort * POSFORCE_CURRENT_SCALE)))


def encode_special(command):
    """Enable, disable, set-zero and clear-error are 7x 0xFF then the code."""
    return bytes([0xFF] * 7 + [command & 0xFF])


def encode_refresh(send_id):
    """Ask a motor for its state without commanding it."""
    return bytes([send_id & 0xFF, (send_id >> 8) & 0xFF, MGMT_REFRESH,
                  0, 0, 0, 0, 0])


def encode_write_param(send_id, rid, value, as_uint=True):
    """Write one register, e.g. the control mode."""
    raw = struct.pack("<I", int(value)) if as_uint else struct.pack("<f", float(value))
    return bytes([send_id & 0xFF, (send_id >> 8) & 0xFF, MGMT_WRITE_PARAM,
                  rid & 0xFF]) + raw


def decode_state(data, p_max, v_max, t_max):
    """Decode a motor's feedback frame."""
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    return {
        "send_id": data[0] & 0x0F,
        "error_code": (data[0] >> 4) & 0x0F,
        "q": uint_to_double(q_uint, -p_max, p_max, 16),
        "dq": uint_to_double(dq_uint, -v_max, v_max, 12),
        "tau": uint_to_double(tau_uint, -t_max, t_max, 12),
        "t_mos": data[6],
        "t_rotor": data[7],
    }


class CanBus:
    """One SocketCAN interface, opened for CAN-FD when the link allows it."""

    def __init__(self, interface):
        self.interface = interface
        self.sock = None
        self.use_fd = True
        self.frames_tx = 0
        self.frames_rx = 0
        self._tx_lock = threading.Lock()

    def open(self):
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            sock.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        except OSError:
            # A link brought up without `mtu 72` only carries classic CAN.
            self.use_fd = False
        sock.bind((self.interface,))
        self.sock = sock

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send(self, can_id, data):
        if self.sock is None:
            return False
        if self.use_fd:
            frame = struct.pack(CANFD_FRAME_FMT, can_id, len(data), 0, 0,
                                data + b"\x00" * (64 - len(data)))
        else:
            frame = struct.pack(CAN_FRAME_FMT, can_id, len(data),
                                data + b"\x00" * (8 - len(data)))
        try:
            with self._tx_lock:
                self.sock.send(frame)
            self.frames_tx += 1
            return True
        except OSError:
            return False

    def recv(self):
        """Block for one frame; returns (can_id, data) or None."""
        if self.sock is None:
            return None
        try:
            raw = self.sock.recv(CANFD_FRAME_SZ)
        except OSError:
            return None
        if len(raw) == CANFD_FRAME_SZ:
            can_id, dlc, _flags, _res, payload = struct.unpack(CANFD_FRAME_FMT, raw)
        elif len(raw) == CAN_FRAME_SZ:
            can_id, dlc, payload = struct.unpack(CAN_FRAME_FMT, raw)
        else:
            return None
        self.frames_rx += 1
        return can_id & CAN_EFF_MASK, payload[:dlc]
