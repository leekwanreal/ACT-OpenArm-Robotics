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

"""Answer Damiao CAN traffic with this model's physics.

The bridge binds a SocketCAN interface and impersonates the 16 Damiao motors
of a bimanual OpenArm: it decodes the command frames a CAN master sends,
drives the matching joint of the MuJoCo model, and replies with the feedback
frame the master expects. Anything that drives the real arm over CAN drives
the simulation instead, unchanged -- openarm_can's tools, or a ROS 2
controller publishing to it.

Unlike the browser page, which takes only keyboard and WebXR input, this runs
MuJoCo natively, so the physics are in the control loop: the arm sags under
gravity while the motors are disabled, and the master's kp/kd decide how well
it tracks.

It is a ROS 2 node, so it is launched and retuned like any other one, though
it publishes nothing: the CAN bus is what carries the robot's data.

    source /opt/ros/humble/setup.bash
    PYTHONPATH=src python3 -m openarm_mujoco.can_bridge
    ros2 param set /openarm_mujoco_sim realtime 0.25

A virtual bus is enough to try it without hardware:

    sudo ip link add dev vcan0 type vcan
    sudo ip link set vcan0 mtu 72
    sudo ip link set vcan0 up
"""

import os
import socket
import struct
import sys
import threading
import time

import mujoco
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node

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

KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

# The gripper motor turns through GRIPPER_OPEN_RAD to open the fingers by
# GRIPPER_MAX_STROKE_M, which is how openarm_can relates the two. In the model
# the finger is a slide joint in metres, so every gripper value crosses here.
GRIPPER_MAX_STROKE_M = 0.043
GRIPPER_OPEN_RAD = -1.20


def double_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    value = max(x_min, min(x_max, x))
    return int((value - x_min) * ((1 << bits) - 1) / span)


def uint_to_double(x, x_min, x_max, bits):
    span = x_max - x_min
    return (float(x) / float((1 << bits) - 1)) * span + x_min


def is_uint_param(rid):
    return (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36)


def gripper_rad_to_stroke(q_rad):
    ratio = max(0.0, min(1.0, q_rad / GRIPPER_OPEN_RAD))
    return ratio * GRIPPER_MAX_STROKE_M


def gripper_stroke_to_rad(stroke_m):
    ratio = max(0.0, min(1.0, stroke_m / GRIPPER_MAX_STROKE_M))
    return ratio * GRIPPER_OPEN_RAD


class Motor:
    """One simulated Damiao motor, bound to one joint of the model."""

    # Per joint: type, position/velocity/torque limits used by the wire format.
    LAYOUT = [
        ("DM8009", 12.5, 45.0, 54.0),
        ("DM8009", 12.5, 45.0, 54.0),
        ("DM4340", 12.5, 10.0, 28.0),
        ("DM4340", 12.5, 10.0, 28.0),
        ("DM4310", 12.5, 30.0, 10.0),
        ("DM4310", 12.5, 30.0, 10.0),
        ("DM4310", 12.5, 30.0, 10.0),
        ("DM4310", 12.5, 30.0, 10.0),
    ]

    def __init__(self, motor_id, arm, joint_idx, send_id, recv_id):
        motor_type, p_max, v_max, t_max = self.LAYOUT[joint_idx - 1]
        self.id = motor_id
        self.arm = arm
        self.joint_idx = joint_idx
        self.motor_type = motor_type
        self.send_id = send_id
        self.recv_id = recv_id
        self.p_max = p_max
        self.v_max = v_max
        self.t_max = t_max
        self.joint_name = (f"openarm_{arm}_finger_joint1" if joint_idx == 8
                           else f"openarm_{arm}_joint{joint_idx}")
        self.actuator_name = (f"{arm}_finger1_ctrl" if joint_idx == 8
                              else f"{arm}_joint{joint_idx}_ctrl")

        self.qpos_adr = -1
        self.dof_adr = -1
        self.ctrl_adr = -1
        self.offset = 0.0           # model value the motor calls zero

        self.enabled = False
        self.error_code = 0
        self.q_des = 0.0
        self.dq_des = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.tau_ff = 0.0

        self.q = 0.0                # motor angle, rad
        self.dq = 0.0
        self.tau = 0.0
        self.t_mos = 33.5
        self.t_rotor = 31.0

        self.params = {
            0: 12.0, 1: 0.12, 2: 95.0, 3: 0.9, 4: 250.0, 5: -250.0,
            6: float(v_max), 7: float(recv_id), 8: float(send_id),
            9: 1000.0, 10: 1.0, 11: 0.04, 12: 0.015, 13: 102.0, 14: 205.0,
            15: float(10000 + send_id), 16: 14.0, 17: 0.15, 18: 0.0003,
            19: 0.015, 20: 1.0, 21: float(p_max), 22: float(v_max),
            23: float(t_max), 24: 1000.0, 25: 5.0, 26: 0.1, 27: 10.0,
            28: 0.0, 35: float(send_id), 36: 1000000.0,
        }

    @property
    def is_gripper(self):
        return self.joint_idx == 8

    def model_to_motor(self, value):
        """Model joint value -> motor angle in rad."""
        if self.is_gripper:
            return gripper_stroke_to_rad(value)
        return value - self.offset

    def motor_to_model(self, q_rad):
        """Motor angle in rad -> model joint value."""
        if self.is_gripper:
            return gripper_rad_to_stroke(q_rad)
        return q_rad + self.offset

    def state_frame(self):
        """The 8-byte feedback a motor returns for every command."""
        q_uint = double_to_uint(self.q, -self.p_max, self.p_max, 16)
        dq_uint = double_to_uint(self.dq, -self.v_max, self.v_max, 12)
        tau_uint = double_to_uint(self.tau, -self.t_max, self.t_max, 12)
        return bytes([
            ((self.error_code & 0x0F) << 4) | (self.send_id & 0x0F),
            (q_uint >> 8) & 0xFF,
            q_uint & 0xFF,
            (dq_uint >> 4) & 0xFF,
            ((dq_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F),
            tau_uint & 0xFF,
            int(max(0, min(255, round(self.t_mos)))),
            int(max(0, min(255, round(self.t_rotor)))),
        ])

    def param_frame(self, rid):
        value = self.params.get(rid, 0.0)
        if is_uint_param(rid):
            raw = struct.pack("<I", int(value))
        else:
            raw = struct.pack("<f", float(value))
        return bytes([self.send_id & 0xFF, (self.send_id >> 8) & 0xFF,
                      MGMT_READ_PARAM, rid & 0xFF]) + raw


def build_motors(right_id_offset):
    """The 16 motors, left on send ids 0x01..0x08 and right above it.

    On the real robot each arm has its own bus and both use 0x01..0x08. One
    bus carrying both arms needs the right one moved, which openarm_can's
    simulator does by 0x20.
    """
    motors = {}
    for arm, id_base, offset in (("left", 1, 0x00),
                                 ("right", 9, right_id_offset)):
        for idx in range(8):
            motors[id_base + idx] = Motor(
                id_base + idx, arm, idx + 1,
                0x01 + idx + offset, 0x11 + idx + offset)
    return motors


class CanMotorBridge:
    """Serve a CAN bus from a stepping MuJoCo model."""

    def __init__(self, model_path, interface="vcan0", right_id_offset=0x20,
                 keyframe="home", control_mode="mit", open_socket=True):
        self.interface = interface
        self.control_mode = control_mode
        self.running = False
        self.sock = None
        self.use_fd = True
        self.frames_rx = 0
        self.frames_tx = 0
        # Physics steps, and the frames that actually commanded a motor
        # rather than only asking it for its state.
        self.steps = 0
        self.command_frames = 0
        # Changed while running, from ROS parameters.
        self.realtime = 1.0
        self.paused = False

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.motors = build_motors(right_id_offset)
        self.by_send_id = {m.send_id: m for m in self.motors.values()}

        self._resolve_indices()
        self._load_keyframe(keyframe)
        if control_mode == "mit":
            self._silence_arm_actuators()
        mujoco.mj_forward(self.model, self.data)
        self.read_state()

        if open_socket:
            self._open_socket()

    # ------------------------------------------------------------------ model

    def _resolve_indices(self):
        for motor in self.motors.values():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    motor.joint_name)
            if jid < 0:
                raise ValueError(
                    f"joint {motor.joint_name!r} is not in this model; "
                    f"the bridge expects a bimanual OpenArm")
            motor.qpos_adr = int(self.model.jnt_qposadr[jid])
            motor.dof_adr = int(self.model.jnt_dofadr[jid])
            motor.ctrl_adr = int(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor.actuator_name))

    def _load_keyframe(self, keyframe):
        if not keyframe:
            return
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            print(f"[bridge] no keyframe {keyframe!r}; starting from qpos 0")
            return
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        for motor in self.motors.values():
            if motor.ctrl_adr >= 0:
                self.data.ctrl[motor.ctrl_adr] = self.data.qpos[motor.qpos_adr]

    def _silence_arm_actuators(self):
        """Turn off the arm's position actuators so only CAN torque drives it.

        The grippers keep theirs: a gripper is commanded in position-force
        mode, which the model's position actuator reproduces directly.
        """
        for motor in self.motors.values():
            if motor.is_gripper or motor.ctrl_adr < 0:
                continue
            self.model.actuator_gainprm[motor.ctrl_adr, :3] = 0.0
            self.model.actuator_biasprm[motor.ctrl_adr, :3] = 0.0

    # ---------------------------------------------------------------- physics

    def apply_commands(self):
        for motor in self.motors.values():
            if motor.is_gripper:
                if motor.ctrl_adr >= 0:
                    target = (motor.motor_to_model(motor.q_des)
                              if motor.enabled
                              else self.data.qpos[motor.qpos_adr])
                    self.data.ctrl[motor.ctrl_adr] = target
                continue

            if self.control_mode == "position":
                if motor.ctrl_adr >= 0:
                    self.data.ctrl[motor.ctrl_adr] = (
                        motor.motor_to_model(motor.q_des) if motor.enabled
                        else self.data.qpos[motor.qpos_adr])
                continue

            if motor.enabled:
                # The torque the motor's own impedance loop would produce.
                tau = (motor.kp * (motor.q_des - motor.q)
                       + motor.kd * (motor.dq_des - motor.dq)
                       + motor.tau_ff)
                tau = max(-motor.t_max, min(motor.t_max, tau))
            else:
                tau = 0.0       # a disabled motor freewheels, so the arm sags
            motor.tau = tau
            self.data.qfrc_applied[motor.dof_adr] = tau

    def read_state(self):
        for motor in self.motors.values():
            motor.q = motor.model_to_motor(float(self.data.qpos[motor.qpos_adr]))
            dq_model = float(self.data.qvel[motor.dof_adr])
            if motor.is_gripper:
                # Stroke rate back to motor rate, through the same ratio.
                motor.dq = dq_model * GRIPPER_OPEN_RAD / GRIPPER_MAX_STROKE_M
                if motor.ctrl_adr >= 0:
                    motor.tau = float(self.data.actuator_force[motor.ctrl_adr])
            else:
                motor.dq = dq_model
                if self.control_mode == "position" and motor.ctrl_adr >= 0:
                    motor.tau = float(self.data.actuator_force[motor.ctrl_adr])
            motor.q = max(-motor.p_max, min(motor.p_max, motor.q))
            motor.dq = max(-motor.v_max, min(motor.v_max, motor.dq))

            load = abs(motor.tau) / max(1.0, motor.t_max)
            motor.t_mos += (load * 4.0 - (motor.t_mos - 30.0) * 0.08) * 0.01
            motor.t_rotor += (load * 2.8 - (motor.t_rotor - 28.0) * 0.08) * 0.01

    def step(self, count=1):
        for _ in range(count):
            self.apply_commands()
            mujoco.mj_step(self.model, self.data)
        self.read_state()

    # -------------------------------------------------------------------- CAN

    def _open_socket(self):
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            sock.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        except OSError:
            self.use_fd = False
        sock.bind((self.interface,))
        self.sock = sock

    def _send(self, can_id, payload, fd=True):
        if self.sock is None:
            return
        if fd:
            frame = struct.pack(CANFD_FRAME_FMT, can_id, len(payload), 0, 0,
                                payload + b"\x00" * (64 - len(payload)))
        else:
            frame = struct.pack(CAN_FRAME_FMT, can_id, len(payload),
                                payload + b"\x00" * (8 - len(payload)))
        try:
            self.sock.send(frame)
            self.frames_tx += 1
        except OSError:
            pass

    def handle_frame(self, can_id, data, is_fd=True):
        """Act on one command frame and reply the way a motor would."""
        if len(data) < 8:
            return

        if can_id == MGMT_ID:
            motor = self.by_send_id.get(data[0] | (data[1] << 8))
            if motor is None:
                return
            command = data[2]
            if command == MGMT_REFRESH:
                self._send(motor.recv_id, motor.state_frame(), is_fd)
            elif command == MGMT_READ_PARAM:
                self._send(motor.recv_id, motor.param_frame(data[3]), is_fd)
            elif command == MGMT_WRITE_PARAM:
                rid = data[3]
                fmt = "<I" if is_uint_param(rid) else "<f"
                motor.params[rid] = float(struct.unpack(fmt, data[4:8])[0])
                self._send(motor.recv_id, motor.param_frame(rid), is_fd)
            return

        motor = self.by_send_id.get(can_id & 0x0FF)
        if motor is None:
            return

        if data[0:7] == b"\xff" * 7:
            command = data[7]
            if command == CMD_ENABLE:
                motor.enabled = True
                motor.error_code = 1
                # Hold station where the arm is, so enabling does not snap it.
                motor.q_des = motor.q
                motor.dq_des = 0.0
            elif command == CMD_DISABLE:
                motor.enabled = False
                motor.error_code = 0
            elif command == CMD_SET_ZERO:
                if not motor.is_gripper:
                    motor.offset = float(self.data.qpos[motor.qpos_adr])
                motor.q_des = 0.0
                motor.dq_des = 0.0
                self.read_state()
            elif command == CMD_CLEAR_ERROR:
                motor.error_code = 1 if motor.enabled else 0
            self._send(motor.recv_id, motor.state_frame(), is_fd)
            return

        mode = can_id & 0xF00
        if mode == MODE_MIT:
            q_uint = (data[0] << 8) | data[1]
            dq_uint = (data[2] << 4) | (data[3] >> 4)
            kp_uint = ((data[3] & 0x0F) << 8) | data[4]
            kd_uint = (data[5] << 4) | (data[6] >> 4)
            tau_uint = ((data[6] & 0x0F) << 8) | data[7]
            motor.q_des = uint_to_double(q_uint, -motor.p_max, motor.p_max, 16)
            motor.dq_des = uint_to_double(dq_uint, -motor.v_max, motor.v_max, 12)
            motor.kp = uint_to_double(kp_uint, KP_MIN, KP_MAX, 12)
            motor.kd = uint_to_double(kd_uint, KD_MIN, KD_MAX, 12)
            motor.tau_ff = uint_to_double(tau_uint, -motor.t_max, motor.t_max, 12)
        elif mode == MODE_POSFORCE:
            motor.q_des = float(struct.unpack("<f", data[0:4])[0])
        elif mode == MODE_POSVEL:
            motor.q_des = float(struct.unpack("<f", data[0:4])[0])
            motor.dq_des = float(struct.unpack("<f", data[4:8])[0])
            motor.kp, motor.kd = 35.0, 1.0
        elif mode == MODE_VEL:
            motor.dq_des = float(struct.unpack("<f", data[0:4])[0])
            motor.kp, motor.kd = 0.0, 2.0
        else:
            return
        self.command_frames += 1
        self._send(motor.recv_id, motor.state_frame(), is_fd)

    def _rx_loop(self):
        while self.running:
            try:
                raw = self.sock.recv(CANFD_FRAME_SZ)
            except OSError:
                if not self.running:
                    break
                continue
            if len(raw) == CANFD_FRAME_SZ:
                can_id, dlc, _flags, _res, payload = struct.unpack(
                    CANFD_FRAME_FMT, raw)
                is_fd = True
            elif len(raw) == CAN_FRAME_SZ:
                can_id, dlc, payload = struct.unpack(CAN_FRAME_FMT, raw)
                is_fd = False
            else:
                continue
            self.frames_rx += 1
            self.handle_frame(can_id & CAN_EFF_MASK, payload[:dlc], is_fd)

    # ------------------------------------------------------------------- loop

    def run(self, viewer=None, status_period=2.0, on_status=None,
            keep_going=None):
        """Step the model in real time while the CAN thread serves the bus.

        `realtime` and `paused` are read every iteration, so a ROS parameter
        can slow the simulation down or freeze it without stopping the bus:
        the motors keep answering with their last state.
        """
        self.running = True
        threading.Thread(target=self._rx_loop, daemon=True).start()

        timestep = self.model.opt.timestep
        max_catch_up = 20
        last_wall = time.perf_counter()
        last_status = last_wall
        last_sync = last_wall
        last_counts = self._counters()
        budget = 0.0
        try:
            while self.running:
                if viewer is not None and not viewer.is_running():
                    break
                if keep_going is not None and not keep_going():
                    break

                now = time.perf_counter()
                if not self.paused:
                    budget += (now - last_wall) * max(0.0, self.realtime)
                last_wall = now

                steps = 0
                while budget >= timestep and steps < max_catch_up:
                    self.apply_commands()
                    mujoco.mj_step(self.model, self.data)
                    budget -= timestep
                    steps += 1
                if steps:
                    self.read_state()
                    self.steps += steps
                # Falling behind must not build up a debt to sprint through.
                budget = min(budget, max_catch_up * timestep)

                if viewer is not None and now - last_sync >= 1.0 / 60.0:
                    viewer.sync()
                    last_sync = now
                if status_period and now - last_status >= status_period:
                    counts = self._counters()
                    report = self.status(counts, last_counts, now - last_status)
                    (on_status or self._print_status)(report)
                    last_counts = counts
                    last_status = now
                time.sleep(timestep * 0.5)
        finally:
            self.stop()

    def _counters(self):
        return (self.steps, self.frames_rx, self.frames_tx, self.command_frames)

    def status(self, counts, previous, elapsed):
        """Rates since the last report, which is what says whether it keeps up.

        `step_hz` against 1/timestep shows whether the physics runs in real
        time; `control_hz` is how fast each motor is being commanded, so a
        controller claiming 400 Hz can be checked from this side of the bus.
        """
        elapsed = max(elapsed, 1e-6)
        step_hz, rx_hz, tx_hz, command_hz = (
            (now - before) / elapsed for now, before in zip(counts, previous))
        return {
            "step_hz": step_hz,
            "nominal_hz": 1.0 / self.model.opt.timestep,
            "rx_hz": rx_hz,
            "tx_hz": tx_hz,
            "control_hz": command_hz / max(1, len(self.motors)),
            "enabled": sum(1 for m in self.motors.values() if m.enabled),
            "motors": len(self.motors),
            "paused": self.paused,
        }

    @staticmethod
    def _print_status(status):
        print(f"[bridge] {CanMotorBridge.format_status(status)}", flush=True)

    @staticmethod
    def format_status(status):
        return (f"sim {status['step_hz']:.0f}/{status['nominal_hz']:.0f} Hz | "
                f"can rx {status['rx_hz']:.0f} Hz tx {status['tx_hz']:.0f} Hz | "
                f"control {status['control_hz']:.0f} Hz/motor | "
                f"enabled {status['enabled']}/{status['motors']}"
                f"{' | PAUSED' if status['paused'] else ''}")

    def stop(self):
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def default_model_path(relative=os.path.join("v1", "scene.xml")):
    """Locate a bundled scene, installed or straight from the checkout.

    An installed package has its assets under the ament share directory; a
    source tree keeps them beside this file. Both are tried so the node runs
    either way.
    """
    roots = []
    try:
        from ament_index_python.packages import get_package_share_directory
        roots.append(get_package_share_directory("openarm_mujoco"))
    except Exception:       # not built, or no ament index on this machine
        pass
    roots.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")))

    for root in roots:
        candidate = os.path.join(root, relative)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(roots[-1], relative)


class MujocoCanSimNode(Node):
    """ROS 2 wrapper: parameters in, MuJoCo physics on the CAN bus out.

    The bus itself carries the robot's data, so this node publishes nothing.
    What it adds is a handle: the simulation is launched, introspected and
    retuned like any other node.

        ros2 param list /openarm_mujoco_sim
        ros2 param set /openarm_mujoco_sim paused true
        ros2 param set /openarm_mujoco_sim realtime 0.25

    `interface`, `right_id_offset`, `model`, `keyframe` and `viewer` decide
    how the bus and the model are opened, so they are read once at startup;
    the rest take effect immediately.
    """

    def __init__(self):
        super().__init__("openarm_mujoco_sim")

        read_only = ParameterDescriptor(
            read_only=True, description="set at launch; cannot be changed later")
        self.declare_parameter("interface", "vcan0", read_only)
        self.declare_parameter("right_id_offset", 0x20, ParameterDescriptor(
            read_only=True,
            description="added to the right arm's CAN ids; 32 keeps both arms "
                        "on one bus, 0 matches a robot with a bus per arm"))
        self.declare_parameter("model", "", ParameterDescriptor(
            read_only=True, description="MJCF path; empty uses v1/scene.xml"))
        self.declare_parameter("keyframe", "home", read_only)
        self.declare_parameter("viewer", True, ParameterDescriptor(
            read_only=True, description="open the MuJoCo window"))
        self.declare_parameter("control_mode", "mit", ParameterDescriptor(
            description="mit honours the kp/kd sent over CAN; "
                        "position uses the model's tuned actuators"))
        self.declare_parameter("realtime", 1.0, ParameterDescriptor(
            description="simulation speed against the wall clock"))
        self.declare_parameter("paused", False, ParameterDescriptor(
            description="freeze physics; the motors keep answering the bus"))
        self.declare_parameter("status_period", 2.0, ParameterDescriptor(
            description="seconds between status lines, 0 to silence them"))

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        model_path = value("model") or default_model_path()
        if not os.path.exists(model_path):
            raise RuntimeError(f"model not found: {model_path}")

        mode = value("control_mode")
        if mode not in ("mit", "position"):
            raise RuntimeError(f"control_mode must be mit or position, got {mode!r}")

        interface = value("interface")
        try:
            self.bridge = CanMotorBridge(
                model_path, interface, int(value("right_id_offset")),
                value("keyframe"), mode)
        except OSError as error:
            raise RuntimeError(
                f"cannot bind {interface!r}: {error}. Create it with: "
                f"sudo ip link add dev {interface} type vcan && "
                f"sudo ip link set {interface} mtu 72 && "
                f"sudo ip link set {interface} up") from error

        self.bridge.realtime = float(value("realtime"))
        self.bridge.paused = bool(value("paused"))
        self.status_period = float(value("status_period"))
        self.add_on_set_parameters_callback(self._on_set_parameters)

        left = self.bridge.motors[1]
        right = self.bridge.motors[9]
        self.get_logger().info(
            f"{os.path.basename(model_path)} on {interface} | "
            f"{len(self.bridge.motors)} motors | {mode} mode")
        self.get_logger().info(
            f"left send ids 0x{left.send_id:02x}-0x{left.send_id + 7:02x}, "
            f"right 0x{right.send_id:02x}-0x{right.send_id + 7:02x}")

    def _on_set_parameters(self, parameters):
        """Validate and apply a `ros2 param set` while the model is running."""
        for parameter in parameters:
            if parameter.name == "control_mode":
                if parameter.value not in ("mit", "position"):
                    return SetParametersResult(
                        successful=False,
                        reason="control_mode must be 'mit' or 'position'")
            elif parameter.name == "realtime":
                if parameter.value < 0.0:
                    return SetParametersResult(
                        successful=False, reason="realtime must not be negative")

        for parameter in parameters:
            if parameter.name == "control_mode":
                self.bridge.control_mode = parameter.value
            elif parameter.name == "realtime":
                self.bridge.realtime = float(parameter.value)
            elif parameter.name == "paused":
                self.bridge.paused = bool(parameter.value)
            elif parameter.name == "status_period":
                self.status_period = float(parameter.value)
        return SetParametersResult(successful=True)

    def log_status(self, status):
        self.get_logger().info(CanMotorBridge.format_status(status))


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = MujocoCanSimNode()
    except RuntimeError as error:
        print(f"[openarm_mujoco_sim] {error}", file=sys.stderr)
        rclpy.shutdown()
        return 1

    # The physics and the viewer own the main thread; ROS spins beside them so
    # parameter changes land while the model keeps stepping.
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    bridge = node.bridge
    viewer_context = None
    if node.get_parameter("viewer").value:
        try:
            import mujoco.viewer
            viewer_context = mujoco.viewer.launch_passive(
                bridge.model, bridge.data,
                show_left_ui=False, show_right_ui=False)
        except Exception as error:      # no display, no GL, ...
            node.get_logger().warning(f"no viewer ({error}); running headless")

    try:
        if viewer_context is not None:
            with viewer_context as viewer:
                bridge.run(viewer, node.status_period, node.log_status,
                           keep_going=rclpy.ok)
        else:
            bridge.run(None, node.status_period, node.log_status,
                       keep_going=rclpy.ok)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
