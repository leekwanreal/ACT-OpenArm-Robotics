#!/usr/bin/env python3
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

"""Drive the OpenArm motors over CAN from ROS 2 topics.

    /openarm/joint_commands  (sensor_msgs/JointState, any rate)
        -> quintic interpolation -> 400 Hz CAN control loop -> motors
    motors -> CAN feedback -> /openarm/joint_states (100 Hz)
    the setpoint the loop writes -> /openarm/controller/joint_commands
        (100 Hz, always, for debugging; see _publish_sent)

The control loop runs at 400 Hz in its own thread, well above the rate
commands arrive at, so the quintic layer in spline.py is what keeps the
motion smooth: each command starts a segment from the joint's current
position, velocity and acceleration instead of stepping the setpoint.

Units on both topics follow openarm_can: arm joints in rad and rad/s,
grippers as finger stroke in m and m/s.
"""

import math
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from . import damiao
from .spline import DEFAULT_BANDWIDTH, KINDS, BimanualInterpolator


# Which gains a joint has, by whether it is a gripper, and the range the CAN
# frame can carry for each. A value outside the range would silently saturate.
GAIN_FIELDS = {False: ("kp", "kd"), True: ("speed", "effort")}
# The value a joint starts with when a profile does not say otherwise.
GAIN_DEFAULTS = {
    False: (("kp", 18.0), ("kd", 2.0)),
    True: (("speed", 10.0), ("effort", 0.15)),
}
GAIN_RANGES = {
    "kp": (damiao.KP_MIN, damiao.KP_MAX),
    "kd": (damiao.KD_MIN, damiao.KD_MAX),
    "speed": (0.0, damiao.POSFORCE_SPEED_MAX),
    "effort": (0.0, damiao.POSFORCE_CURRENT_MAX),
}


class OpenArmController(Node):
    """CAN control loop and the two joint topics around it."""

    def __init__(self):
        super().__init__("openarm_controller")

        # How the bus and the motor table are built: fixed once the CAN
        # threads are up, so these are read-only.
        fixed = ParameterDescriptor(
            read_only=True, description="set at launch; cannot be changed later")
        self.declare_parameter("left_interface", "can1", fixed)
        self.declare_parameter("right_interface", "can0", fixed)
        self.declare_parameter("arms", "both", ParameterDescriptor(
            read_only=True, description="both, left or right"))
        self.declare_parameter("right_id_offset", 0, ParameterDescriptor(
            read_only=True,
            description="added to the right arm's CAN ids; 32 when both arms "
                        "share one bus"))
        self.declare_parameter("control_rate", 400.0, ParameterDescriptor(
            read_only=True, description="CAN read/write rate in Hz"))
        self.declare_parameter("publish_rate", 100.0, ParameterDescriptor(
            read_only=True, description="joint_states rate in Hz"))
        self.declare_parameter("auto_enable", False, fixed)
        self.declare_parameter("state_topic", "/openarm/joint_states", fixed)
        self.declare_parameter("left_gripper_topic", "/openarm/left_gripper", fixed)
        self.declare_parameter("right_gripper_topic", "/openarm/right_gripper", fixed)
        self.declare_parameter("command_topic", "/openarm/joint_commands", fixed)
        self.declare_parameter(
            "sent_command_topic", "/openarm/controller/joint_commands",
            ParameterDescriptor(
                read_only=True,
                description="what the control loop writes to CAN, at "
                            "publish_rate whether or not a source commands"))

        # Gains live in profiles: one set of numbers per way of driving the
        # arm. Teleoperation wants a compliant arm the operator can push
        # against, a replayed policy wants a stiff one that tracks, and
        # switching between them should be one parameter rather than sixteen.
        self.declare_parameter("profiles", ["default"], ParameterDescriptor(
            read_only=True,
            description="gain profile names; each needs a block under gains"))
        self.declare_parameter("active_profile", "default", ParameterDescriptor(
            description="which profile's gains the motors run with"))

        self.profiles = [str(name) for name in self.get_parameter("profiles").value]
        if not self.profiles:
            raise ValueError("profiles must not be empty")

        # Within a profile the gains are per joint: a DM8009 carrying a
        # shoulder and a DM4310 on a wrist want nothing like the same numbers.
        # Arm joints take the MIT pair; the grippers run in position-force
        # mode, so theirs is a speed and a current cap instead.
        for profile in self.profiles:
            for name in damiao.JOINT_NAMES:
                for field, default in GAIN_DEFAULTS[name.endswith("gripper")]:
                    low, high = GAIN_RANGES[field]
                    self.declare_parameter(
                        f"gains.{profile}.{name}.{field}", default,
                        ParameterDescriptor(
                            description=f"{name} {field} in the {profile} "
                                        f"profile ({low}..{high})"))

        self.active_profile = str(self.get_parameter("active_profile").value)
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"active_profile {self.active_profile!r} is not one of {self.profiles}")
        self.declare_parameter("interpolator", "quintic", ParameterDescriptor(
            description="how commands become the 400 Hz setpoint and velocity "
                        "the MIT PD gets: quintic (spline.py, segments within "
                        "each joint's speed and acceleration) or filter4 "
                        "(filter4.py, a critically damped fourth-order filter)"))
        self.declare_parameter("interp_w", DEFAULT_BANDWIDTH, ParameterDescriptor(
            description="filter4's rad/s: a smooth stream is followed ~4/w s "
                        "behind, a step settles in ~7.75/w s. 1..100"))
        self.declare_parameter("command_timeout", 0.5, ParameterDescriptor(
            description="seconds without a command before the loop idles"))
        self.declare_parameter("feedback_timeout", 0.5, ParameterDescriptor(
            description="seconds without a motor reply before warning"))
        # Starts out matching auto_enable, so this reports what the motors
        # are actually doing: with auto_enable false nothing is armed yet, and
        # the dashboard would otherwise show them enabled before they are.
        self.declare_parameter(
            "motors_enabled", self.get_parameter("auto_enable").value,
            ParameterDescriptor(
                description="false disables every motor, so the arm goes limp; "
                            "true arms them again"))
        # Reported, not set: joint_states is published whether or not the
        # motors answer, so this is the number that says whether what it
        # carries is the robot. The dashboard reads it to tell a connection
        # from a controller talking to an empty bus.
        self.declare_parameter("motors_answering", 0, ParameterDescriptor(
            description="how many motors are currently answering on CAN; "
                        "written by this node, not settable"))

        param = lambda name: self.get_parameter(name).value  # noqa: E731
        left_if = param("left_interface")
        right_if = param("right_interface")
        arms = param("arms")
        if arms not in ("both", "left", "right"):
            raise ValueError(f"arms must be both, left or right, not {arms!r}")

        self.control_rate = float(param("control_rate"))
        self.publish_rate = float(param("publish_rate"))
        self.auto_enable = bool(param("auto_enable"))
        self.command_timeout = float(param("command_timeout"))
        self.feedback_timeout = float(param("feedback_timeout"))
        self._was_stale = False
        self._answering = None       # last value published to the parameter
        self._reporting = False      # true while this node writes it itself
        self._rearming = None
        self.control_dt = 1.0 / max(1.0, self.control_rate)
        # Control ticks between two idle state polls.
        self._ticks_per_poll = max(1, round(self.control_rate
                                            / max(1.0, self.publish_rate)))
        self._idle_tick = 0
        # What the loop last wrote per motor, (q, dq) in motor rad, and
        # whether it is writing them now or idling on refresh polls.
        self.sent = {}
        self.commanding = False

        self.motors = damiao.build_motors(left_if, right_if, arms,
                                          int(param("right_id_offset")))
        self._load_gains()
        self.interp = BimanualInterpolator(
            sorted(self.motors), kind=str(param("interpolator")),
            w=float(param("interp_w")))

        # Guards the motor feedback that the CAN threads write and the
        # publish timer reads.
        self.state_lock = threading.Lock()
        self.last_command_time = 0.0
        self.last_command_interval = 0.02
        self.have_command = False
        self.running = False

        self.buses = {}
        self._open_buses()
        # recv id -> motor, per bus: both arms share ids when each has its own.
        self.by_recv = {name: {} for name in self.buses}
        for motor in self.motors.values():
            self.by_recv[motor.interface][motor.recv_id] = motor

        sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE)
        # State is split the way commands are: the arm joints travel together
        # in one JointState, and each gripper has its own topic, because a
        # gripper is a stroke in metres rather than an angle and most
        # consumers want one without the other.
        self.state_pub = self.create_publisher(
            JointState, param("state_topic"), sensor_qos)
        self.gripper_pubs = {
            damiao.LEFT_GRIPPER_ID: self.create_publisher(
                Float64MultiArray, param("left_gripper_topic"), sensor_qos),
            damiao.RIGHT_GRIPPER_ID: self.create_publisher(
                Float64MultiArray, param("right_gripper_topic"), sensor_qos),
        }
        self.sent_pub = self.create_publisher(
            JointState, param("sent_command_topic"), sensor_qos)
        self.command_sub = self.create_subscription(
            JointState, param("command_topic"), self._on_command, sensor_qos)

        self._start_threads()
        self.create_timer(1.0 / max(1.0, self.publish_rate), self._publish_state)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"{len(self.motors)} motors on {sorted(self.buses)} | "
            f"control {self.control_rate:.0f} Hz ({self.interp.kind}"
            f"{f', w {self.interp.w:g}' if self.interp.kind == 'filter4' else ''}) | "
            f"state {self.publish_rate:.0f} Hz on {param('state_topic')}")

    # ------------------------------------------------------------------ setup

    def _load_gains(self):
        """Copy the active profile's gains onto every motor."""
        for motor in self.motors.values():
            name = damiao.JOINT_NAMES[motor.id - 1]
            for field in GAIN_FIELDS[motor.is_gripper]:
                value = self.get_parameter(
                    f"gains.{self.active_profile}.{name}.{field}").value
                setattr(motor, field, float(value))

    def _gain_target(self, parameter_name):
        """Split `gains.<profile>.<joint>.<field>` into its parts.

        Returns (motor, field, profile). The motor is None when the name is
        not a gain at all; the profile tells the caller whether the change
        affects what is running or only what another profile would use.
        """
        parts = parameter_name.split(".")
        if len(parts) != 4 or parts[0] != "gains":
            return None, None, None
        _, profile, joint, field = parts
        if profile not in self.profiles:
            return None, None, None
        motor_id = damiao.JOINT_NAME_TO_ID.get(joint)
        motor = self.motors.get(motor_id) if motor_id else None
        if motor is None or field not in GAIN_FIELDS[motor.is_gripper]:
            return None, None, None
        return motor, field, profile

    def _open_buses(self):
        for name in {m.interface for m in self.motors.values()}:
            bus = damiao.CanBus(name)
            try:
                bus.open()
            except OSError as error:
                raise RuntimeError(
                    f"cannot open CAN interface {name!r}: {error}. Bring it up "
                    f"with: sudo ip link set {name} up type can bitrate 1000000 "
                    f"dbitrate 5000000 fd on") from error
            if not bus.use_fd:
                self.get_logger().warning(
                    f"{name} has no CAN-FD (set mtu 72); using classic frames")
            self.buses[name] = bus

    def _start_threads(self):
        self.running = True
        self.threads = []
        for name, bus in self.buses.items():
            thread = threading.Thread(target=self._rx_loop, args=(name, bus),
                                      daemon=True)
            thread.start()
            self.threads.append(thread)

        if self.auto_enable:
            self._arm_motors()

        control = threading.Thread(target=self._control_loop, daemon=True)
        control.start()
        self.threads.append(control)

    def _rearm(self, force=False):
        """Arm the motors again, off the executor thread.

        `_arm_motors` paces itself with short sleeps, so it must not run in
        the timer callback that calls this. `force` is for an explicit request
        through the motors_enabled parameter, which arms without waiting to
        see whether the motors are already answering.

        What gates the unforced call is motors_enabled, not auto_enable.
        auto_enable only says whether this node arms the motors by itself at
        startup; once something has asked for them to be on, motors coming
        back -- a simulator that was still loading, a robot power-cycled
        mid-session -- should be armed, or the arm stays limp with the
        dashboard showing it enabled.
        """
        if not self.running:
            return
        if not force:
            # Someone disabled the motors on purpose; a bus dropout coming
            # back is not a reason to overrule them.
            if not self.get_parameter("motors_enabled").value:
                return
        if self._rearming is not None and self._rearming.is_alive():
            return
        self._rearming = threading.Thread(target=self._arm_motors, daemon=True)
        self._rearming.start()

    def _arm_motors(self):
        """Clear faults, put grippers in position-force mode, enable output."""
        for motor in self.motors.values():
            bus = self.buses[motor.interface]
            bus.send(motor.send_id, damiao.encode_special(damiao.CMD_CLEAR_ERROR))
            time.sleep(0.002)
            if motor.is_gripper:
                bus.send(damiao.MGMT_ID, damiao.encode_write_param(
                    motor.send_id, damiao.RID_CTRL_MODE,
                    damiao.CTRL_MODE_POSFORCE))
                time.sleep(0.002)
            bus.send(motor.send_id, damiao.encode_special(damiao.CMD_ENABLE))
            time.sleep(0.002)
            bus.send(damiao.MGMT_ID, damiao.encode_refresh(motor.send_id))
            time.sleep(0.002)

        # Let the replies land, then start the interpolators from where the
        # arm actually is so the first command does not jump.
        time.sleep(0.05)
        with self.state_lock:
            answered = 0
            for motor in self.motors.values():
                motor.enabled = True
                self.interp.set_state(motor.id, motor.q, motor.dq)
                self.sent.pop(motor.id, None)   # nothing written since arming
                if motor.last_feedback:
                    answered += 1

        if answered == len(self.motors):
            self.get_logger().info(f"all {answered} motors answered")
        elif answered:
            self.get_logger().warning(
                f"only {answered}/{len(self.motors)} motors answered")
        else:
            self.get_logger().error(
                f"no motor answered on {sorted(self.buses)}. Nothing is "
                f"driving the bus: start the robot, or openarm_can's "
                f"sim/motor_simulator.py for a virtual one. Check the ids "
                f"too -- both arms on one bus need right_id_offset:=32.")

    # ------------------------------------------------------------------- loops

    def _rx_loop(self, name, bus):
        """Decode feedback frames into motor state."""
        recv_map = self.by_recv[name]
        while self.running:
            frame = bus.recv()
            if frame is None:
                if not self.running:
                    break
                continue
            can_id, data = frame
            if len(data) < 8:
                continue
            motor = recv_map.get(can_id)
            if motor is None:
                continue
            state = damiao.decode_state(data, motor.p_max, motor.v_max,
                                        motor.t_max)
            now = time.monotonic()
            with self.state_lock:
                motor.q = state["q"]
                motor.dq = state["dq"]
                motor.tau = state["tau"]
                motor.error_code = state["error_code"]
                motor.t_mos = state["t_mos"]
                motor.t_rotor = state["t_rotor"]
                motor.last_feedback = now

    def _control_loop(self):
        """Step the interpolators and write the bus at `control_rate`."""
        next_tick = time.perf_counter()
        while self.running:
            next_tick += self.control_dt
            self._control_tick()
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                # Fell behind: give up the missed ticks rather than burst.
                next_tick = time.perf_counter()

    def _control_tick(self):
        with self.state_lock:
            have_command = self.have_command
            age = time.monotonic() - self.last_command_time

        # A quiet publisher is not a reason to abandon a move already under
        # way: a quintic segment for a wide swing runs far longer than the
        # timeout, and dropping it leaves the arm wherever the stream stopped.
        # So the loop keeps driving until the interpolators have run their
        # segments out, and only then idles.
        stale = not have_command or age > self.command_timeout
        if stale and self.interp.settled():
            self.commanding = False
            self._idle_tick += 1
            if self._idle_tick < self._ticks_per_poll:
                return
            self._idle_tick = 0
            for motor in self.motors.values():
                self.buses[motor.interface].send(
                    damiao.MGMT_ID, damiao.encode_refresh(motor.send_id))
            return
        self._idle_tick = 0
        self.commanding = True

        for motor in self.motors.values():
            q_cmd, v_cmd = self.interp.step(motor.id, self.control_dt)
            self.sent[motor.id] = (q_cmd, v_cmd)
            bus = self.buses[motor.interface]
            if motor.is_gripper:
                bus.send(motor.send_id + damiao.MODE_POSFORCE,
                         damiao.encode_posforce(
                             q_cmd, motor.speed, motor.effort))
            else:
                bus.send(motor.send_id, damiao.encode_mit(
                    q_cmd, v_cmd, motor.kp, motor.kd, 0.0,
                    motor.p_max, motor.v_max, motor.t_max))

    # ---------------------------------------------------------------- ROS side

    def _on_set_parameters(self, parameters):
        """Apply a `ros2 param set` to the running control loop.

        The gains and the timeouts are the ones worth retuning against a live
        arm. Everything that decides how the bus and the motor table are built
        is declared read-only, so it cannot change underneath the CAN threads.
        """
        for parameter in parameters:
            _motor, field, _profile = self._gain_target(parameter.name)
            if field is not None:
                low, high = GAIN_RANGES[field]
                if not low <= parameter.value <= high:
                    return SetParametersResult(
                        successful=False,
                        reason=f"{parameter.name} must be between {low} and {high}")
            if parameter.name == "active_profile" and parameter.value not in self.profiles:
                return SetParametersResult(
                    successful=False,
                    reason=f"active_profile must be one of {self.profiles}")
            if parameter.name == "interpolator" and parameter.value not in KINDS:
                return SetParametersResult(
                    successful=False, reason=f"interpolator must be one of {KINDS}")
            if parameter.name == "interp_w" and not 1.0 <= parameter.value <= 100.0:
                # Semi-implicit Euler at 400 Hz stays well inside its stable
                # range up to here.
                return SetParametersResult(
                    successful=False, reason="interp_w must be within 1..100 rad/s")
            if (parameter.name in ("command_timeout", "feedback_timeout")
                    and parameter.value <= 0.0):
                return SetParametersResult(
                    successful=False, reason=f"{parameter.name} must be positive")
            if parameter.name == "motors_answering" and not self._reporting:
                return SetParametersResult(
                    successful=False,
                    reason="motors_answering is reported by the controller")

        for parameter in parameters:
            motor, field, profile = self._gain_target(parameter.name)
            if field is not None:
                # Editing a profile that is not running changes what it will
                # use next time, not what the motors are doing now.
                if profile == self.active_profile:
                    setattr(motor, field, float(parameter.value))
            elif parameter.name == "active_profile":
                self.active_profile = str(parameter.value)
                self._load_gains()
                self.get_logger().info(f"gain profile: {self.active_profile}")
            elif parameter.name == "command_timeout":
                self.command_timeout = float(parameter.value)
            elif parameter.name == "interpolator":
                # Each joint carries on from where it is: no jump.
                self.interp.set_kind(str(parameter.value))
                self.get_logger().info(f"interpolator: {parameter.value}")
            elif parameter.name == "interp_w":
                self.interp.set_bandwidth(float(parameter.value))
            elif parameter.name == "feedback_timeout":
                self.feedback_timeout = float(parameter.value)
            elif parameter.name == "motors_enabled":
                if parameter.value:
                    self._rearm(force=True)
                else:
                    self._disarm()
        return SetParametersResult(successful=True)

    def _disarm(self):
        """Disable every motor, which lets the arm fall limp."""
        for motor in self.motors.values():
            bus = self.buses.get(motor.interface)
            if bus is not None:
                bus.send(motor.send_id,
                         damiao.encode_special(damiao.CMD_DISABLE))
            motor.enabled = False
        self.get_logger().info("motors disabled")

    def _on_command(self, msg):
        """Take a new target per named joint and re-plan its quintic segment."""
        if not msg.name or not msg.position:
            return
        now = time.monotonic()
        gap = now - self.last_command_time if self.have_command else float("inf")
        if gap > self.command_timeout:
            # First command, or the stream resumed after a pause: that gap
            # says nothing about the publisher's rate.
            interval = 0.02
        else:
            # Ignore a bursting publisher when sizing the segment.
            interval = min(max(gap, 0.005), 0.1)

        count = 0
        for name, position in zip(msg.name, msg.position):
            motor_id = damiao.JOINT_NAME_TO_ID.get(name)
            if motor_id is None:
                continue
            motor = self.motors.get(motor_id)
            if motor is None:
                continue
            if not math.isfinite(position):
                continue
            target = motor.joint_to_motor_rad(position)
            target = max(-motor.p_max, min(motor.p_max, target))
            self.interp.update_target(motor_id, target, interval)
            count += 1

        if not count:
            self.get_logger().warning(
                f"no known joint in command: {list(msg.name)[:4]}",
                throttle_duration_sec=5.0)
            return

        with self.state_lock:
            self.last_command_time = now
            self.last_command_interval = interval
            self.have_command = True

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        now = time.monotonic()
        stale = []
        grippers = {}
        with self.state_lock:
            for motor_id in sorted(self.motors):
                motor = self.motors[motor_id]
                reading = (motor.joint_position(), motor.joint_velocity(),
                           motor.tau)
                if motor.is_gripper:
                    grippers[motor_id] = reading
                else:
                    msg.name.append(damiao.JOINT_NAMES[motor_id - 1])
                    msg.position.append(reading[0])
                    msg.velocity.append(reading[1])
                    msg.effort.append(reading[2])
                if (motor.last_feedback == 0.0
                        or now - motor.last_feedback > self.feedback_timeout):
                    stale.append(motor_id)

        self.state_pub.publish(msg)
        self._publish_sent(msg.header.stamp)
        for motor_id, reading in grippers.items():
            # data[0] is the finger stroke in metres, matching what the
            # command topic takes; velocity and effort follow it.
            self.gripper_pubs[motor_id].publish(
                Float64MultiArray(data=list(reading)))

        # A motor that never answers still gets published, as zeros. Say so,
        # otherwise joint_states looks alive but frozen.
        if stale:
            never = all(self.motors[i].last_feedback == 0.0 for i in stale)
            detail = ("nothing has ever answered" if never
                      else "last reply is stale")
            self.get_logger().warning(
                f"no CAN feedback from {len(stale)}/{len(self.motors)} motors "
                f"on {sorted(self.buses)} ({detail}); "
                f"send ids {[hex(self.motors[i].send_id) for i in stale[:4]]}"
                f"{'...' if len(stale) > 4 else ''}. The positions published "
                f"are not the robot's.", throttle_duration_sec=5.0)
        elif self._was_stale:
            self.get_logger().info("CAN feedback restored from all motors")
            # The motors were not there to be armed when this node started --
            # launched together with a simulator, or power-cycled since. They
            # answer now, so arm them, otherwise the arm stays limp.
            self._rearm()
        self._was_stale = bool(stale)
        self._report_answering(len(self.motors) - len(stale))

    def _publish_sent(self, stamp):
        """The setpoint on the bus, every joint, at the joint_states rate.

        For debugging, so it is published whether or not anything commands
        the arm: positions and velocities in the topics' units (rad, m on a
        gripper). header.frame_id says what the loop is doing --
        "commanding" (writing these frames at control_rate), "holding" (no
        source: only polling, the motors keep the last frame sent),
        "unsent" (nothing written since the motors were armed: the values
        are where the interpolators start) or "disabled".
        """
        enabled = any(motor.enabled for motor in self.motors.values())
        if not enabled:
            state = "disabled"
        elif self.commanding:
            state = "commanding"
        elif self.sent:
            state = "holding"
        else:
            state = "unsent"
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = state
        for motor_id in sorted(self.motors):
            motor = self.motors[motor_id]
            joint = self.interp.joints.get(motor_id)
            q, dq = self.sent.get(motor_id, (
                joint.current_q if joint else 0.0, 0.0))
            msg.name.append(damiao.JOINT_NAMES[motor_id - 1])
            msg.position.append(motor.motor_to_joint(q))
            msg.velocity.append(motor.motor_to_joint_rate(dq))
        self.sent_pub.publish(msg)

    def _report_answering(self, answering):
        """Publish the count, only when it changes.

        _publish_state runs at publish_rate, and setting a parameter fires
        the callback and a parameter event every time, so an unchanged count
        is not worth announcing.
        """
        if answering == self._answering:
            return
        self._answering = answering
        self._reporting = True
        try:
            self.set_parameters([Parameter(
                "motors_answering", Parameter.Type.INTEGER, answering)])
        except Exception as error:      # never let telemetry stop the loop
            self.get_logger().debug(f"cannot report motors_answering: {error}")
        finally:
            self._reporting = False

    # ----------------------------------------------------------------- cleanup

    def shutdown(self):
        """Disable the motors before dropping the bus."""
        if not self.running:
            return
        self.running = False
        for motor in self.motors.values():
            bus = self.buses.get(motor.interface)
            if bus is not None:
                bus.send(motor.send_id, damiao.encode_special(damiao.CMD_DISABLE))
        time.sleep(0.02)
        for bus in self.buses.values():
            bus.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = OpenArmController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print(f"[openarm_controller] {error}")
        return 1
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
