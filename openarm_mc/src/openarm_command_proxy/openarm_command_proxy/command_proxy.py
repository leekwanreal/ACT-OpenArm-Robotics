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

"""Pick which source commands the arm, and merge its topics into one.

Several things can drive the robot -- the dashboard UI, teleoperation, an ACT
policy -- and only one of them may at a time. Each publishes under its own
name, following openarm_can's convention:

    /openarm/<source>/joint_commands   std_msgs/Float64MultiArray, 14 values
                                       (left j1..j7 then right j1..j7)
    /openarm/<source>/left_gripper     std_msgs/Float64MultiArray, data[0]
    /openarm/<source>/right_gripper    std_msgs/Float64MultiArray, data[0]

This node subscribes to all of them, forwards only the selected one, and
merges its three topics into the 16-joint command the controller reads:

    /openarm/joint_commands            sensor_msgs/JointState

The selection is a parameter, so it changes while everything runs:

    ros2 param set /openarm_command_proxy active_source teleop
    ros2 param set /openarm_command_proxy active_source none   # block all

It starts on `none`: nothing reaches the arm until a source is chosen on
purpose.

Switching never jumps the arm. A source can only be selected while the
robot's own joint states are fresh, and the first command after a switch is
compared with them: when it is further than switch_threshold from where the
robot is, the output eases from where the robot is held onto the source's
targets (blend.py) before passing them straight through. Where it is held
is the controller's own setpoint (/openarm/controller/joint_commands) while
the motors hold one, else the measured pose.
"""

import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .blend import SwitchBlend, hold_pose

# Motor ids and names as openarm_can orders them.
JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]
ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
LEFT_GRIPPER_ID = 8
RIGHT_GRIPPER_ID = 16

# The selection that lets nothing through.
NO_SOURCE = "none"


class CommandProxy(Node):
    """One active source in, one merged joint command out."""

    def __init__(self):
        super().__init__("openarm_command_proxy")

        fixed = ParameterDescriptor(
            read_only=True, description="set at launch; cannot be changed later")
        self.declare_parameter("sources", ["ui", "teleop", "act"],
                               ParameterDescriptor(
                                   read_only=True,
                                   description="source names to subscribe, each "
                                               "under /openarm/<name>/"))
        self.declare_parameter("topic_prefix", "/openarm", fixed)
        self.declare_parameter("command_topic", "/openarm/joint_commands", fixed)
        self.declare_parameter("active_source", NO_SOURCE, ParameterDescriptor(
            description="which source reaches the arm; 'none' blocks all"))
        self.declare_parameter("status_period", 5.0, ParameterDescriptor(
            description="seconds between status lines, 0 to silence them"))
        # The robot's own state, read back before any source may drive it.
        self.declare_parameter("state_topic", "/openarm/joint_states", fixed)
        self.declare_parameter("left_gripper_state_topic", "/openarm/left_gripper", fixed)
        self.declare_parameter("right_gripper_state_topic", "/openarm/right_gripper", fixed)
        # The setpoint the controller writes to the motors: what a switch
        # starts from, so a new source does not let the arm sag first.
        self.declare_parameter("setpoint_topic", "/openarm/controller/joint_commands",
                               fixed)
        self.declare_parameter("state_timeout", 0.5, ParameterDescriptor(
            description="seconds; older joint states cannot start a source"))
        self.declare_parameter("switch_threshold", 0.02, ParameterDescriptor(
            description="rad; a new source's first command closer than this "
                        "to the robot passes straight through"))
        self.declare_parameter("switch_speed", 0.5, ParameterDescriptor(
            description="rad/s the easing after a switch averages on its "
                        "farthest arm joint"))
        self.declare_parameter("switch_min_duration", 0.5, ParameterDescriptor(
            description="seconds the easing after a switch takes at least"))

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.sources = [str(name) for name in value("sources")]
        if not self.sources:
            raise RuntimeError("sources must not be empty")
        if NO_SOURCE in self.sources:
            raise RuntimeError(f"{NO_SOURCE!r} is reserved and cannot be a source")
        duplicates = {name for name in self.sources
                      if self.sources.count(name) > 1}
        if duplicates:
            raise RuntimeError(f"duplicate sources: {sorted(duplicates)}")

        prefix = str(value("topic_prefix")).rstrip("/")
        self.topics = self._resolve_topics(prefix)
        requested = str(value("active_source"))
        if requested != NO_SOURCE and requested not in self.sources:
            raise RuntimeError(
                f"active_source {requested!r} is not one of {self.sources}")
        self.active_source = requested
        self.status_period = float(value("status_period"))
        self.state_timeout = float(value("state_timeout"))
        self.blend = SwitchBlend(
            ARM_JOINT_IDS, threshold=float(value("switch_threshold")),
            speed=float(value("switch_speed")),
            min_duration=float(value("switch_min_duration")))
        # Measured position per motor id, and when the last one arrived.
        self.measured = {}
        self.measured_time = None
        # The controller's setpoint per motor id, its state and arrival.
        self.setpoint = {}
        self.setpoint_state = None
        self.setpoint_time = None
        if requested != NO_SOURCE:
            self.blend.switched()       # its first command is checked too

        # Latest target per motor id, from whichever topics the active source
        # has published so far. Cleared on a switch so one source's last
        # position cannot leak into the next one's command.
        self.targets = {}
        self.lock = threading.Lock()
        self.counts = {name: 0 for name in self.sources}
        self.forwarded = 0
        self._last_counts = dict(self.counts)
        self._last_forwarded = 0
        self._last_status = time.monotonic()

        # A source may publish BEST_EFFORT or RELIABLE; a BEST_EFFORT
        # subscription accepts either. The output is RELIABLE to match what
        # openarm_controller subscribes with -- a BEST_EFFORT publisher would
        # silently never connect to it.
        source_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE)

        self.command_pub = self.create_publisher(
            JointState, str(value("command_topic")), output_qos)

        self.subscriptions_by_source = {}
        for name in self.sources:
            topics = self.topics[name]
            self.subscriptions_by_source[name] = [
                self.create_subscription(
                    Float64MultiArray, topics["joint_commands"],
                    self._arm_handler(name), source_qos),
                self.create_subscription(
                    Float64MultiArray, topics["left_gripper"],
                    self._gripper_handler(name, LEFT_GRIPPER_ID), source_qos),
                self.create_subscription(
                    Float64MultiArray, topics["right_gripper"],
                    self._gripper_handler(name, RIGHT_GRIPPER_ID), source_qos),
            ]

        state_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, str(value("state_topic")),
                                 self._on_state, state_qos)
        self.create_subscription(
            Float64MultiArray, str(value("left_gripper_state_topic")),
            lambda msg: self._on_gripper_state(LEFT_GRIPPER_ID, msg), state_qos)
        self.create_subscription(
            Float64MultiArray, str(value("right_gripper_state_topic")),
            lambda msg: self._on_gripper_state(RIGHT_GRIPPER_ID, msg), state_qos)
        self.create_subscription(JointState, str(value("setpoint_topic")),
                                 self._on_setpoint, state_qos)
        # While an easing runs the output keeps moving even when the source
        # is quiet (a slider sends once), so it is also published on a timer.
        self.create_timer(0.02, self._on_blend_tick)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        if self.status_period:
            self.create_timer(self.status_period, self._log_status)

        self.get_logger().info(
            f"{len(self.sources)} sources -> {value('command_topic')}")
        for name in self.sources:
            self.get_logger().info(f"  {name}: {self.topics[name]['joint_commands']}")
        self.get_logger().info(f"active source: {self.active_source}")

    def _resolve_topics(self, prefix):
        """Work out each source's three topics.

        A source follows <topic_prefix>/<name>/... unless it declares its own,
        so adding one is a line in the config and nothing else. A source that
        already publishes elsewhere, or that has no gripper of its own, points
        the odd topic somewhere else instead of being special-cased here.
        """
        resolved = {}
        for name in self.sources:
            topics = {}
            for kind in ("joint_commands", "left_gripper", "right_gripper"):
                parameter = f"topics.{name}.{kind}"
                self.declare_parameter(parameter, "", ParameterDescriptor(
                    read_only=True,
                    description=f"{name}'s {kind} topic; "
                                f"empty uses {prefix}/{name}/{kind}"))
                override = str(self.get_parameter(parameter).value).strip()
                topics[kind] = override or f"{prefix}/{name}/{kind}"
            resolved[name] = topics
        return resolved

    # ------------------------------------------------------------ robot state

    def _on_state(self, msg):
        with self.lock:
            for name, position in zip(msg.name, msg.position):
                if name in JOINT_NAMES:
                    self.measured[JOINT_NAMES.index(name) + 1] = float(position)
            self.measured_time = time.monotonic()

    def _on_setpoint(self, msg):
        with self.lock:
            for name, position in zip(msg.name, msg.position):
                if name in JOINT_NAMES:
                    self.setpoint[JOINT_NAMES.index(name) + 1] = float(position)
            self.setpoint_state = msg.header.frame_id
            self.setpoint_time = time.monotonic()

    def _held(self):
        """Where the arm is held, per motor id (lock held)."""
        age = (None if self.setpoint_time is None
               else time.monotonic() - self.setpoint_time)
        return hold_pose(self.measured, self.setpoint, self.setpoint_state, age)

    def _on_gripper_state(self, motor_id, msg):
        if msg.data:
            with self.lock:
                self.measured[motor_id] = float(msg.data[0])

    def _state_problem(self):
        """Why the robot's state cannot start a source, or None."""
        if self.measured_time is None:
            return "no joint state from the robot yet"
        age = time.monotonic() - self.measured_time
        if age > self.state_timeout:
            return f"the robot's joint state is {age:.1f} s old"
        missing = [JOINT_NAMES[i - 1] for i in ARM_JOINT_IDS if i not in self.measured]
        if missing:
            return f"no state for {', '.join(missing)}"
        return None

    # --------------------------------------------------------------- incoming

    def _arm_handler(self, source):
        def handle(msg):
            if source != self.active_source:
                return
            self.counts[source] += 1
            if len(msg.data) != len(ARM_JOINT_IDS):
                self.get_logger().warning(
                    f"{source}/joint_commands has {len(msg.data)} values, "
                    f"expected {len(ARM_JOINT_IDS)} (left j1..j7 then "
                    f"right j1..j7); ignored", throttle_duration_sec=5.0)
                return
            with self.lock:
                for motor_id, position in zip(ARM_JOINT_IDS, msg.data):
                    self.targets[motor_id] = float(position)
            self._publish()
        return handle

    def _gripper_handler(self, source, motor_id):
        def handle(msg):
            if source != self.active_source:
                return
            self.counts[source] += 1
            if not msg.data:
                return
            with self.lock:
                self.targets[motor_id] = float(msg.data[0])
            self._publish()
        return handle

    # --------------------------------------------------------------- outgoing

    def _publish(self):
        """Send whatever targets the active source has given so far.

        Only the joints that have arrived are named, so the arm starts moving
        on the first arm message without waiting for the grippers.
        """
        with self.lock:
            if not self.targets:
                return
            output = self.blend.output(self.targets, self._held(), time.monotonic())
            motor_ids = sorted(output)
            positions = [output[motor_id] for motor_id in motor_ids]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [JOINT_NAMES[motor_id - 1] for motor_id in motor_ids]
        msg.position = positions
        self.command_pub.publish(msg)
        self.forwarded += 1

    def _on_blend_tick(self):
        if self.blend.active:
            self._publish()

    # -------------------------------------------------------------- selection

    def _on_set_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == "active_source":
                if (parameter.value != NO_SOURCE
                        and parameter.value not in self.sources):
                    return SetParametersResult(
                        successful=False,
                        reason=f"active_source must be {NO_SOURCE} or one of "
                               f"{self.sources}")
                if (parameter.value != NO_SOURCE
                        and parameter.value != self.active_source):
                    # Easing in needs to know where the robot is.
                    with self.lock:
                        problem = self._state_problem()
                    if problem:
                        self.get_logger().warning(
                            f"not switching to {parameter.value}: {problem}")
                        return SetParametersResult(successful=False, reason=problem)
            elif parameter.name in ("switch_speed", "state_timeout") and \
                    parameter.value <= 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{parameter.name} must be positive")
            elif parameter.name == "status_period" and parameter.value < 0.0:
                return SetParametersResult(
                    successful=False, reason="status_period must not be negative")

        for parameter in parameters:
            if parameter.name == "active_source":
                self._switch_to(parameter.value)
            elif parameter.name == "status_period":
                self.status_period = float(parameter.value)
            elif parameter.name == "state_timeout":
                self.state_timeout = float(parameter.value)
            elif parameter.name == "switch_threshold":
                self.blend.threshold = float(parameter.value)
            elif parameter.name == "switch_speed":
                self.blend.speed = float(parameter.value)
            elif parameter.name == "switch_min_duration":
                self.blend.min_duration = float(parameter.value)
        return SetParametersResult(successful=True)

    def _switch_to(self, source):
        if source == self.active_source:
            return
        previous = self.active_source
        self.active_source = source
        # Drop the old source's targets: the new one must say where it wants
        # the arm before anything else is forwarded.
        with self.lock:
            self.targets.clear()
            self.blend.switched()
        self.get_logger().info(f"active source: {previous} -> {source}")

    # ----------------------------------------------------------------- status

    def _log_status(self):
        now = time.monotonic()
        elapsed = max(now - self._last_status, 1e-6)
        rates = []
        for name in self.sources:
            rate = (self.counts[name] - self._last_counts[name]) / elapsed
            rates.append(f"{name} {rate:.0f} Hz")
        out = (self.forwarded - self._last_forwarded) / elapsed

        self._last_counts = dict(self.counts)
        self._last_forwarded = self.forwarded
        self._last_status = now

        self.get_logger().info(
            f"active {self.active_source} | {' | '.join(rates)} | "
            f"out {out:.0f} Hz")


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = CommandProxy()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(f"[openarm_command_proxy] {error}")
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
