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

"""The ROS half of the dashboard: topics in, topics out, parameters across.

Nothing here knows about HTTP or WebSockets. The page talks to this through
web_server.py, and so could anything else.
"""

import json
import os
import threading
import time

from rcl_interfaces.msg import (
    Parameter, ParameterEvent, ParameterType, ParameterValue)
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_parameter_events)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .joints import (
    ARM_JOINT_IDS,
    JOINT_NAMES,
    JOINT_NAME_TO_ID,
    LEFT_GRIPPER_ID,
    RIGHT_GRIPPER_ID,
)
from .zero_pose import FINISHING, MOVING, ZeroPoseMotion, ZeroPoseState

CONTROLLER_NODE = "/openarm_controller"
PROXY_NODE = "/openarm_command_proxy"
ACT_NODE = "/openarm_act"
# openarm_camera's roles, in order; a virtual camera stands in for one that
# is not plugged in, from a node named so.
CAMERAS = ("chest", "front")
VIRTUAL_CAMERA_NODE = "virtual_camera"
SMOLVLA_NODE = "/openarm_smolvla"
# A policy node's status arrives twice a second; older than this, it is gone.
POLICY_STATUS_STALE_S = 2.0
# The proxy's selection that lets nothing through.
NO_SOURCE = "none"
# Older joint states than this cannot start, or keep running, a zero pose.
ZERO_STATE_TIMEOUT_S = 0.5
# The controller's setpoint counts as where the arm is held while it is
# this fresh and the motors hold it (frame_id commanding or holding).
SETPOINT_STALE_S = 0.25
HOLDING_STATES = ("commanding", "holding")


class PolicyLink:
    """One policy node -- openarm_act, openarm_smolvla -- as the page sees it.

    What the policy would do, shown before anyone lets it drive: its status,
    the preview pose (which only says the prediction is current) and the
    last inference's whole chunk, one JointTrajectory point per control
    step, drawn as arrows. Out: its generate and reset services, and the
    plan topic a trajectory planned on the page goes to. `name` is also the
    proxy source the node commands as.
    """

    def __init__(self, node, name, label, node_name, prefix, trajectory_topic,
                 qos):
        self.node = node
        self.name = name
        self.label = label              # what the page and its messages call it
        self.node_name = node_name
        self.lock = threading.Lock()
        self.status = None              # (monotonic, dict)
        self.pose = None                # (monotonic, {name: value})
        self.trajectory = None          # (sequence, [[16 values], ...])
        self.sequence = 0
        prefix = prefix.rstrip("/")
        node.create_subscription(String, f"{prefix}/status",
                                 self._on_status, qos)
        node.create_subscription(JointState, f"{prefix}/preview",
                                 self._on_preview, qos)
        node.create_subscription(JointTrajectory, trajectory_topic,
                                 self._on_trajectory, qos)
        self.generate_client = node.create_client(
            Trigger, f"{node_name}/generate")
        self.reset_client = node.create_client(Trigger, f"{node_name}/reset")
        self.plan_pub = node.create_publisher(
            JointTrajectory, f"{node_name}/plan", qos)

    def _on_status(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status = (time.monotonic(), status)

    def _on_preview(self, msg):
        with self.lock:
            self.pose = (time.monotonic(),
                         dict(zip(msg.name, (float(v) for v in msg.position))))

    def _on_trajectory(self, msg):
        # Rows in JOINT_NAMES order, whatever order the message names them in.
        try:
            columns = [list(msg.joint_names).index(name) for name in JOINT_NAMES]
        except ValueError:
            return
        rows = [[float(point.positions[c]) for c in columns]
                for point in msg.points if len(point.positions) == len(msg.joint_names)]
        if not rows:
            return
        with self.lock:
            self.sequence += 1
            self.trajectory = (self.sequence, rows)

    def snapshot(self):
        """{status or None, pose or None, trajectory (seq, rows) or None}."""
        now = time.monotonic()
        with self.lock:
            status = self.status
            pose = self.pose
            trajectory = self.trajectory
        alive = status is not None and now - status[0] <= POLICY_STATUS_STALE_S
        return {
            "status": status[1] if alive else None,
            "pose": pose[1] if alive and pose and now - pose[0] <= 1.0 else None,
            "trajectory": trajectory if alive else None,
        }

    def running(self):
        """Whether the node is up: its status is current."""
        return self.snapshot()["status"] is not None

    def generate(self):
        """Ask for one trajectory; -> (ok, message). The node answers at
        once; the trajectory, and whether it worked, follow on its status."""
        return self._trigger(self.generate_client)

    def reset(self):
        """Ask the node to take the arm to its reset pose; -> (ok, message)."""
        return self._trigger(self.reset_client)

    def plan(self, rows, step_s=0.02):
        """Hand the node a planned trajectory: rows of 16 values in
        JOINT_NAMES order, one per control step. -> (ok, message); whether
        it was taken comes back on its status, as for a generation."""
        if self.plan_pub.get_subscription_count() == 0:
            return False, f"{self.node_name} is not running"
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(JOINT_NAMES)
        step_ns = int(round(step_s * 1e9))
        for index, row in enumerate(rows):
            ns = index * step_ns
            msg.points.append(JointTrajectoryPoint(
                positions=[float(value) for value in row],
                time_from_start=Duration(sec=ns // 1_000_000_000,
                                         nanosec=ns % 1_000_000_000)))
        self.plan_pub.publish(msg)
        return True, f"planned {len(rows)} steps"

    def _trigger(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, f"{self.node_name} is not reachable"
        future = client.call_async(Trigger.Request())
        if not self.node._spin_until(future, 2.0):
            return False, f"{self.node_name} did not answer"
        result = future.result()
        return result.success, result.message


class UiServer(Node):
    """The ROS half: topics in, topics out, parameters on the other nodes."""

    def __init__(self):
        super().__init__("openarm_ui")

        self.declare_parameter("http_port", 8080)
        self.declare_parameter("websocket_port", 8081)
        self.declare_parameter("web_root", "")
        self.declare_parameter("telemetry_rate", 30.0)
        self.declare_parameter("command_rate", 50.0)
        self.declare_parameter("controller_node", CONTROLLER_NODE)
        self.declare_parameter("proxy_node", PROXY_NODE)
        self.declare_parameter("act_node", ACT_NODE)
        self.declare_parameter("act_prefix", "/openarm/act")
        self.declare_parameter("trajectory_topic", "/openarm/trajectory")
        self.declare_parameter("smolvla_node", SMOLVLA_NODE)
        self.declare_parameter("smolvla_prefix", "/openarm/smolvla")
        self.declare_parameter("smolvla_trajectory_topic",
                               "/openarm/smolvla/trajectory")
        self.declare_parameter("source_name", "ui")
        # The camera views: each camera's RGB-D pair as openarm_camera
        # prepares it, <camera_prefix>/<camera>/{rgb,depth} -- what the
        # recorder and the policies see.
        self.declare_parameter("cameras", list(CAMERAS))
        self.declare_parameter("camera_prefix", "/openarm/camera")
        self.declare_parameter("camera_rate", 10.0)
        self.declare_parameter("camera_jpeg_quality", 70)
        # Depth is coloured over the recorder's working range.
        self.declare_parameter("depth_min_mm", 200)
        self.declare_parameter("depth_max_mm", 1200)
        # Zero pose: every arm joint to 0, its farthest joint averaging this
        # many rad/s, and taking this long at least. The grippers stay put.
        self.declare_parameter("zero_pose_speed", 0.5)
        self.declare_parameter("zero_pose_min_duration", 1.5)

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.http_port = int(value("http_port"))
        self.websocket_port = int(value("websocket_port"))
        self.telemetry_rate = float(value("telemetry_rate"))
        self.command_rate = float(value("command_rate"))
        self.controller_node = str(value("controller_node"))
        self.proxy_node = str(value("proxy_node"))
        # Resolved by web_server, which is what actually serves the files.
        self.web_root = str(value("web_root"))
        self.cameras = tuple(name for name in value("cameras") if name)
        camera_prefix = str(value("camera_prefix")).rstrip("/")
        # "<camera>/<kind>" -> topic: the streams a page may ask for.
        self.camera_topics = {f"{camera}/{kind}": f"{camera_prefix}/{camera}/{kind}"
                              for camera in self.cameras for kind in ("rgb", "depth")}
        self.camera_rate = float(value("camera_rate"))
        self.camera_jpeg_quality = int(value("camera_jpeg_quality"))
        self.depth_range_mm = (int(value("depth_min_mm")),
                               int(value("depth_max_mm")))
        self.zero_speed = max(1e-3, float(value("zero_pose_speed")))
        self.zero_min_duration = max(0.1, float(value("zero_pose_min_duration")))

        # Camera frames, only while some page is watching. The page says
        # which streams it wants; the subscriptions follow on the executor
        # thread (see _follow_camera_wanted), so an image nobody looks at is
        # never sent to this process at all.
        self.camera_lock = threading.Lock()
        self.camera_wanted = set()
        self.camera_subs = {}
        self.camera_frames = {}     # stream -> (msg, monotonic arrival)
        self.camera_counts = {}     # stream -> frames since subscribing

        # Latest robot state, and the targets the page is asking for.
        self.state_lock = threading.Lock()
        self.joints = {}        # name -> {name, position, velocity, effort}
        self.last_state_time = 0.0
        self.state_count = 0
        self.targets = {}
        self.targets_dirty = False
        # The Zero pose button's motion, under state_lock: while it runs it
        # owns the targets above and the sliders are refused (zero_pose.py).
        self.zero = ZeroPoseState()
        # The setpoint the controller writes to the motors, {name: value},
        # with its state and arrival: where the arm is held. With no gravity
        # compensation the measured pose hangs a little below it, so a
        # command that starts from the measured pose lets the arm drop.
        self.setpoint = {}
        self.setpoint_state = None
        self.setpoint_time = None
        # The proxy's active_source, from its parameter events and from the
        # switches made here; None until known.
        self.proxy_source = None
        # One switch of the proxy's source at a time, from any page or
        # thread: a zero pose handing the arm back must not land after a
        # click that picked another source.
        self.source_lock = threading.Lock()

        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                              reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)

        # State arrives split the way commands are sent: the arm joints in one
        # JointState, each gripper on its own topic. The page wants one table
        # of sixteen, so they are merged back here.
        self.create_subscription(JointState, "/openarm/joint_states",
                                 self._on_state, reliable)
        self.create_subscription(
            Float64MultiArray, "/openarm/left_gripper",
            lambda msg: self._on_gripper(msg, "left_gripper"), reliable)
        self.create_subscription(
            Float64MultiArray, "/openarm/right_gripper",
            lambda msg: self._on_gripper(msg, "right_gripper"), reliable)
        self.create_subscription(JointState, "/openarm/controller/joint_commands",
                                 self._on_setpoint, best_effort)

        source = str(value("source_name"))
        self.source_name = source          # this dashboard's command source
        self.arm_pub = self.create_publisher(
            Float64MultiArray, f"/openarm/{source}/joint_commands", best_effort)
        self.left_gripper_pub = self.create_publisher(
            Float64MultiArray, f"/openarm/{source}/left_gripper", best_effort)
        self.right_gripper_pub = self.create_publisher(
            Float64MultiArray, f"/openarm/{source}/right_gripper", best_effort)

        # The policies the page drives, each by the proxy source it commands
        # as: the ACT card and the SmolVLA card.
        self.policies = {
            "act": PolicyLink(self, "act", "ACT", str(value("act_node")),
                              str(value("act_prefix")),
                              str(value("trajectory_topic")), reliable),
            "smolvla": PolicyLink(self, "smolvla", "SmolVLA",
                                  str(value("smolvla_node")),
                                  str(value("smolvla_prefix")),
                                  str(value("smolvla_trajectory_topic")),
                                  reliable),
        }
        self.act_node = self.policies["act"].node_name

        # Parameter clients, so the page drives the other nodes the same way
        # `ros2 param` would.
        self.set_clients = {}
        self.get_clients = {}
        # Each camera's preprocessor (openarm_camera), whose `source` the
        # dashboard's Switch swaps.
        self.preprocessor_nodes = {camera: f"/openarm_{camera}_preprocessor"
                                   for camera in self.cameras}
        for node_name in (self.controller_node, self.proxy_node,
                          *(policy.node_name for policy in self.policies.values()),
                          *self.preprocessor_nodes.values()):
            self.set_clients[node_name] = self.create_client(
                SetParameters, f"{node_name}/set_parameters")
            self.get_clients[node_name] = self.create_client(
                GetParameters, f"{node_name}/get_parameters")

        self.create_subscription(ParameterEvent, "/parameter_events",
                                 self._on_parameter_event,
                                 qos_profile_parameter_events)

        self.create_timer(1.0 / max(1.0, self.command_rate), self._send_targets)
        self.create_timer(0.2, self._follow_camera_wanted)
        self.get_logger().info(
            f"dashboard on http://localhost:{self.http_port} "
            f"(websocket {self.websocket_port}), source '{source}'")

    # ------------------------------------------------------------------ state

    def _on_state(self, msg):
        with self.state_lock:
            for index, name in enumerate(msg.name):
                self.joints[name] = {
                    "name": name,
                    "position": float(msg.position[index]) if index < len(msg.position) else 0.0,
                    "velocity": float(msg.velocity[index]) if index < len(msg.velocity) else 0.0,
                    "effort": float(msg.effort[index]) if index < len(msg.effort) else 0.0,
                }
            self.last_state_time = time.monotonic()
            self.state_count += 1

    def _on_gripper(self, msg, name):
        """data is [stroke in metres, stroke rate, effort]."""
        values = list(msg.data) + [0.0, 0.0, 0.0]
        with self.state_lock:
            self.joints[name] = {
                "name": name,
                "position": float(values[0]),
                "velocity": float(values[1]),
                "effort": float(values[2]),
            }

    def _on_setpoint(self, msg):
        with self.state_lock:
            for name, position in zip(msg.name, msg.position):
                self.setpoint[name] = float(position)
            self.setpoint_state = msg.header.frame_id
            self.setpoint_time = time.monotonic()

    def _held(self, name):
        """Where joint `name` is held (state_lock held): the controller's
        setpoint while the motors hold one, else the measured position."""
        if self.setpoint_state in HOLDING_STATES and name in self.setpoint and \
                self.setpoint_time is not None and \
                time.monotonic() - self.setpoint_time <= SETPOINT_STALE_S:
            return self.setpoint[name]
        joint = self.joints.get(name)
        return None if joint is None else joint["position"]

    def state_snapshot(self):
        """What the page needs to draw one frame, in JOINT_NAMES order."""
        with self.state_lock:
            if not self.joints:
                return {"joints": [], "age": None}
            age = time.monotonic() - self.last_state_time
            joints = [self.joints[name] for name in JOINT_NAMES
                      if name in self.joints]
        return {"joints": joints, "age": age}

    # --------------------------------------------------------------- commands

    def set_target(self, name, position):
        """Remember what the page wants; the timer does the publishing.

        Sliders fire far faster than the arm needs, and a target that is
        superseded before the next tick never has to reach the bus.
        -> None, or why the target was refused.
        """
        if name not in JOINT_NAMES:
            return f"unknown joint {name!r}"
        with self.state_lock:
            # Checked under the lock the zero pose starts under, so a slider
            # cannot slip in between the check and the motion.
            if self.zero.busy:
                return "the arm is going to the zero pose; stop it first"
            self.targets[JOINT_NAMES.index(name) + 1] = float(position)
            self.targets_dirty = True
        return None

    def _send_targets(self):
        with self.state_lock:
            if self.zero.phase == MOVING:
                targets = self._zero_tick()
                if targets is None:
                    return
            elif self.zero.busy or not self.targets_dirty or not self.targets:
                return
            else:
                targets = dict(self.targets)
                self.targets_dirty = False

        if any(motor_id in targets for motor_id in ARM_JOINT_IDS):
            # The arm topic carries all fourteen, so anything the page has
            # not touched holds the position it was last given -- or, never
            # given one, where it is held now; never a made-up 0.
            missing = [m for m in ARM_JOINT_IDS if m not in targets]
            if missing:
                with self.state_lock:
                    for motor_id in missing:
                        held = self._held(JOINT_NAMES[motor_id - 1])
                        if held is None:
                            return      # nothing safe to send for it yet
                        targets[motor_id] = self.targets[motor_id] = held
            data = [targets[motor_id] for motor_id in ARM_JOINT_IDS]
            self.arm_pub.publish(Float64MultiArray(data=data))
        if LEFT_GRIPPER_ID in targets:
            self.left_gripper_pub.publish(
                Float64MultiArray(data=[targets[LEFT_GRIPPER_ID]]))
        if RIGHT_GRIPPER_ID in targets:
            self.right_gripper_pub.publish(
                Float64MultiArray(data=[targets[RIGHT_GRIPPER_ID]]))

    def seed_targets_from_state(self):
        """Start the sliders where the arm is, so nothing jumps on first move.

        Called when a page connects and again whenever ui is selected: the
        arm may have been moved by teleop or ACT since, and the next slider
        sends all fourteen arm joints, not only the one that moved.
        """
        snapshot = self.state_snapshot()
        with self.state_lock:
            self._seed_locked()
        return snapshot

    def _seed_locked(self):
        """The targets onto where the arm is held (state_lock held)."""
        if self.zero.busy:
            return                      # the zero pose owns the targets
        for motor_id, name in enumerate(JOINT_NAMES, start=1):
            held = self._held(name)
            if held is not None:
                self.targets[motor_id] = held
        self.targets_dirty = False      # nothing to send until a slider moves

    # -------------------------------------------------------------- zero pose

    def go_to_zero(self):
        """Ease every arm joint to 0 from wherever the arm is, as ui.

        Blocks on service calls: call it off the executor thread.
        -> (ok, message).
        """
        with self.state_lock:
            token, problem = self.zero.claim()
        if problem:
            return False, problem
        try:
            ok, message = self._start_zero(token)
        except Exception as error:      # noqa: BLE001 -- the claim must end
            ok, message = False, f"zero pose failed: {error}"
        if not ok:
            with self.state_lock:
                self.zero.finish(token, False, message)
        return ok, message

    def _start_zero(self, token):
        controller = self.get_remote_parameters(
            self.controller_node, ["motors_enabled"])
        if not controller:
            return False, "the controller is not running"
        if not controller.get("motors_enabled"):
            return False, "the motors are off: connect first"
        proxy = self.get_remote_parameters(
            self.proxy_node, ["active_source", "sources"])
        if proxy is None:
            return False, f"{self.proxy_node} is not reachable"
        if self.source_name not in (proxy.get("sources") or []):
            return False, f"the proxy has no {self.source_name!r} source"
        previous = proxy.get("active_source")
        with self.state_lock:
            _, problem = self._arm_pose()
        if problem:
            return False, problem

        # Cut the old source off first, so nothing else reaches the arm
        # while the motion is planned from where it stands.
        ok, reason = self.set_proxy_source(self.source_name, zero_token=token)
        if not ok:
            return False, f"the proxy would not take {self.source_name}: {reason}"
        # A policy still executing behind the proxy would take over again
        # the moment its source is picked; stop it the way its card's Stop
        # does.
        for policy in self.policies.values():
            status = policy.snapshot()["status"]
            if status and status.get("mode") == "execute":
                self.set_remote_parameter(policy.node_name, "mode", "preview")
                self.set_remote_parameter(policy.node_name, "running", False)

        with self.state_lock:
            pose, problem = self._arm_pose()
            if problem:
                return False, problem
            if self.proxy_source != self.source_name:
                return False, f"{self.proxy_source} took the arm"
            goal = {motor_id: 0.0 for motor_id in ARM_JOINT_IDS}
            motion = ZeroPoseMotion(pose, goal, time.monotonic(),
                                    speed=self.zero_speed,
                                    min_duration=self.zero_min_duration)
            # Back to the sliders when they were driving; otherwise to no
            # one: teleop or a policy would pull the arm away again.
            restore = (self.source_name if previous == self.source_name
                       else NO_SOURCE)
            problem = self.zero.begin(token, motion, restore)
            if problem:
                return False, problem
            self.targets_dirty = False
        self.get_logger().info(
            f"zero pose: {motion.duration:.1f} s from {previous}")
        return True, f"going to the zero pose ({motion.duration:.1f} s)"

    def _arm_pose(self):
        """-> ({id: position} of the arm joints, None) or (None, why not).
        Call under state_lock."""
        if not self.joints:
            return None, "no joint state from the robot yet"
        age = time.monotonic() - self.last_state_time
        if age > ZERO_STATE_TIMEOUT_S:
            return None, f"the robot's joint state is {age:.1f} s old"
        pose = {}
        for motor_id in ARM_JOINT_IDS:
            name = JOINT_NAMES[motor_id - 1]
            if name not in self.joints:
                return None, f"no state for {name}"
            # From the setpoint the motors hold, not the sagging measured
            # pose: the first command then changes nothing.
            pose[motor_id] = self._held(name)
        return pose, None

    def _zero_tick(self):
        """One step of the motion, under state_lock: -> targets to send,
        or None when it ended this tick."""
        now = time.monotonic()
        if now - self.last_state_time > ZERO_STATE_TIMEOUT_S:
            token = self.zero.arrived()
            self._zero_end_async(token, False,
                                 "zero pose stopped: lost the robot's joint "
                                 "state; the arm holds", hand_back=False)
            return None
        sample, done = self.zero.motion.sample(now)
        self.targets.update(sample)
        if done:
            token = self.zero.arrived()
            self._zero_end_async(token, True, "at the zero pose",
                                 hand_back=True)
        # Only the arm: the grippers keep what they were last told.
        return {motor_id: self.targets[motor_id] for motor_id in ARM_JOINT_IDS}

    def _zero_end_async(self, token, ok, message, hand_back):
        # Service calls cannot wait on the executor thread they need.
        threading.Thread(target=self._zero_end,
                         args=(token, ok, message, hand_back),
                         daemon=True).start()

    def _zero_end(self, token, ok, message, hand_back):
        """Hand the source back and close the claim `token`."""
        with self.source_lock:
            with self.state_lock:
                if token != self.zero.token or self.zero.phase != FINISHING:
                    return              # someone else ended it meanwhile
                restore = self.zero.restore
            if hand_back and restore and restore != self.source_name and \
                    self.proxy_source == self.source_name:
                switched, _ = self._switch_proxy(restore)
                if switched:
                    message += "; no source selected, the arm holds"
            with self.state_lock:
                self.zero.finish(token, ok, message)
        self.get_logger().info(message)

    def zero_stop(self):
        """Stop the zero pose where the arm is; -> (ok, message)."""
        with self.state_lock:
            if not self.zero.busy:
                return False, "the arm is not going to the zero pose"
            token = self.zero.arrived()
        self._zero_end(token, True, "zero pose stopped; the arm holds",
                       hand_back=True)
        return True, "zero pose stopped; the arm holds"

    def zero_cancel(self, why):
        """End a zero pose without touching the source, e.g. on disconnect."""
        with self.state_lock:
            self.zero.finish(None, False, f"zero pose stopped: {why}")

    def zero_snapshot(self):
        with self.state_lock:
            return self.zero.snapshot(time.monotonic())

    # ---------------------------------------------------------- proxy source

    def set_proxy_source(self, source, zero_token=None):
        """Pick the proxy's active_source; -> (ok, reason).

        Every switch made from the page comes through here, one at a time.
        Any but the zero pose's own (zero_token) ends a zero pose in
        progress: the arm is someone else's now.
        """
        with self.source_lock:
            with self.state_lock:
                if zero_token is None:
                    self.zero.finish(None, False,
                                     f"zero pose stopped: {source} selected")
                elif zero_token != self.zero.token or not self.zero.busy:
                    return False, "the zero pose was cancelled"
            return self._switch_proxy(source)

    def _switch_proxy(self, source):
        """Under source_lock."""
        ok, reason = self.set_remote_parameter(
            self.proxy_node, "active_source", source)
        if ok:
            with self.state_lock:
                self.proxy_source = source
        return ok, reason

    def _on_parameter_event(self, event):
        if event.node != self.proxy_node:
            return
        for parameter in list(event.new_parameters) + list(event.changed_parameters):
            if parameter.name != "active_source":
                continue
            source = parameter.value.string_value
            with self.state_lock:
                if source == self.source_name and \
                        self.proxy_source != self.source_name:
                    # Picked anywhere -- this page, another, a terminal: the
                    # sliders start from where the arm is held now.
                    self._seed_locked()
                self.proxy_source = source
                # Picked elsewhere (a terminal, another tool): it is theirs.
                if self.zero.phase == MOVING and source != self.source_name:
                    self.zero.finish(None, False,
                                     f"zero pose stopped: {source} took the arm")

    # ------------------------------------------------------------- parameters

    def set_remote_parameter(self, node_name, name, value):
        """Set one parameter on another node, as `ros2 param set` does."""
        client = self.set_clients.get(node_name)
        if client is None or not client.wait_for_service(timeout_sec=1.0):
            return False, f"{node_name} is not reachable"

        parameter = Parameter()
        parameter.name = name
        if isinstance(value, bool):
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_BOOL, bool_value=value)
        elif isinstance(value, int):
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        elif isinstance(value, float):
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=value)
        else:
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_STRING, string_value=str(value))

        request = SetParameters.Request(parameters=[parameter])
        future = client.call_async(request)
        if not self._spin_until(future, 2.0):
            return False, f"{node_name} did not answer"
        results = future.result().results
        if results and not results[0].successful:
            return False, results[0].reason or "rejected"
        return True, "ok"

    def get_remote_parameters(self, node_name, names):
        client = self.get_clients.get(node_name)
        if client is None or not client.wait_for_service(timeout_sec=0.3):
            return None
        future = client.call_async(GetParameters.Request(names=names))
        if not self._spin_until(future, 1.0):
            return None
        values = {}
        for name, value in zip(names, future.result().values):
            values[name] = parameter_value_to_python(value)
        return values

    def _spin_until(self, future, timeout):
        """Wait for a service call from a thread that is not the executor."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.01)
        return False

    # --------------------------------------------------------------- policies

    def policy(self, name):
        """The PolicyLink for `name` (act, smolvla), or None."""
        return self.policies.get(name)

    # ----------------------------------------------------------------- camera

    def want_camera(self, streams):
        """Which streams some page is showing; any thread may call this."""
        with self.camera_lock:
            self.camera_wanted = set(streams)

    def _follow_camera_wanted(self):
        """Subscribe to what is wanted and drop the rest.

        Runs as a timer so subscriptions are only ever created and destroyed
        on the executor's own thread, never underneath a spin in progress.
        """
        with self.camera_lock:
            wanted = set(self.camera_wanted)
        camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        for stream in wanted - set(self.camera_subs):
            self.camera_subs[stream] = self.create_subscription(
                Image, self.camera_topics[stream],
                lambda msg, stream=stream: self._on_camera(stream, msg),
                camera_qos)
            self.get_logger().info(
                f"camera view on: {self.camera_topics[stream]}")
        for stream in set(self.camera_subs) - wanted:
            self.destroy_subscription(self.camera_subs.pop(stream))
            with self.camera_lock:
                self.camera_frames.pop(stream, None)
                self.camera_counts.pop(stream, None)
            self.get_logger().info(
                f"camera view off: {self.camera_topics[stream]}")

    def _on_camera(self, stream, msg):
        with self.camera_lock:
            self.camera_frames[stream] = (msg, time.monotonic())
            self.camera_counts[stream] = self.camera_counts.get(stream, 0) + 1

    def camera_snapshot(self):
        """{stream: (msg or None, age in s or None, frames so far)}."""
        now = time.monotonic()
        snapshot = {}
        with self.camera_lock:
            for stream in self.camera_wanted:
                frame = self.camera_frames.get(stream)
                if frame is None:
                    snapshot[stream] = (None, None, 0)
                else:
                    snapshot[stream] = (frame[0], now - frame[1],
                                        self.camera_counts.get(stream, 0))
        return snapshot

    # ----------------------------------------------------------------- status

    def status(self):
        """Who is connected, to what, and is it answering."""
        with self.state_lock:
            age = (time.monotonic() - self.last_state_time
                   if self.joints else None)
            count = self.state_count

        # Which profile is running decides which gains to ask for, so this
        # takes two rounds rather than one.
        head = self.get_remote_parameters(
            self.controller_node,
            ["left_interface", "right_interface", "motors_enabled",
             "motors_answering", "arms", "right_id_offset", "profiles",
             "active_profile", "interpolator", "interp_w"]) or {}
        controller = dict(head)
        profile = head.get("active_profile")
        if profile:
            wanted = []
            for name in JOINT_NAMES:
                fields = (("speed", "effort") if name.endswith("gripper")
                          else ("kp", "kd"))
                wanted += [f"gains.{profile}.{name}.{field}" for field in fields]
            controller.update(
                self.get_remote_parameters(self.controller_node, wanted) or {})
        proxy = self.get_remote_parameters(
            self.proxy_node, ["active_source", "sources"]) or {}
        if proxy.get("active_source") is not None:
            with self.state_lock:
                # Events keep it current; this only fills it in at first.
                if self.proxy_source is None:
                    self.proxy_source = proxy["active_source"]

        # Both arms share one bus in simulation, so the pair is deduplicated
        # rather than reported as "vcan0, vcan0".
        interfaces = []
        for name in (controller.get("left_interface"),
                     controller.get("right_interface")):
            if name and name not in interfaces:
                interfaces.append(name)
        # A virtual bus means the simulator is standing in for the robot.
        simulated = bool(interfaces) and all(
            str(name).startswith("vcan") for name in interfaces)

        return {
            "controller_up": bool(controller),
            "proxy_up": bool(proxy),
            "interfaces": interfaces,
            "mode": "sim" if simulated else ("real" if interfaces else "unknown"),
            "motors_enabled": controller.get("motors_enabled"),
            "motors_answering": controller.get("motors_answering"),
            "arms": controller.get("arms"),
            "profiles": controller.get("profiles") or [],
            "active_profile": profile,
            "interpolator": controller.get("interpolator"),
            "interp_w": controller.get("interp_w"),
            "gains": {
                name: {
                    field: controller.get(f"gains.{profile}.{name}.{field}")
                    for field in (("speed", "effort") if name.endswith("gripper")
                                  else ("kp", "kd"))
                }
                for name in JOINT_NAMES
            } if profile else {},
            "active_source": proxy.get("active_source"),
            "sources": proxy.get("sources") or [],
            "state_age": age,
            "state_count": count,
            "cameras": self.camera_sources(),
        }

    def camera_sources(self):
        """{camera: realsense | virtual | None}: who publishes its frames --
        openarm_camera's preprocessor on a RealSense, its virtual (black)
        camera, or no one."""
        sources = {}
        for camera in self.cameras:
            infos = self.get_publishers_info_by_topic(
                self.camera_topics[f"{camera}/rgb"])
            if not infos:
                sources[camera] = None
            elif all(VIRTUAL_CAMERA_NODE in info.node_name for info in infos):
                sources[camera] = "virtual"
            else:
                sources[camera] = "realsense"
        return sources


def parameter_value_to_python(value):
    kinds = {
        ParameterType.PARAMETER_BOOL: "bool_value",
        ParameterType.PARAMETER_INTEGER: "integer_value",
        ParameterType.PARAMETER_DOUBLE: "double_value",
        ParameterType.PARAMETER_STRING: "string_value",
        ParameterType.PARAMETER_STRING_ARRAY: "string_array_value",
        ParameterType.PARAMETER_DOUBLE_ARRAY: "double_array_value",
    }
    field = kinds.get(value.type)
    if field is None:
        return None
    result = getattr(value, field)
    return list(result) if field.endswith("array_value") else result
