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

"""Run the ACT policy: preview what it would do, then let it drive.

    /openarm/camera/<name>/{rgb,depth}          (openarm_camera), for each
                                                camera the policy reads:
                                                policy.cameras, chest first
    /openarm/joint_states, /openarm/*_gripper   (openarm_controller)
        -> policy: one observation in, a chunk of 50 actions out
        -> chunker: temporal ensembling, one action per step at 50 Hz
        -> preview:  /openarm/act/preview, /openarm/trajectory
        -> execute:  /openarm/act/joint_commands, /openarm/act/*_gripper
        -> otherwise, every step (idle_output): the pose held while the proxy
           has act selected, else the measured pose, on the same topics

It starts idle: the model is loaded but nothing is inferred. Two ways on:

  one shot   the ~/generate service (Generate trajectory on the dashboard)
             infers once from the live camera and joint state, publishes
             the chunk on /openarm/trajectory -- a JointTrajectory of 50
             steps from the moment the observation was taken, drawn as an
             arrow per gripper -- keeps it as the plan, and rests again.
             `mode: execute` (Run trajectory) then runs that plan once,
             one step per tick, and stops: the arm holds its last step.
             ~/reset (Reset to zero) eases the arm to reset_pose: every
             arm joint at 0, the grippers open.
  continuous `running: true` (Start) runs act_pipeline's
             tools/infer_robot.py loop: infer, ensemble the chunks
             (exp(-m * age), the newest heaviest), smooth (smoothing.py:
             per-joint speed and acceleration limits) and send one step.
             Each chunk goes out on /openarm/trajectory, the step's action
             on /openarm/act/preview. `mode: execute` (Execute) starts from
             a fresh chunk's first action and runs until Stop (the arm
             holds), a timeout, or the arm is back home and still; those
             two ease it home (home_on_finish) and stop inferring.

Nothing is sent towards the arm outside `mode: execute`, and even then the
proxy decides whether it reaches the arm (`active_source act`). Switching
to execute first eases the arm from where it is onto the prediction with a
cosine S-curve, as act_pipeline's infer_robot.py does, so the first command
is never a jump.

Two loops, because inference may be slower than a control step. With
`loop_mode: async` (the default) a 50 Hz timer advances the step counter on
the wall clock and takes the chunker's action for it, while a worker thread
takes the newest observation, runs the policy and files the chunk under the
step the observation was taken at: on a GPU a chunk a step, exactly
infer_robot.py; on a CPU a few a second, the arm still at the recorded
speed, waiting (decelerating onto the last action) only when the chunks run
out before the next arrives. `loop_mode: sync` is infer_robot.py's loop as
written: one inference, one step, at whatever rate the model manages.

    ros2 launch openarm_act act.launch.py checkpoint:=policy.onnx
    ros2 service call /openarm_act/generate std_srvs/srv/Trigger
    ros2 param set /openarm_act mode execute     # run the trajectory once
    ros2 service call /openarm_act/reset std_srvs/srv/Trigger   # zero, grippers open
    ros2 param set /openarm_act running true     # or: the policy, previewing
    ros2 param set /openarm_act mode execute     # after watching the preview
    ros2 param set /openarm_act mode preview     # stop commanding; arm holds
    ros2 param set /openarm_act running false    # stop inferring
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor, ParameterEvent, SetParametersResult
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_parameter_events)
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .chunking import ActionChunker
from .smoothing import TrajectorySmoother
from .models import list_models
from .policy import NUM_JOINTS, close_grippers, load_policy

# The dataset's joint order, which is the policy's input and output order.
JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]
LEFT_GRIPPER = JOINT_NAMES.index("left_gripper")
RIGHT_GRIPPER = JOINT_NAMES.index("right_gripper")
# The fourteen values a source's joint_commands carries: left j1..j7, right
# j1..j7.
ARM_INDICES = [i for i, name in enumerate(JOINT_NAMES)
               if not name.endswith("gripper")]
MODES = ("preview", "execute")
LOOP_MODES = ("async", "sync")
# Steps a late async tick may catch up at once (0.2 s at 50 Hz); later than
# that it is a stall, and the run resumes from now instead of sprinting.
MAX_CATCH_UP = 10
CAMERA_FALLBACKS = ("mock", "off")
CAMERA_KINDS = ("rgb", "depth")
# A frame from openarm_camera's virtual (black) camera, which stands in for
# one that is not plugged in, has a frame_id ending so.
VIRTUAL_SUFFIX = "_virtual"
# What the policy saw of a camera, from best to worst.
CAMERA_STATES = ("live", "virtual", "mock")
# A prediction older than this cannot be what execute starts from.
PREDICTION_FRESH_S = 0.5
# The controller's setpoint is where the arm is held while it is this fresh
# and the motors hold it (frame_id commanding or holding).
SETPOINT_STALE_S = 0.25
HOLDING_STATES = ("commanding", "holding")


class ActNode(Node):
    """Observation in, a previewed or executed joint target per step out."""

    def __init__(self, node_name="openarm_act"):
        # Another policy runs on this node by subclassing it under its own
        # name (openarm_smolvla): the hooks are _extra_policy_options,
        # _model_problem, _models and _status.
        super().__init__(node_name)

        fixed = ParameterDescriptor(
            read_only=True, description="set at launch; cannot be changed later")
        # The model and how it is loaded.
        self.declare_parameter("policy", "openarm_act.policy:OnnxPolicy",
                               ParameterDescriptor(
                                   read_only=True,
                                   description="policy class as module:Class"))
        # Settable while the node is idle: the new model loads beside the
        # old one, which keeps answering until the swap (_load_model).
        self.declare_parameter("checkpoint", "", ParameterDescriptor(
            description="the model file; set it while nothing runs to load "
                        "another (its stats beside it or inside it)"))
        self.declare_parameter("stats_file", "", ParameterDescriptor(
            read_only=True,
            description="act_pipeline's dataset_stats.pkl; not needed for an "
                        ".onnx made by utils/export_onnx.py"))
        self.declare_parameter("device", "auto", ParameterDescriptor(
            read_only=True,
            description="auto (cuda when the runtime has it, else cpu), "
                        "cuda, cuda:<n> or cpu"))
        self.declare_parameter("input_names", "", ParameterDescriptor(
            read_only=True,
            description="ONNX inputs as image=<name>,qpos=<name>; empty "
                        "matches them by name or rank"))
        self.declare_parameter("act_pipeline_path", "", ParameterDescriptor(
            read_only=True,
            description="directory holding act_pipeline, for ActPipelinePolicy"))
        self.declare_parameter("gripper_unit", "auto", ParameterDescriptor(
            read_only=True,
            description="the model's gripper unit: auto, stroke_m or rad"))
        self.declare_parameter("cpu_threads", 4, ParameterDescriptor(
            read_only=True,
            description="CPU threads inference may use; the rest stay free "
                        "for the controller. 0 lets the runtime take all"))
        self.declare_parameter("chunk_size", 50, fixed)
        # How the observation is prepared: as the recorder writes a dataset.
        self.declare_parameter("image_width", 640, fixed)
        self.declare_parameter("image_height", 480, fixed)
        self.declare_parameter("depth_scale", 0.001, ParameterDescriptor(
            read_only=True, description="metres per unit of a float depth image"))
        self.declare_parameter("depth_min_mm", 200, fixed)
        self.declare_parameter("depth_max_mm", 1200, fixed)
        # Topics.
        # Each camera the policy reads is <camera_prefix>/<name>/{rgb,depth}.
        self.declare_parameter("camera_prefix", "/openarm/camera", fixed)
        self.declare_parameter("state_topic", "/openarm/joint_states", fixed)
        # What the controller writes to the motors: holding the arm, or
        # easing it in, starts there rather than at the measured pose, which
        # hangs below it with no gravity compensation.
        self.declare_parameter("setpoint_topic", "/openarm/controller/joint_commands",
                               fixed)
        self.declare_parameter("left_gripper_topic", "/openarm/left_gripper", fixed)
        self.declare_parameter("right_gripper_topic", "/openarm/right_gripper", fixed)
        self.declare_parameter("output_prefix", "/openarm/act", fixed)
        self.declare_parameter("trajectory_topic", "/openarm/trajectory",
                               ParameterDescriptor(
                                   read_only=True,
                                   description="each inference's chunk as a "
                                               "JointTrajectory"))
        # Timing.
        self.declare_parameter("control_rate", 50.0, ParameterDescriptor(
            read_only=True,
            description="Hz; must be the dataset's rate, one action per step"))
        self.declare_parameter("query_period", 1, ParameterDescriptor(
            description="steps between two inferences; 1 = as often as the "
                        "model keeps up, as infer_robot.py does"))
        self.declare_parameter("temporal_ensemble", True, ParameterDescriptor(
            description="average overlapping chunks instead of taking the newest"))
        self.declare_parameter("ensemble_k", 0.01, ParameterDescriptor(
            description="ensemble weight decay m in exp(-m * age)"))
        self.declare_parameter("ensemble_weighting", "newest", ParameterDescriptor(
            description="newest (act_pipeline) or oldest (the ACT paper) "
                        "weighs most"))
        self.declare_parameter("loop_mode", "async", ParameterDescriptor(
            description="async: the arm steps at control_rate on the newest "
                        "ensembled chunks while inference runs beside it; "
                        "sync: one inference, one step, as infer_robot.py "
                        "loops (as fast as the model)"))
        self.declare_parameter("vel_scale", 1.0, ParameterDescriptor(
            description="scales infer_robot.py's per-joint speed and "
                        "acceleration limits on the policy's commands, 0.1..2"))
        # A model trained on the measured joints predicts a grasp as the
        # object's width, which squeezes nothing (policy.close_grippers).
        self.declare_parameter("gripper_close_below", 0.0, ParameterDescriptor(
            description="m; a predicted gripper stroke under this is a grasp "
                        "and is commanded to gripper_close_to instead, as "
                        "teleop closed it; 0 sends the prediction as is"))
        self.declare_parameter("gripper_close_to", 0.0, ParameterDescriptor(
            description="m; the stroke a grasp is commanded to (0: shut, the "
                        "gripper's force limit holding the object)"))
        self.declare_parameter("max_obs_age", 0.25, ParameterDescriptor(
            description="seconds; an older image or joint state is not used"))
        self.declare_parameter("camera_fallback", "mock", ParameterDescriptor(
            description="with no camera: mock feeds a gray frame at 600 mm, "
                        "as act_pipeline's infer_robot.py does; off does "
                        "not infer"))
        # Preview and execute.
        self.declare_parameter("mode", "preview", ParameterDescriptor(
            description="preview publishes the prediction only; execute "
                        "commands the arm"))
        # Zero, grippers open (43 mm): closed, the policy reads its task as done.
        zero_open = [0.0] * NUM_JOINTS
        zero_open[LEFT_GRIPPER] = zero_open[RIGHT_GRIPPER] = 0.043
        self.declare_parameter("reset_pose", zero_open, ParameterDescriptor(
            description="where ~/reset takes the arm: 16 values in the "
                        "dataset's joint order (act.yaml: zero, grippers open)"))
        self.declare_parameter("reset_speed", 0.5, ParameterDescriptor(
            description="rad/s (m/s on a gripper) the reset may average on "
                        "its farthest joint; it takes warmup_duration at least"))
        self.declare_parameter("warmup_duration", 1.5, ParameterDescriptor(
            description="seconds of S-curve from the arm onto the prediction "
                        "when execute starts"))
        self.declare_parameter("warmup_threshold", 0.03, ParameterDescriptor(
            description="largest gap (rad, m on a gripper) that needs no warm-up"))
        self.declare_parameter("max_start_jump", 1.0, ParameterDescriptor(
            description="refuse execute when the prediction is further than "
                        "this from the arm; 0 disables"))
        # When execute ends by itself, as infer_robot.py's loop does.
        self.declare_parameter("execute_timeout", 30.0, ParameterDescriptor(
            description="seconds of steps before it stops by itself "
                        "(infer_robot --max_timesteps / rate); 0 disables"))
        self.declare_parameter("stop_delta_rad", 0.008, ParameterDescriptor(
            description="done when the command stays this close to the arm..."))
        self.declare_parameter("stop_home_dist", 0.15, ParameterDescriptor(
            description="...the arm within this of finish_pose (norm)..."))
        self.declare_parameter("stop_duration", 1.0, ParameterDescriptor(
            description="...for this many seconds of steps; 0 disables"))
        self.declare_parameter("home_on_finish", True, ParameterDescriptor(
            description="once done or timed out, ease the arm to finish_pose "
                        "(infer_robot --safe_home_on_stop); Stop never does"))
        self.declare_parameter("finish_pose", [0.0] * NUM_JOINTS, ParameterDescriptor(
            description="infer_robot's home: 16 values in the dataset's order"))
        self.declare_parameter("finish_duration", 2.0, ParameterDescriptor(
            description="seconds the way home takes at least"))
        self.declare_parameter("plan_max_age", 30.0, ParameterDescriptor(
            description="seconds a generated trajectory may wait to be run; "
                        "older, the scene may have changed. 0 disables"))
        self.declare_parameter("plan_max_drift", 0.05, ParameterDescriptor(
            description="refuse to run a trajectory when the arm has moved "
                        "this far (rad, m on a gripper) since it was "
                        "generated; 0 disables"))
        self.declare_parameter("plan_max_start_jump", 0.15, ParameterDescriptor(
            description="max_start_jump for a generated trajectory, whose "
                        "ease-in is not drawn; 0 disables"))
        self.declare_parameter("running", False, ParameterDescriptor(
            description="true runs inference; false (the start) stops it and "
                        "drops pending chunks"))
        self.declare_parameter("idle_output", True, ParameterDescriptor(
            description="while not executing, keep publishing joint_commands "
                        "and the grippers every step: the pose held when the "
                        "proxy has this source selected, else the measured one"))
        self.declare_parameter("proxy_node", "/openarm_command_proxy", ParameterDescriptor(
            read_only=True,
            description="the command proxy, whose active_source idle_output follows"))

        param = lambda name: self.get_parameter(name).value  # noqa: E731
        self.image_size = (int(param("image_width")), int(param("image_height")))
        self.depth_scale = float(param("depth_scale")) or 0.001
        self.depth_range = (int(param("depth_min_mm")), int(param("depth_max_mm")))
        self.query_period = max(1, int(param("query_period")))
        self.max_obs_age = float(param("max_obs_age"))
        self.camera_fallback = str(param("camera_fallback"))
        if self.camera_fallback not in CAMERA_FALLBACKS:
            raise ValueError(f"camera_fallback must be one of {CAMERA_FALLBACKS}")
        # What infer_robot.py feeds the policy without a camera: mid-gray,
        # everything 600 mm away. Raw, so it is preprocessed like a real one.
        width, height = self.image_size
        self.mock_rgb = np.full((height, width, 3), 128, dtype=np.uint8)
        self.mock_depth = np.full((height, width), 600, dtype=np.uint16)
        # What the last observation saw, worst camera first (CAMERA_STATES),
        # and each camera's.
        self.camera = None
        self.camera_states = {}
        self.warmup_duration = float(param("warmup_duration"))
        self.reset_pose = np.asarray(param("reset_pose"), dtype=np.float64)
        if self.reset_pose.shape != (NUM_JOINTS,):
            raise ValueError(f"reset_pose needs {NUM_JOINTS} values")
        self.reset_speed = float(param("reset_speed"))
        self.warmup_threshold = float(param("warmup_threshold"))
        self.max_start_jump = float(param("max_start_jump"))
        self.execute_timeout = float(param("execute_timeout"))
        self.loop_mode = str(param("loop_mode"))
        if self.loop_mode not in LOOP_MODES:
            raise ValueError(f"loop_mode must be one of {LOOP_MODES}")
        self.vel_scale = float(param("vel_scale"))
        self.gripper_close_below = float(param("gripper_close_below"))
        self.gripper_close_to = float(param("gripper_close_to"))
        self.stop_delta_rad = float(param("stop_delta_rad"))
        self.stop_home_dist = float(param("stop_home_dist"))
        self.stop_duration = float(param("stop_duration"))
        self.home_on_finish = bool(param("home_on_finish"))
        self.finish_pose = np.asarray(param("finish_pose"), dtype=np.float64)
        if self.finish_pose.shape != (NUM_JOINTS,):
            raise ValueError(f"finish_pose needs {NUM_JOINTS} values")
        self.finish_duration = float(param("finish_duration"))
        self.idle_output = bool(param("idle_output"))
        self.plan_max_age = float(param("plan_max_age"))
        self.plan_max_drift = float(param("plan_max_drift"))
        self.plan_max_start_jump = float(param("plan_max_start_jump"))
        self.running = bool(param("running"))
        self.control_dt = 1.0 / float(param("control_rate"))
        if str(param("mode")) != "preview":
            # Executing straight from launch would skip the preview, which is
            # the point of having one.
            raise ValueError("mode must start as preview; switch to execute "
                             "after watching the prediction")

        spec = str(param("policy"))
        self.get_logger().info(f"loading {spec}")
        self.policy_spec = spec
        self.policy_options = {
            "checkpoint": str(param("checkpoint")),
            "stats_file": str(param("stats_file")),
            "device": str(param("device")),
            "input_names": str(param("input_names")),
            "act_pipeline_path": str(param("act_pipeline_path")),
            "gripper_unit": str(param("gripper_unit")),
            "chunk_size": int(param("chunk_size")),
            "cpu_threads": int(param("cpu_threads")),
            **self._extra_policy_options(),
        }
        self.policy = load_policy(spec, self.policy_options)
        self._check_image_size(self.policy)
        self.model_path = self.policy_options["checkpoint"]
        # A model being loaded in place of this one (its path), and how the
        # last load went, for the dashboard: {id, ok, message}.
        self.model_loading = None
        self.last_model_load = None
        # The .onnx files the dashboard may pick from, looked up now and
        # then rather than with every status.
        self.models_cache = (0.0, [])
        self.get_logger().info(self.policy.describe())
        self.chunker = ActionChunker(
            self.policy.chunk_size, bool(param("temporal_ensemble")),
            float(param("ensemble_k")), str(param("ensemble_weighting")))

        self.bridge = CvBridge()
        # Latest inputs with their arrival time (monotonic), written by the
        # subscriptions and read by the worker.
        self.input_lock = threading.Lock()
        # The cameras the policy reads, as it says (policy.cameras), and
        # their newest frames: "<camera>/<kind>" -> Image.
        self.cameras = self._policy_cameras(self.policy)
        self.frames = {}
        self.arm_state = None           # {name: position}
        self.gripper_state = {}         # {name: stroke}
        self.received = {}              # input name -> monotonic seconds

        # The step counter, the chunks and the execution state, shared by the
        # timer and the worker. `run_id` changes whenever inference starts or
        # stops, so a chunk computed for an earlier run is thrown away.
        self.step_cond = threading.Condition()
        self.step = 0
        self.run_id = 0
        self.last_query_step = None
        self.prediction = None          # (monotonic, action) for this step
        self.executing = False          # commands are being published
        self.warmup = None              # (start, q_from, q_to) while easing in
        self.execute_started = None     # monotonic, while executing
        # Continuous execute, as infer_robot.py's loop: the smoother between
        # the ensembled action and the arm, the steps sent, how many of the
        # last ones met the stop criterion, and whether the arm has been
        # away from home yet (it starts there; done means back there).
        self.smoother = None
        self.steps_executed = 0
        self.still_steps = 0
        self.left_home = False
        self.going_home = False
        self.chunk_times = []           # monotonic of recent continuous chunks
        self.last_chunk = None          # the newest chunk, raw, for execute
        # async: steps follow the wall clock from here, so a late tick
        # catches up instead of slowing the run down. (monotonic, step)
        self.step_origin = None
        self.last_command = None        # the newest command sent, 16 values
        # Not executing, the act topics still carry a command every step
        # (idle_output). The controller has no gravity compensation, so with
        # this source selected on the proxy that is a pose held once -- the
        # last command, or the measured pose when act was selected -- never
        # the measured pose again each step, which the arm would sag after.
        # Not selected, it is the measured pose: the proxy does not forward
        # it, and on a switch to act the arm then starts from where it is.
        self.proxy_source = None        # the proxy's active_source; None unknown
        self.hold_pose = None           # 16 values, while selected
        # What the worker is doing, and when the last tick ran: a tick much
        # later than control_dt is logged with it, so a starved control loop
        # says why.
        self.worker_phase = "idle"
        self.last_tick = None
        self.late_ticks = 0
        # Steps the run has waited for the model: its chunks ran out before
        # the next arrived (a CPU, slower than a chunk). The arm decelerates
        # onto the last action meanwhile rather than stopping dead.
        self.waited_steps = 0
        self.dropped_steps = 0          # steps a stalled tick skipped

        self.held_action = None
        self.stop_reason = None         # why the last execute ended
        self.waiting = "starting"       # why there is no prediction, or None
        self.latency_ms = None
        self.inferences = 0
        # One-shot generation: ~/generate infers once, publishes the chunk
        # and keeps it as the plan; mode execute then runs the plan, once.
        self.generate_requested = False
        self.generating = False
        self.plan = None                # (monotonic, actions[chunk, 16], qpos)
        self.playback = None            # (actions, next index) while running
        self.last_generate = None       # {"id", "ok", "message", "ms"}
        # What the playback running now is for.
        self.playback_done = None       # stop reason once it has run
        self.plan_source = None         # "model", or "ik" for ~/plan
        self.returning = False          # the playback is ~/reset's

        camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE)

        camera_prefix = str(param("camera_prefix")).rstrip("/")
        for camera in self.cameras:
            for kind in CAMERA_KINDS:
                self.create_subscription(
                    Image, f"{camera_prefix}/{camera}/{kind}",
                    lambda msg, key=f"{camera}/{kind}": self._store(key, msg),
                    camera_qos)
        self.create_subscription(JointState, param("state_topic"),
                                 self._on_state, state_qos)
        self.setpoint = None            # (monotonic, frame_id, {name: value})
        self.create_subscription(JointState, param("setpoint_topic"),
                                 self._on_setpoint, state_qos)
        self.create_subscription(
            Float64MultiArray, param("left_gripper_topic"),
            lambda msg: self._on_gripper("left_gripper", msg), state_qos)
        self.create_subscription(
            Float64MultiArray, param("right_gripper_topic"),
            lambda msg: self._on_gripper("right_gripper", msg), state_qos)

        prefix = str(param("output_prefix")).rstrip("/")
        self.arm_pub = self.create_publisher(
            Float64MultiArray, f"{prefix}/joint_commands", output_qos)
        self.gripper_pubs = {
            LEFT_GRIPPER: self.create_publisher(
                Float64MultiArray, f"{prefix}/left_gripper", output_qos),
            RIGHT_GRIPPER: self.create_publisher(
                Float64MultiArray, f"{prefix}/right_gripper", output_qos),
        }
        self.preview_pub = self.create_publisher(
            JointState, f"{prefix}/preview", output_qos)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, str(param("trajectory_topic")), output_qos)
        self.status_pub = self.create_publisher(
            String, f"{prefix}/status", output_qos)

        self.add_on_set_parameters_callback(self._on_set_parameters)

        # The proxy's active_source: asked once, then followed on its
        # parameter events (which a restarted proxy sends too).
        self.source_name = prefix.rsplit("/", 1)[-1]
        self.proxy_node = "/" + str(param("proxy_node")).strip("/")
        self.proxy_params = self.create_client(
            GetParameters, f"{self.proxy_node}/get_parameters")
        self.proxy_query = None
        self.create_subscription(ParameterEvent, "/parameter_events",
                                 self._on_parameter_event,
                                 qos_profile_parameter_events)
        self.create_timer(1.0, self._query_proxy_source)

        self.create_service(Trigger, "~/generate", self._on_generate)
        self.create_service(Trigger, "~/reset", self._on_reset)
        # A trajectory made elsewhere -- the dashboard's IK goal -- run like
        # one the model generated: drawn, approved, then Run.
        self.create_subscription(JointTrajectory, "~/plan", self._on_plan,
                                 output_qos)

        self.alive = True
        self.worker = threading.Thread(target=self._infer_loop, daemon=True)
        self.worker.start()
        self.create_timer(self.control_dt, self._tick)
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            f"{param('control_rate'):.0f} Hz, previewing on {prefix}/preview; "
            f"set mode execute to command {prefix}/joint_commands")

    # ------------------------------------------------------------------ inputs

    def _store(self, key, msg):
        with self.input_lock:
            self.frames[key] = msg
            self.received[key] = time.monotonic()

    def _on_state(self, msg):
        with self.input_lock:
            self.arm_state = dict(zip(msg.name, msg.position))
            self.received["state"] = time.monotonic()

    def _on_gripper(self, name, msg):
        if not msg.data:
            return
        with self.input_lock:
            self.gripper_state[name] = float(msg.data[0])
            self.received[name] = time.monotonic()

    def _on_setpoint(self, msg):
        with self.input_lock:
            self.setpoint = (time.monotonic(), msg.header.frame_id,
                             dict(zip(msg.name, msg.position)))

    def _held_qpos(self):
        """Where the arm is held, 16 values: the controller's setpoint over
        the measured pose while the motors hold one. A command that starts
        from the measured pose instead lets the arm drop by its sag."""
        qpos = self._current_qpos()
        if qpos is None:
            return None
        with self.input_lock:
            setpoint = self.setpoint
        if setpoint is None or setpoint[1] not in HOLDING_STATES or \
                time.monotonic() - setpoint[0] > SETPOINT_STALE_S:
            return qpos
        held = np.asarray(qpos, dtype=np.float64)
        for index, name in enumerate(JOINT_NAMES):
            if name in setpoint[2]:
                held[index] = setpoint[2][name]
        return held

    def _current_qpos(self):
        """The arm's 16 positions in topic units, or None if incomplete."""
        with self.input_lock:
            positions = dict(self.arm_state or {})
            positions.update(self.gripper_state)
        if any(name not in positions for name in JOINT_NAMES):
            return None
        return np.asarray([positions[name] for name in JOINT_NAMES],
                          dtype=np.float32)

    def _snapshot(self):
        """The newest observation, or (None, reason) when it is not usable.

        -> ((rgb, depth, qpos, others), None): the first camera's frames,
        and {camera: (rgb, depth)} for the others the policy reads. A camera
        with no frames is the mock one, if camera_fallback allows it; the
        joint state is never made up.
        """
        now = time.monotonic()
        mocked = {}                     # camera -> why
        with self.input_lock:
            for name in ("state", "left_gripper", "right_gripper"):
                problem = self._stale(name, now)
                if problem:
                    return None, problem
            for camera in self.cameras:
                for kind in CAMERA_KINDS:
                    problem = self._stale(f"{camera}/{kind}", now)
                    if problem is None:
                        continue
                    if self.camera_fallback != "mock":
                        return None, problem
                    mocked.setdefault(camera, problem)
            frames = {camera: (self.frames.get(f"{camera}/rgb"),
                               self.frames.get(f"{camera}/depth"))
                      for camera in self.cameras if camera not in mocked}

        qpos = self._current_qpos()
        if qpos is None:
            return None, "joint state is missing joints"
        if not np.all(np.isfinite(qpos)):
            return None, "joint state is not finite"
        states = {}
        images = {}
        for camera in self.cameras:
            if camera in mocked:
                states[camera] = "mock"
                images[camera] = (self.mock_rgb, self.mock_depth)
                continue
            rgb_msg, depth_msg = frames[camera]
            states[camera] = ("virtual" if rgb_msg.header.frame_id.endswith(
                VIRTUAL_SUFFIX) else "live")
            images[camera] = (self._decode_rgb(rgb_msg),
                              self._decode_depth(depth_msg))
        self._set_camera(states, mocked)
        first = self.cameras[0]
        others = {camera: images[camera] for camera in self.cameras[1:]}
        return (*images[first], qpos, others), None

    def _stale(self, name, now):
        """Why input `name` cannot be used, or None (input_lock held)."""
        when = self.received.get(name)
        if when is None:
            return f"no {name} received yet"
        if now - when > self.max_obs_age:
            # Worded without the age, so the log says it once rather than at
            # every step while it lasts.
            return f"no {name} for over {self.max_obs_age:g} s"
        return None

    def _set_camera(self, states, mocked):
        """Log when a camera turns mock or virtual, or comes back."""
        for camera, state in states.items():
            before = self.camera_states.get(camera)
            if state == before:
                continue
            if state == "mock":
                self.get_logger().warning(
                    f"{mocked[camera]}: the policy sees a mock {camera} frame "
                    f"(gray, 600 mm), as infer_robot.py does without a camera")
            elif state == "virtual":
                self.get_logger().warning(
                    f"{camera} is a virtual camera (black frames, no RealSense "
                    f"for it): the policy sees nothing there")
            elif before in ("mock", "virtual"):
                self.get_logger().info(f"{camera} is back: the policy sees it again")
        self.camera_states = states
        self.camera = max(states.values(), key=CAMERA_STATES.index) \
            if states else None

    def _blind(self):
        """The cameras the policy does not really see, or ""."""
        return ", ".join(f"{camera} {state}" for camera, state
                         in self.camera_states.items() if state != "live")

    @staticmethod
    def _policy_cameras(policy):
        """The cameras a policy reads, first the one predict()'s rgb and
        depth are from; chest for one that does not say."""
        cameras = tuple(getattr(policy, "cameras", None) or ("chest",))
        if len(set(cameras)) != len(cameras):
            raise ValueError(f"the policy names a camera twice: {cameras}")
        return cameras

    def _decode_rgb(self, msg):
        """RGB uint8 at the dataset's size, as the recorder stores it."""
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if (rgb.shape[1], rgb.shape[0]) != self.image_size:
            rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
        return np.asarray(rgb, dtype=np.uint8)

    def _decode_depth(self, msg):
        """Depth in mm, 0 kept for no reading, the rest clipped -- as recorded."""
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if (depth.shape[1], depth.shape[0]) != self.image_size:
            depth = cv2.resize(depth, self.image_size,
                               interpolation=cv2.INTER_NEAREST)
        if depth.dtype == np.float32:
            depth_mm = np.nan_to_num(depth, nan=0.0) / self.depth_scale
        else:
            depth_mm = depth.astype(np.float32)
        low, high = self.depth_range
        return np.where(depth_mm == 0, 0,
                        np.clip(depth_mm, low, high)).astype(np.uint16)

    # ---------------------------------------------------------------- inference

    def _due(self):
        """Whether the worker should start an inference now (cond held)."""
        if self.generate_requested:
            return True
        if not self.running:
            return False
        if self.last_query_step is None:
            return True
        if self.step == self.last_query_step:
            return False                # at most one inference per step
        if self.step - self.last_query_step >= self.query_period:
            return True
        # About to run dry: ask now rather than wait out the period.
        return self.chunker.remaining(self.step) <= self.query_period

    def _infer_loop(self):
        while True:
            with self.step_cond:
                self.step_cond.wait_for(lambda: not self.alive or self._due())
                if not self.alive:
                    return
                one_shot = self.generate_requested
                self.generate_requested = False
                self.generating = one_shot
                run_id = self.run_id
                # This iteration's model: a load swaps self.policy only
                # while nothing infers, but a reference keeps it whole.
                policy = self.policy
                if one_shot or self.last_query_step is None:
                    policy.reset()
                if not one_shot:
                    self.last_query_step = self.step

            self.worker_phase = "snapshot"
            observation, reason = self._snapshot()
            if observation is None:
                if one_shot:
                    self._finish_generate(False, f"cannot generate: {reason}")
                else:
                    self._set_waiting(reason)
                continue

            with self.step_cond:
                obs_step = self.step
            obs_stamp = self.get_clock().now()
            started = time.perf_counter()
            self.worker_phase = "predict"
            try:
                rgb, depth, qpos, others = observation
                # The other cameras only to a policy that reads them, so
                # one written for a single camera need not know about them.
                prediction = (policy.predict(rgb, depth, qpos, cameras=others)
                              if others else policy.predict(rgb, depth, qpos))
                actions = np.asarray(prediction, dtype=np.float64)
            except Exception as error:      # keep the node up for a retry
                if one_shot:
                    self._finish_generate(False, f"policy failed: {error}")
                else:
                    self._set_waiting(f"policy failed: {error}")
                continue
            elapsed = time.perf_counter() - started
            if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
                problem = (f"policy returned {actions.shape}, expected "
                           f"[chunk, {NUM_JOINTS}]")
                if one_shot:
                    self._finish_generate(False, problem)
                else:
                    self._set_waiting(problem)
                continue
            # What is drawn, ensembled and run alike.
            actions = close_grippers(actions, self.gripper_close_below,
                                     self.gripper_close_to)

            if one_shot:
                # Raw, not ensembled: there is no other chunk to blend with.
                # The trajectory goes out before the plan exists, and the
                # status saying it does right after, so the dashboard holds
                # the arrows by the time it hears there is a plan to show.
                self._publish_trajectory(obs_stamp, actions)
                with self.step_cond:
                    # The observed qpos too: run checks the arm is still there.
                    self.plan_source = "model"
                    self.plan = (time.monotonic(), actions,
                                 observation[2].astype(np.float64))
                    self.inferences += 1
                    self.latency_ms = elapsed * 1e3
                blind = (f" -- not seeing: {self._blind()}"
                         if self._blind() else "")
                self._finish_generate(
                    True, f"trajectory generated: {len(actions)} steps in "
                          f"{elapsed * 1e3:.0f} ms{blind}", elapsed * 1e3)
                continue

            self.worker_phase = "file"
            if self.loop_mode == "sync":
                self._sync_step(run_id, obs_stamp, actions, elapsed)
                self.worker_phase = "idle"
                continue

            with self.step_cond:
                if run_id != self.run_id:
                    continue
                if self.step - obs_step >= len(actions):
                    self._set_waiting(f"inference took {elapsed * 1e3:.0f} ms, "
                                      f"longer than the whole chunk")
                    continue
                self.chunker.add(obs_step, actions)
                self.last_chunk = actions
                # The chunk as it will be followed: each step ensembled with
                # the older chunks that still cover it, as _tick will take it.
                trajectory = [self.chunker.action(obs_step + ahead)
                              for ahead in range(len(actions))]
                self._count_chunk(elapsed)
            self.worker_phase = "publish"
            self._publish_trajectory(obs_stamp, trajectory)
            self._set_waiting(None)
            self.worker_phase = "idle"

    def _count_chunk(self, elapsed):
        """Latency and chunk rate of continuous inference (step_cond held)."""
        self.inferences += 1
        ms = elapsed * 1e3
        self.latency_ms = ms if self.latency_ms is None \
            else 0.9 * self.latency_ms + 0.1 * ms
        now = time.monotonic()
        self.chunk_times = [t for t in self.chunk_times if now - t < 5.0] + [now]

    def _sync_step(self, run_id, obs_stamp, actions, elapsed):
        """loop_mode sync: this chunk's step is taken, then the next one is.

        infer_robot.py's loop exactly: observe, infer, ensemble, smooth,
        send one step. The arm moves as fast as the model infers.
        """
        with self.step_cond:
            if run_id != self.run_id:
                return
            step = self.step
            self.chunker.add(step, actions)
            self.last_chunk = actions
            action = self.chunker.action(step)
            trajectory = [self.chunker.action(step + ahead)
                          for ahead in range(len(actions))]
            self.prediction = (time.monotonic(), action)
            executing = self.executing and self.playback is None
            warming = self.warmup is not None
            self.step += 1
            self.chunker.prune(self.step)
            self._count_chunk(elapsed)
            self.step_cond.notify_all()
        self._publish_trajectory(obs_stamp, trajectory)
        self._publish_preview(action)
        self._set_waiting(None)
        if executing and not warming:
            self._execute_step([action])

    def _on_generate(self, request, response):
        """~/generate: infer once and publish the trajectory, then rest.

        Answers at once; the result comes in the status (last_generate) and
        the log when the inference is done.
        """
        with self.step_cond:
            if self.running:
                problem = "continuous inference is running; set running false"
            elif self.executing:
                problem = "a trajectory is running; stop it first"
            elif self.generate_requested or self.generating:
                problem = "already generating"
            elif self.model_loading:
                problem = "a model is loading"
            else:
                problem = None
                self.generate_requested = True
                # The old plan goes now, so Run cannot pick it up while the
                # new one is on its way.
                self.plan = None
                self.step_cond.notify_all()
        response.success = problem is None
        response.message = problem or "generating"
        if problem is None:
            self.get_logger().info("generating a trajectory")
            # Now, so the dashboard drops the old plan's Run and arrows at
            # once instead of at the next status tick.
            self._publish_status()
        return response

    # Longest trajectory ~/plan takes: 30 s at the control rate.
    MAX_PLAN_S = 30.0

    def _on_plan(self, msg):
        """~/plan: keep an outside trajectory as the plan, as ~/generate does.

        Points are taken one per control step, whatever time_from_start
        says; the dashboard plans at the control rate. The result is
        reported the way a generation is, in last_generate.
        """
        try:
            columns = [list(msg.joint_names).index(name) for name in JOINT_NAMES]
            actions = np.asarray([[point.positions[c] for c in columns]
                                  for point in msg.points], dtype=np.float64)
        except (ValueError, IndexError):
            problem = f"a plan must name all of {', '.join(JOINT_NAMES)}"
            actions = None
        else:
            limit = int(self.MAX_PLAN_S / self.control_dt)
            if len(actions) < 2:
                problem = "a plan needs two points at least"
            elif len(actions) > limit:
                problem = f"a plan may last {self.MAX_PLAN_S:g} s at most"
            elif not np.all(np.isfinite(actions)):
                problem = "a plan must be finite"
            else:
                problem = None
        qpos = self._current_qpos()
        with self.step_cond:
            if problem is None:
                if self.running:
                    problem = "continuous inference is running; set running false"
                elif self.executing:
                    problem = "the arm is moving; stop it first"
                elif self.generate_requested or self.generating:
                    problem = "a trajectory is being generated; wait for it"
                elif qpos is None:
                    problem = "joint state is incomplete"
            if problem is None:
                self.plan_source = "ik"
                self.plan = (time.monotonic(), actions, qpos.astype(np.float64))
        if problem is not None:
            self._finish_generate(False, f"cannot take the plan: {problem}")
            return
        self._publish_trajectory(self.get_clock().now(), actions)
        self._finish_generate(
            True, f"trajectory planned to the goal: {len(actions)} steps "
                  f"({len(actions) * self.control_dt:.1f} s)")

    def _finish_generate(self, ok, message, ms=None):
        with self.step_cond:
            self.generating = False
            number = (self.last_generate or {}).get("id", 0) + 1
            self.last_generate = {"id": number, "ok": ok, "message": message,
                                  "ms": ms}
        # Now rather than at the next status tick: the dashboard enables Run
        # and shows the arrows off the status.
        self._publish_status()
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

    def _set_waiting(self, reason):
        """Log a change in why there is (or is not) a prediction."""
        if reason == self.waiting:
            return
        if reason is None:
            self.get_logger().info("predicting")
        else:
            self.get_logger().warning(f"not predicting: {reason}")
        self.waiting = reason

    # ------------------------------------------------------------------- output

    def _warmup_command(self, warmup):
        """The eased command while warming up, or None once it is over."""
        start, q_from, q_to, duration = warmup
        fraction = (time.monotonic() - start) / max(duration, 1e-3)
        if fraction < 1.0:
            # Cosine S-curve: zero velocity at both ends.
            blend = 0.5 - 0.5 * math.cos(math.pi * fraction)
            return q_from + (q_to - q_from) * blend
        with self.step_cond:
            self.warmup = None
        self.get_logger().info("warm-up done")
        return None

    def _tick_playback(self):
        """Run the generated trajectory: one step per tick, then stop."""
        with self.step_cond:
            warmup = self.warmup
        if warmup is not None:
            command = self._warmup_command(warmup)
            if command is not None:
                self._publish_command(command)
                return
        with self.step_cond:
            if self.playback is None:   # stopped meanwhile
                return
            actions, index = self.playback
            self.playback = (actions, index + 1)
        self._publish_command(actions[index])
        self._publish_preview(actions[index])
        if index + 1 < len(actions):
            return
        with self.step_cond:
            reason = self.playback_done or f"done: {len(actions)} steps"
            self._stop_execute(reason)
            self.stop_reason = reason
        threading.Thread(target=self._reset_mode_parameter, daemon=True).start()

    def _tick(self):
        now = time.monotonic()
        if self.last_tick is not None and now - self.last_tick > 3 * self.control_dt:
            self.late_ticks += 1
            self.get_logger().warning(
                f"control tick {1e3 * (now - self.last_tick):.0f} ms late "
                f"(worker: {self.worker_phase}); {self.late_ticks} so far",
                throttle_duration_sec=1.0)
        self.last_tick = now
        with self.step_cond:
            playing = self.playback is not None
        if playing:
            self._tick_playback()
            return
        with self.step_cond:
            idle = not self.executing
        if idle:
            self._publish_idle()
        with self.step_cond:
            if not self.running:
                return
            if self.loop_mode == "sync":
                # Steps follow inferences (_sync_step); only the ease-in onto
                # the first prediction runs on the clock.
                warmup = self.warmup if self.executing else None
            else:
                warmup = False
        if warmup:
            command = self._warmup_command(warmup)
            if command is not None:
                self._publish_command(command)
        if warmup is not False:
            return
        with self.step_cond:
            # The steps the clock says are due: one, or more after a late tick
            # (the CPU busy inferring), at most MAX_CATCH_UP.
            now = time.monotonic()
            if self.step_origin is None:
                self.step_origin = (now, self.step)
            # The anchor's time never moves, so no fraction of a step is lost;
            # steps not taken -- a stall, or the wait for the model -- are
            # taken off its step count instead.
            t0, s0 = self.step_origin
            # One step a tick, the clock allowed a step either way: ticks
            # jitter around control_dt, and following it exactly would mean
            # 0, 2, 0, 2 steps -- a command that leaps twice as far as a
            # joint may move. Further behind than that, it catches up.
            lag = s0 + int((now - t0) / self.control_dt) - self.step
            advance = lag - 1 if lag >= 2 else (1 if lag >= 0 else 0)
            if advance > MAX_CATCH_UP:              # a stall: drop, not sprint
                self.dropped_steps += advance - MAX_CATCH_UP
                s0 -= advance - MAX_CATCH_UP
                advance = MAX_CATCH_UP
            first = self.step + 1
            if self.executing:
                # Never past the newest chunk: when the model is slower than
                # a chunk lasts, the run waits for it instead of running dry.
                # (Reaching a chunk's end prunes it, so none left is waiting
                # too.)
                available = max(0, self.chunker.remaining(first))
                if available < advance:
                    self.waited_steps += advance - available
                    s0 -= advance - available
                    advance = available
            self.step_origin = (t0, s0)
            # While executing, a step without an action (waiting for the
            # model) settles onto the last one rather than sending nothing.
            holding = self.held_action if self.executing else None
            first = self.step + 1
            self.step += advance
            self.chunker.prune(first)
            actions = [self.chunker.action(step) for step in range(first, self.step + 1)]
            actions = [a for a in actions if a is not None]
            action = actions[-1] if actions else None
            if action is not None:
                self.prediction = (time.monotonic(), action)
                self.held_action = action
            executing = self.executing
            warmup = self.warmup
            self.step_cond.notify_all()

        if action is not None:
            self._publish_preview(action)
        if not executing:
            return

        if warmup is not None:
            command = self._warmup_command(warmup)
            if command is not None:
                self._publish_command(command)
                return

        if action is None:
            if holding is not None:
                # Waiting for the model: settle onto the last action, as
                # smoothly as the smoother allows, without counting a step.
                self._execute_step([holding], count=False)
                return
            self.get_logger().warning(
                "executing but no action for this step; the arm holds",
                throttle_duration_sec=2.0)
            return
        self._execute_step(actions)

    def _execute_step(self, actions, count=True):
        """Steps of continuous execute: smooth each, send the last, and see
        whether it is done. More than one when a tick came late; none
        counted while waiting for the model."""
        with self.step_cond:
            smoother = self.smoother
            if smoother is None or not self.executing:
                return
            for action in actions:
                command = smoother.step(action)
            if count:
                self.steps_executed += len(actions)
        self._publish_command(command)
        if count:
            self._check_finished(command, len(actions))

    def _check_finished(self, command, steps=1):
        """Stop on a timeout, or once the policy has brought the arm home.

        infer_robot.py's two conditions, counted in steps: a run has a length
        (execute_timeout), and a policy whose command stays within
        stop_delta_rad of the arm, the arm within stop_home_dist of
        finish_pose, for stop_duration, has finished its task. The arm starts
        at home too, so that only counts once it has been away.
        """
        qpos = self._current_qpos()
        rate = 1.0 / self.control_dt
        reason = None
        with self.step_cond:
            if not self.executing or self.playback is not None:
                return
            if self.execute_timeout > 0 and \
                    self.steps_executed >= self.execute_timeout * rate:
                reason = f"timeout after {self.steps_executed} steps"
            elif qpos is not None and self.stop_duration > 0:
                delta = float(np.max(np.abs(np.asarray(command) - qpos)))
                home = float(np.linalg.norm(qpos - self.finish_pose))
                if home >= self.stop_home_dist:
                    self.left_home = True
                if self.left_home and delta < self.stop_delta_rad and \
                        home < self.stop_home_dist:
                    self.still_steps += steps
                else:
                    self.still_steps = 0
                if self.still_steps >= self.stop_duration * rate:
                    reason = (f"finished: back home and still for "
                              f"{self.stop_duration:g} s, after "
                              f"{self.steps_executed} steps")
            if reason is None:
                return
            self._finish(reason)

    def _finish(self, reason):
        """End continuous execute by itself (step_cond held): inference
        stops, as infer_robot's loop does, and the arm goes home."""
        self._stop_execute(reason)
        self._end_inference()
        self.stop_reason = reason
        self.get_logger().info(reason)
        if self.home_on_finish:
            # From the last command, not the measured pose: the arm trails
            # its command while moving, and starting from where it is would
            # step the command back.
            problem = self._go_to(self.finish_pose, "going home",
                                  self.finish_duration, f"{reason}; home",
                                  origin=self.last_command)
            if problem is None:
                self.going_home = True
                self.stop_reason = f"{reason}; going home"
                return
            self.get_logger().warning(f"not going home: {problem}")
        # Nothing left to command: say so in the parameters.
        threading.Thread(target=self._reset_mode_parameter, daemon=True).start()

    def _go_to(self, pose, what, min_duration, done, origin=None):
        """Ease the arm to `pose` and hold it there (step_cond held).

        Its farthest joint averages reset_speed at most, min_duration at
        least. Any distance is allowed: it is a known pose, approached slowly.
        """
        qpos = self._held_qpos() if origin is None else origin
        gap = 0.0 if qpos is None else float(np.max(np.abs(pose - qpos)))
        duration = max(min_duration, gap / max(self.reset_speed, 1e-3))
        problem = self._begin(pose, what, max_jump=0, duration=duration,
                              origin=origin)
        if problem is None:
            self.playback = (np.asarray(pose, dtype=np.float64)[None], 0)
            self.playback_done = done
            self.returning = True
        return problem

    def _start_execute(self):
        """Begin commanding from the current prediction (step_cond held).

        Returns a reason when it cannot, None when it has started.
        """
        if not self.running:
            return self._start_playback()
        fresh_s = PREDICTION_FRESH_S
        if self.loop_mode == "sync" and self.latency_ms:
            fresh_s += 2 * self.latency_ms / 1e3    # a step per inference
        if self.prediction is None or \
                time.monotonic() - self.prediction[0] > fresh_s:
            return ("no current prediction to execute; preview first"
                    + (f" ({self.waiting})" if self.waiting else ""))
        if self.camera in ("mock", "virtual"):
            # Preview may run blind; driving the arm on a made-up image may not.
            return f"the policy is not seeing every camera ({self._blind()})"
        if self.last_chunk is not None:
            # Previewing, the arm stood still while the steps went on, so the
            # chunks' actions for now lie well into motions it never made.
            # Start as infer_robot.py does, from the first action of a fresh
            # chunk: the newest one, taken as starting now.
            self.chunker.reset()
            self.chunker.add(self.step, self.last_chunk)
            self.prediction = (time.monotonic(), self.chunker.action(self.step))
        target = np.asarray(self.prediction[1], dtype=np.float64)
        problem = self._begin(target, f"executing ({self.loop_mode})")
        if problem is None:
            qpos = self._held_qpos()
            self.smoother = TrajectorySmoother(self.vel_scale, self.control_dt)
            # infer_robot: from the first target after the ease-in, else from
            # where the arm is.
            self.smoother.reset(target if self.warmup is not None else qpos)
            self.steps_executed = 0
            self.still_steps = 0
            self.left_home = False
            self.waited_steps = 0
            self.held_action = target
        return problem

    def _start_playback(self):
        """Begin running the generated trajectory (step_cond held)."""
        if self.generate_requested or self.generating:
            return "a trajectory is being generated; wait for it"
        if self.plan is None:
            return "no trajectory to run; generate one first"
        made, actions, observed = self.plan
        age = time.monotonic() - made
        if self.plan_max_age > 0 and age > self.plan_max_age:
            return (f"the trajectory is {age:.0f} s old (plan_max_age "
                    f"{self.plan_max_age:g}); generate a new one")
        # What was approved was drawn from where the arm was then; from
        # anywhere else it is a different motion.
        qpos = self._current_qpos()
        if qpos is None:
            return "joint state is incomplete"
        drift = np.abs(qpos - observed)
        worst = int(np.argmax(drift))
        if self.plan_max_drift > 0 and drift[worst] > self.plan_max_drift:
            return (f"{JOINT_NAMES[worst]} has moved {drift[worst]:.3f} since "
                    f"the trajectory was generated (plan_max_drift "
                    f"{self.plan_max_drift:g}); generate a new one")
        reason = self._begin(actions[0], f"running the trajectory "
                                         f"({len(actions)} steps)",
                             self.plan_max_start_jump)
        if reason is None:
            self.plan = None            # run once
            self.playback = (actions, 0)
            self.playback_done = f"trajectory done: {len(actions)} steps"
        return reason

    def _on_reset(self, request, response):
        """~/reset: ease the arm to reset_pose, then hold.

        The same S-curve as the ease-in of a run, slowed down so that its
        farthest joint averages reset_speed at most. Commands go out on the
        same topics as a run, so the proxy must have act as its source.
        """
        with self.step_cond:
            if self.running:
                problem = "continuous inference is running; set running false"
            elif self.executing:
                problem = "the arm is moving; stop it first"
            elif self.generate_requested or self.generating:
                problem = "a trajectory is being generated; wait for it"
            else:
                problem = self._go_to(self.reset_pose, "going to the zero pose",
                                      self.warmup_duration, "at the zero pose")
        response.success = problem is None
        response.message = problem or "going to the zero pose"
        if problem is not None:
            self.get_logger().warning(f"not resetting: {problem}")
        self._publish_status()
        return response

    def _begin(self, target, what, max_jump=None, duration=None, origin=None):
        """Start commanding towards `target`, easing in (step_cond held).

        `max_jump` overrides max_start_jump; 0 disables the check.
        `duration` overrides warmup_duration for the ease-in. The ease starts
        where the arm is held, or at `origin` (16 values) when given.
        """
        qpos = self._held_qpos()
        if origin is not None and qpos is not None:
            qpos = np.asarray(origin, dtype=np.float64)
        if qpos is None:
            return "joint state is incomplete"
        if max_jump is None:
            max_jump = self.max_start_jump
        target = np.asarray(target, dtype=np.float64)
        gap = np.abs(target - qpos)
        worst = int(np.argmax(gap))
        if max_jump > 0 and gap[worst] > max_jump:
            return (f"the prediction puts {JOINT_NAMES[worst]} {gap[worst]:.3f} "
                    f"from where it is (max_start_jump {max_jump:g})")
        self.executing = True
        self.execute_started = time.monotonic()
        self.stop_reason = None
        if gap[worst] > self.warmup_threshold:
            duration = self.warmup_duration if duration is None else duration
            self.warmup = (time.monotonic(), qpos.astype(np.float64), target,
                           duration)
            self.get_logger().info(
                f"{what}: easing in over {duration:.1f} s, "
                f"largest gap {gap[worst]:.3f} on {JOINT_NAMES[worst]}")
        else:
            self.warmup = None
            self.get_logger().info(what)
        return None

    def _stop_execute(self, why):
        """Stop commanding (step_cond held); the arm holds its last target."""
        if self.executing:
            self.get_logger().info(f"stopped executing: {why}")
            # Set again by _check_finished when execute ended by itself; a
            # stop someone asked for needs no explaining back to them. Not
            # touched when nothing was executing: resetting the mode
            # parameter after an automatic stop must not erase its reason.
            self.stop_reason = None
        self.executing = False
        self.warmup = None
        self.execute_started = None
        self.playback = None
        self.returning = False
        self.going_home = False
        self.smoother = None

    def _publish_command(self, action):
        action = [float(value) for value in action]
        self.last_command = np.asarray(action, dtype=np.float64)
        self.hold_pose = self.last_command
        self._send(action)

    def _send(self, action):
        self.arm_pub.publish(Float64MultiArray(
            data=[action[i] for i in ARM_INDICES]))
        for index, publisher in self.gripper_pubs.items():
            publisher.publish(Float64MultiArray(data=[action[index]]))

    def _publish_idle(self):
        """A step while not executing: the pose held once while the proxy
        has act selected, else where the arm is held now; nothing while
        either is unknown.

        Never the measured pose: the proxy can pass act's next steps on
        before this node hears it was selected, and the measured pose hangs
        below the setpoint by the arm's sag, which it would then drop by."""
        if not self.idle_output or self.proxy_source is None:
            return
        if self.proxy_source == self.source_name:
            if self.hold_pose is None:
                # Selected just now: hold where the arm is held, once.
                self.hold_pose = self._fresh_held_qpos()
            pose = self.hold_pose
        else:
            self.hold_pose = None
            pose = self._fresh_held_qpos()
        if pose is not None:
            self._send([float(value) for value in pose])

    def _fresh_held_qpos(self):
        """_held_qpos, or None when the measured pose is stale."""
        if self._fresh_qpos() is None:
            return None
        return self._held_qpos()

    def _fresh_qpos(self):
        """The measured 16 positions, or None when any is stale or missing."""
        now = time.monotonic()
        with self.input_lock:
            for name in ("state", "left_gripper", "right_gripper"):
                when = self.received.get(name)
                if when is None or now - when > self.max_obs_age:
                    return None
        qpos = self._current_qpos()
        if qpos is None or not np.all(np.isfinite(qpos)):
            return None
        return np.asarray(qpos, dtype=np.float64)

    def _query_proxy_source(self):
        """Ask the proxy for its active_source until it has answered."""
        if self.proxy_source is not None or self.proxy_query is not None:
            return
        if not self.proxy_params.service_is_ready():
            return
        self.proxy_query = self.proxy_params.call_async(
            GetParameters.Request(names=["active_source"]))
        self.proxy_query.add_done_callback(self._on_proxy_source)

    def _on_proxy_source(self, future):
        self.proxy_query = None
        try:
            values = future.result().values
        except Exception:                   # noqa: BLE001 -- asked again
            return
        if values and self.proxy_source is None:
            self._set_proxy_source(values[0].string_value)

    def _on_parameter_event(self, event):
        if event.node != self.proxy_node:
            return
        for parameter in list(event.new_parameters) + list(event.changed_parameters):
            if parameter.name == "active_source":
                self._set_proxy_source(parameter.value.string_value)

    def _set_proxy_source(self, source):
        if source != self.proxy_source:
            self.get_logger().info(
                f"proxy source {self.proxy_source} -> {source}: idle output "
                + ("holds the pose" if source == self.source_name
                   else "follows where the arm is held"))
        self.proxy_source = source

    def _publish_preview(self, action):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(value) for value in action]
        self.preview_pub.publish(msg)

    def _publish_trajectory(self, stamp, actions):
        """One chunk: actions[i] is for `stamp` + i control steps.

        Starts at the observation, not at now: by the time inference is done
        its first steps are already in the past, and they are drawn anyway so
        the whole chunk the model produced can be seen.
        """
        msg = JointTrajectory()
        msg.header.stamp = stamp.to_msg()
        msg.joint_names = list(JOINT_NAMES)
        step_ns = int(round(self.control_dt * 1e9))
        for index, action in enumerate(actions):
            ns = index * step_ns
            msg.points.append(JointTrajectoryPoint(
                positions=[float(value) for value in action],
                time_from_start=Duration(sec=ns // 1_000_000_000,
                                         nanosec=ns % 1_000_000_000)))
        self.trajectory_pub.publish(msg)

    def _publish_status(self):
        with self.step_cond:
            status = self._status()
        self.status_pub.publish(String(data=json.dumps(status)))

    def _status(self):
        """The node's state for the dashboard, a JSON-able dict (step_cond held)."""
        fresh = (self.prediction is not None and
                 time.monotonic() - self.prediction[0] <= PREDICTION_FRESH_S)
        status = {
            "mode": "execute" if self.executing else "preview",
            "warming_up": self.warmup is not None,
            "running": self.running,
            "predicting": fresh,
            "waiting": self.waiting,
            "stop_reason": self.stop_reason,
            "executing_s": (time.monotonic() - self.execute_started
                            if self.execute_started else None),
            "latency_ms": self.latency_ms,
            "inferences": self.inferences,
            "generating": self.generate_requested or self.generating,
            "camera": self.camera,
            "cameras": dict(self.camera_states) or
                       {camera: None for camera in self.cameras},
            "plan": ({"steps": len(self.plan[1]),
                      "age_s": time.monotonic() - self.plan[0],
                      "source": self.plan_source}
                     if self.plan is not None else None),
            "playback": ({"step": self.playback[1],
                          "steps": len(self.playback[0])}
                         if self.playback is not None else None),
            "last_generate": self.last_generate,
            "can_reset": not self.running,
            "returning": self.returning,
            "going_home": self.going_home,
            "loop_mode": self.loop_mode,
            "vel_scale": self.vel_scale,
            "gripper_close_below": self.gripper_close_below,
            "ensemble_m": self.chunker.ensemble_k,
            "execute_timeout": self.execute_timeout,
            "infer_hz": self._chunk_rate(),
            "steps_executed": self.steps_executed,
            # Steps the run waited for the model (CPU slower than a chunk).
            "waited_steps": self.waited_steps,
            "late_ticks": self.late_ticks,
            "dropped_steps": self.dropped_steps,
            # What joint_commands carries while not executing.
            "idle_output": (None if not self.idle_output
                            or self.proxy_source is None
                            else "hold" if self.proxy_source == self.source_name
                            else "state"),
            "proxy_source": self.proxy_source,

            "policy": self.policy.describe(),
            "checkpoint": self.model_path,
            "model_loading": self.model_loading,
            "last_model_load": self.last_model_load,
            "models": self._models(),
        }
        return status

    def _chunk_rate(self):
        """Chunks a second over the last few seconds (step_cond held)."""
        times = [t for t in self.chunk_times if time.monotonic() - t < 5.0]
        if len(times) < 2:
            return None
        return (len(times) - 1) / max(times[-1] - times[0], 1e-6)

    # --------------------------------------------------------------- parameters

    def _on_set_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == "query_period" and parameter.value < 1:
                return SetParametersResult(
                    successful=False, reason="query_period must be at least 1")
            if parameter.name == "reset_pose" and len(parameter.value) != NUM_JOINTS:
                return SetParametersResult(
                    successful=False, reason=f"reset_pose needs {NUM_JOINTS} values")
            if parameter.name == "reset_speed" and parameter.value <= 0.0:
                return SetParametersResult(
                    successful=False, reason="reset_speed must be positive")
            if parameter.name == "max_obs_age" and parameter.value <= 0.0:
                return SetParametersResult(
                    successful=False, reason="max_obs_age must be positive")
            if parameter.name == "camera_fallback" and \
                    parameter.value not in CAMERA_FALLBACKS:
                return SetParametersResult(
                    successful=False,
                    reason=f"camera_fallback must be one of {CAMERA_FALLBACKS}")
            if parameter.name == "loop_mode":
                if parameter.value not in LOOP_MODES:
                    return SetParametersResult(
                        successful=False, reason=f"loop_mode must be one of {LOOP_MODES}")
                if self.executing:
                    return SetParametersResult(
                        successful=False, reason="stop executing before changing loop_mode")
            if parameter.name == "execute_timeout" and parameter.value < 0.0:
                return SetParametersResult(
                    successful=False, reason="execute_timeout must be 0 (no limit) or more")
            if parameter.name in ("gripper_close_below", "gripper_close_to") and \
                    not 0.0 <= parameter.value <= 0.043:
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be within 0..0.043 m")
            if parameter.name == "vel_scale" and not 0.1 <= parameter.value <= 2.0:
                return SetParametersResult(
                    successful=False, reason="vel_scale must be within 0.1..2")
            if parameter.name == "finish_pose" and len(parameter.value) != NUM_JOINTS:
                return SetParametersResult(
                    successful=False, reason=f"finish_pose needs {NUM_JOINTS} values")
            if parameter.name == "mode" and parameter.value not in MODES:
                return SetParametersResult(
                    successful=False, reason=f"mode must be one of {MODES}")
            if self.model_loading and (
                    (parameter.name == "mode" and parameter.value == "execute")
                    or (parameter.name == "running" and parameter.value)):
                return SetParametersResult(
                    successful=False, reason="a model is loading; wait for it")
            if parameter.name == "ensemble_weighting" and \
                    parameter.value not in ("newest", "oldest"):
                return SetParametersResult(
                    successful=False,
                    reason="ensemble_weighting must be newest or oldest")

        with self.step_cond:
            for parameter in parameters:
                if parameter.name == "running":
                    self._set_running(bool(parameter.value))
                elif parameter.name == "mode":
                    if parameter.value == "execute" and not self.executing:
                        reason = self._start_execute()
                        if reason is not None:
                            self.get_logger().warning(f"not executing: {reason}")
                            return SetParametersResult(successful=False,
                                                       reason=reason)
                    elif parameter.value == "preview":
                        self._stop_execute("back to preview")
                elif parameter.name == "query_period":
                    self.query_period = int(parameter.value)
                elif parameter.name == "temporal_ensemble":
                    self.chunker.temporal_ensemble = bool(parameter.value)
                elif parameter.name == "ensemble_k":
                    self.chunker.ensemble_k = float(parameter.value)
                elif parameter.name == "ensemble_weighting":
                    self.chunker.weighting = str(parameter.value)
                elif parameter.name == "max_obs_age":
                    self.max_obs_age = float(parameter.value)
                elif parameter.name == "camera_fallback":
                    self.camera_fallback = str(parameter.value)
                elif parameter.name == "warmup_duration":
                    self.warmup_duration = float(parameter.value)
                elif parameter.name == "reset_speed":
                    self.reset_speed = float(parameter.value)
                elif parameter.name == "reset_pose":
                    self.reset_pose = np.asarray(parameter.value, dtype=np.float64)
                elif parameter.name == "warmup_threshold":
                    self.warmup_threshold = float(parameter.value)
                elif parameter.name == "max_start_jump":
                    self.max_start_jump = float(parameter.value)
                elif parameter.name == "execute_timeout":
                    self.execute_timeout = float(parameter.value)
                elif parameter.name == "loop_mode":
                    if parameter.value != self.loop_mode:
                        self.loop_mode = str(parameter.value)
                        # Chunks filed under the other clock mean nothing now.
                        self.chunker.reset()
                        self.last_query_step = None
                        self.prediction = None
                        self.last_chunk = None
                elif parameter.name == "vel_scale":
                    self.vel_scale = float(parameter.value)
                elif parameter.name == "gripper_close_below":
                    self.gripper_close_below = float(parameter.value)
                elif parameter.name == "gripper_close_to":
                    self.gripper_close_to = float(parameter.value)
                elif parameter.name == "stop_delta_rad":
                    self.stop_delta_rad = float(parameter.value)
                elif parameter.name == "stop_home_dist":
                    self.stop_home_dist = float(parameter.value)
                elif parameter.name == "stop_duration":
                    self.stop_duration = float(parameter.value)
                elif parameter.name == "home_on_finish":
                    self.home_on_finish = bool(parameter.value)
                elif parameter.name == "idle_output":
                    self.idle_output = bool(parameter.value)
                elif parameter.name == "finish_pose":
                    self.finish_pose = np.asarray(parameter.value, dtype=np.float64)
                elif parameter.name == "finish_duration":
                    self.finish_duration = float(parameter.value)
                elif parameter.name == "plan_max_age":
                    self.plan_max_age = float(parameter.value)
                elif parameter.name == "plan_max_drift":
                    self.plan_max_drift = float(parameter.value)
                elif parameter.name == "plan_max_start_jump":
                    self.plan_max_start_jump = float(parameter.value)
                elif parameter.name == "checkpoint":
                    problem = self._begin_load(str(parameter.value))
                    if problem is not None:
                        self.get_logger().warning(f"not loading: {problem}")
                        return SetParametersResult(successful=False,
                                                   reason=problem)
            self.step_cond.notify_all()
        if any(parameter.name in ("mode", "running", "checkpoint")
               for parameter in parameters):
            # Now, so the dashboard's Run and Reset follow at once.
            self._publish_status()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ model

    def _models(self):
        """The .onnx files under checkpoint/onnx/, newest first, cached."""
        stamp, models = self.models_cache
        if time.monotonic() - stamp > 5.0:
            models = list_models()
            self.models_cache = (time.monotonic(), models)
        return models

    def _model_problem(self, path):
        """Why `path` cannot be a model, or None: a file, for these policies."""
        if not os.path.isfile(path):
            return f"no such file: {path}"
        return None

    def _extra_policy_options(self):
        """More keyword arguments for the policy class, beyond the ones every
        policy gets: a subclass declares its parameters here and returns them."""
        return {}

    def _begin_load(self, path):
        """Start loading `path` in place of the model (step_cond held).

        Only while nothing uses the model or moves the arm, so the swap can
        never land under an inference or a run. -> None, or why not.
        """
        path = os.path.expanduser(path.strip())
        if path == self.model_path and not self.model_loading:
            return None                 # also how a failed load is undone
        if self.model_loading:
            return f"already loading {os.path.basename(self.model_loading)}"
        if self.running:
            return "the policy is running; stop it first"
        if self.executing or self.playback is not None:
            return "the arm is moving; stop it first"
        if self.generate_requested or self.generating:
            return "a trajectory is being generated"
        problem = self._model_problem(path)
        if problem is not None:
            return problem
        self.model_loading = path
        threading.Thread(target=self._load_model, args=(path,),
                         daemon=True).start()
        self.get_logger().info(f"loading model {path}")
        return None

    def _check_image_size(self, policy):
        """Refuse a model trained on other frames than the camera gives.

        The policy would resize to whatever its graph fixes, so a model
        trained at another size would run on stretched frames it never saw.
        """
        expected = getattr(policy, "image_size", None)
        if expected is None:            # HoldPolicy: no images
            return
        width, height = self.image_size
        if tuple(expected) != (height, width):
            raise ValueError(
                f"the model takes {expected[1]}x{expected[0]} images, but the "
                f"camera and the recorder give {width}x{height}; export one "
                f"trained at {width}x{height}")

    def _load_model(self, path):
        """Load beside the running model, then swap (worker thread).

        The stats_file of the launch belonged to the first model; another
        carries its own, inside it or beside it.
        """
        started = time.perf_counter()
        try:
            policy = load_policy(self.policy_spec, {
                **self.policy_options, "checkpoint": path, "stats_file": ""})
            self._check_image_size(policy)
            missing = [camera for camera in self._policy_cameras(policy)
                       if camera not in self.cameras]
            if missing:
                raise ValueError(
                    f"it reads {', '.join(missing)}, which this node did not "
                    f"subscribe to at launch ({', '.join(self.cameras)}): "
                    f"restart the node to load it")
            problem = None
        except Exception as error:      # noqa: BLE001 -- the old model stays
            policy, problem = None, str(error)
        seconds = time.perf_counter() - started
        name = os.path.basename(path)
        replaced = None
        with self.step_cond:
            self.model_loading = None
            if policy is not None:
                replaced, self.policy = self.policy, policy
                self.chunker = ActionChunker(
                    policy.chunk_size, self.chunker.temporal_ensemble,
                    self.chunker.ensemble_k, self.chunker.weighting)
                # The old model's chunks and plan are not this one's.
                self._forget_chunks()
                self.model_path = path
                message = f"model loaded: {name} in {seconds:.1f} s"
            else:
                message = f"cannot load {name}: {problem}"
            serial = (self.last_model_load or {}).get("id", 0) + 1
            self.last_model_load = {"id": serial, "ok": policy is not None,
                                    "message": message, "path": path}
            current = self.model_path
        # A policy holding a connection (SmolVLAPolicy) lets it go; nothing
        # infers on the old one, since a load waits for the node to be idle.
        if replaced is not None and hasattr(replaced, "close"):
            replaced.close()
        if policy is not None:
            self.get_logger().info(f"{message}: {policy.describe()}")
        else:
            self.get_logger().error(message)
            # The parameter says what is loaded, not what was asked for.
            self.set_parameters([Parameter("checkpoint", Parameter.Type.STRING,
                                           current)])
        self._publish_status()

    def _set_running(self, running):
        """Start or stop inference (step_cond held).

        Either way the chunks go: they were planned from observations of a
        run that is over, and the arm may have been moved since. Stopping
        also stops executing.
        """
        if running == self.running:
            return
        if running:
            self.running = True
            self._forget_chunks()
        else:
            self._end_inference()
        if not running:
            self._stop_execute("inference stopped")
            if self.get_parameter("mode").value == "execute":
                # Leave the parameter saying what is happening; set from
                # another thread, since this runs inside a parameter callback.
                threading.Thread(target=self._reset_mode_parameter,
                                 daemon=True).start()
        self.get_logger().info("inference running" if running
                               else "inference stopped")

    def _forget_chunks(self):
        """Drop what an earlier run planned (step_cond held)."""
        self.run_id += 1
        self.last_query_step = None
        self.prediction = None
        self.last_chunk = None
        self.step_origin = None
        self.held_action = None
        self.chunker.reset()
        self.plan = None
        self.chunk_times = []

    def _end_inference(self):
        """Stop continuous inference (step_cond held), the parameter too."""
        if not self.running:
            return
        self.running = False
        self._forget_chunks()
        threading.Thread(target=lambda: self.set_parameters(
            [Parameter("running", Parameter.Type.BOOL, False)]), daemon=True).start()

    def _reset_mode_parameter(self):
        self.set_parameters([Parameter("mode", Parameter.Type.STRING, "preview")])

    def shutdown(self):
        with self.step_cond:
            self.alive = False
            self.step_cond.notify_all()
        self.worker.join(timeout=2.0)


def main(argv=None, node_class=ActNode, name="openarm_act"):
    rclpy.init(args=argv)
    node = None
    try:
        node = node_class()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (RuntimeError, ValueError, KeyError, ImportError, OSError) as error:
        print(f"[{name}] {error}")
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
