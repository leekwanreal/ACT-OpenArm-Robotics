#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record synchronized OpenArm RGB-D demonstrations in ACT HDF5 format.

Subscribes the robot's state and the cameras (--cameras, chest by default):
    /openarm/joint_states    the 14 arm joints
    /openarm/left_gripper    finger stroke, metres
    /openarm/right_gripper
    /openarm/camera/<name>/rgb, /openarm/camera/<name>/depth   (openarm_camera)

Each camera is kept as observations/images/<name>_rgb and <name>_depth, so
an episode of the chest camera alone is the same file as before there was
a second one; `cameras_json` lists them, `virtual_cameras_json` the ones
whose frames were a virtual camera's (black, no RealSense for it).

The three joint topics are merged back into the sixteen the dataset holds.
`action` comes from qpos unless --command-topic names a command topic.

With --command-topic, what was commanded is kept besides, as
`joint_commands` [N, 16]: the dashboard points it at the proxy's output,
/openarm/joint_commands, so a teleop episode holds the leader's targets as
well as where the arm went. A joint with no command, or one older than
COMMAND_STALE_S, is NaN there, never a stand-in. --use-state-as-action
keeps `action` on qpos all the same.

Supports two synchronization architectures:
1. 50Hz Periodic Timer with Zero-Order Hold (ZOH) Frame Buffer (Default, Recommended):
   - Samples CAN bus motor joints (50Hz) and the latest RGB-D frame at exact 20ms intervals (dt=0.02s).
   - Eliminates camera FPS bottlenecks and guarantees strict 50Hz HDF5 compliance with DATA_FORMAT_SPECIFICATION.md.
2. Approximate Time Synchronizer (Legacy / High-FPS Camera):
   - 1:1 camera-triggered callback via message_filters.ApproximateTimeSynchronizer.

With --standby it starts armed but idle: the imports (two seconds of them),
the HDF5 writer process and the subscriptions are all up, and nothing is
written until a line `start <episode> [<folder>]` arrives on stdin (the
episode then goes in <output-dir>/<folder>/) -- so the dashboard's
Start recording begins at once. It answers `@ready` when armed, then
`@recording <episode>` or `@error <reason>`; SIGINT ends the episode and
saves it, as without --standby, and stdin closing before a start leaves no
file behind.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import h5py
import numpy as np

# Reconfigure stdout for Windows unicode support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optional ROS 2 imports for cross-platform portability
try:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    import message_filters
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object
    ExternalShutdownException = Exception
    CvBridge = None
    Image = None
    JointState = None


# A command older than this is not what the arm is being told any more.
COMMAND_STALE_S = 0.2
# The cameras there can be, in the order they are recorded (openarm_camera).
CAMERAS = ("chest", "front")
# Samples sit on a fixed grid, one per period. A tick that comes late by up
# to this many periods still fills the slots it missed, with the newest
# state and frames (the zero-order hold the sampling is anyway); a longer
# stall leaves a gap, counted in the file.
MAX_CATCH_UP = 4

DEFAULT_JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]


class EpisodeWriter:
    """
    Quản lý việc ghi dữ liệu HDF5 chuẩn ACT cho 1 Episode theo DATA_FORMAT_SPECIFICATION.md.
    """
    def __init__(
        self,
        output_dir: Path,
        episode_name: str,
        batch_size: int = 10,
        frequency_hz: float = 50.0,
        img_height: int = 480,
        img_width: int = 640,
        depth_scale: float = 0.001,
        depth_min_mm: int = 200,
        depth_max_mm: int = 1200,
        joint_names: Optional[List[str]] = None,
        is_sim: bool = False,
        command_topic: str = "",
        cameras=("chest",),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.partial_path = output_dir / f"{episode_name}.partial.hdf5"
        self.final_path = output_dir / f"{episode_name}.hdf5"
        if self.partial_path.exists() or self.final_path.exists():
            raise FileExistsError(f"Episode already exists: {episode_name}")

        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.num_joints = len(self.joint_names)
        self.frequency_hz = float(frequency_hz)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.depth_scale = float(depth_scale)
        self.depth_min_mm = int(depth_min_mm)
        self.depth_max_mm = int(depth_max_mm)
        self.batch_size = int(batch_size)
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("an episode needs a camera")

        self.file = h5py.File(self.partial_path, "x", libver="latest")
        observations = self.file.create_group("observations")
        images = observations.create_group("images")

        self.datasets = {}
        for camera in self.cameras:
            self.datasets[f"{camera}_rgb"] = images.create_dataset(
                f"{camera}_rgb",
                shape=(0, self.img_height, self.img_width, 3),
                maxshape=(None, self.img_height, self.img_width, 3),
                chunks=(1, self.img_height, self.img_width, 3),
                dtype=np.uint8,
            )
            self.datasets[f"{camera}_depth"] = images.create_dataset(
                f"{camera}_depth",
                shape=(0, self.img_height, self.img_width),
                maxshape=(None, self.img_height, self.img_width),
                chunks=(1, self.img_height, self.img_width),
                dtype=np.uint16,
            )
        self.datasets.update({
            "qpos": observations.create_dataset(
                "qpos",
                shape=(0, self.num_joints),
                maxshape=(None, self.num_joints),
                chunks=(self.batch_size, self.num_joints),
                dtype=np.float32,
            ),
            "qvel": observations.create_dataset(
                "qvel",
                shape=(0, self.num_joints),
                maxshape=(None, self.num_joints),
                chunks=(self.batch_size, self.num_joints),
                dtype=np.float32,
            ),
            "effort": observations.create_dataset(
                "effort",
                shape=(0, self.num_joints),
                maxshape=(None, self.num_joints),
                chunks=(self.batch_size, self.num_joints),
                dtype=np.float32,
            ),
            "action": self.file.create_dataset(
                "action",
                shape=(0, self.num_joints),
                maxshape=(None, self.num_joints),
                chunks=(self.batch_size, self.num_joints),
                dtype=np.float32,
            ),
            # What the source commanded (teleop's targets through the
            # proxy), NaN where nothing was; `action` stays the same.
            "joint_commands": self.file.create_dataset(
                "joint_commands",
                shape=(0, self.num_joints),
                maxshape=(None, self.num_joints),
                chunks=(self.batch_size, self.num_joints),
                dtype=np.float32,
            ),
            "timestamp_ns": self.file.create_dataset(
                "timestamp_ns",
                shape=(0,),
                maxshape=(None,),
                chunks=(self.batch_size,),
                dtype=np.int64,
            ),
        })

        # Metadata attributes chuẩn hóa theo DATA_FORMAT_SPECIFICATION.md
        self.file.attrs["sim"] = bool(is_sim)
        self.file.attrs["frequency_hz"] = self.frequency_hz
        self.file.attrs["control_dt"] = float(1.0 / self.frequency_hz)
        self.file.attrs["robot_type"] = "OpenArm_Bimanual_16DOF"
        self.file.attrs["num_joints"] = self.num_joints
        self.file.attrs["depth_scale"] = self.depth_scale
        self.file.attrs["depth_range_m"] = np.asarray(
            [self.depth_min_mm / 1000.0, self.depth_max_mm / 1000.0],
            dtype=np.float32,
        )
        self.file.attrs["action_representation"] = "absolute_joint_position"
        # Arm joints: rad, official OpenArm URDF convention. Grippers: finger stroke in m.
        self.file.attrs["joint_convention"] = "openarm_urdf"
        self.file.attrs["gripper_unit"] = "stroke_m"
        # Where joint_commands came from; empty when nothing was recorded.
        self.file.attrs["joint_commands_topic"] = str(command_topic or "")
        self.file.attrs["joint_names_json"] = json.dumps(self.joint_names)
        # Which cameras the episode holds, and which of them were virtual
        # in any sample (black frames, standing in for a camera that was not
        # plugged in), written as the episode closes.
        self.file.attrs["cameras_json"] = json.dumps(list(self.cameras))
        self.virtual = set()
        self.file.attrs["virtual_cameras_json"] = json.dumps([])
        self.file.attrs["complete"] = False
        self.file.attrs["created_at"] = datetime.now().astimezone().isoformat()

        self.sample_count = 0
        self.first_timestamp_ns = None
        self.last_timestamp_ns = None
        # Samples written for a slot their tick reached late, and jumps of
        # more than one slot between samples: 0 and 0 is a clean 50 Hz.
        self.late_samples = 0
        self.gaps = 0

    # The images are written as they come, uncompressed: compressing them
    # costs about 36 ms a sample, more than a 50 Hz step, so it is done once
    # the episode is over (episode_compress.py). Datasets grow GROW samples at
    # a time, not one, and the file is flushed about once a second.
    GROW = 250
    FLUSH_EVERY = 50

    def append(self, images, qpos, qvel, effort, action, timestamp_ns,
               command=None, virtual=(), late=False):
        """`images`: {camera: (rgb, depth)} for every camera of the episode;
        `virtual`: those of them whose frames are a virtual camera's; `late`:
        the sample fills a slot its tick reached late."""
        self.virtual.update(virtual)
        self.late_samples += bool(late)
        if self.last_timestamp_ns is not None and \
                timestamp_ns - self.last_timestamp_ns > 1.5e9 / self.frequency_hz:
            self.gaps += 1
        index = self.sample_count
        if index >= self.datasets["timestamp_ns"].shape[0]:
            for dataset in self.datasets.values():
                dataset.resize(index + self.GROW, axis=0)
        if command is None:
            command = np.full(self.num_joints, np.nan, dtype=np.float32)
        row = {"qpos": qpos, "qvel": qvel, "effort": effort, "action": action,
               "joint_commands": command, "timestamp_ns": timestamp_ns}
        for camera in self.cameras:
            row[f"{camera}_rgb"], row[f"{camera}_depth"] = images[camera]
        for name, dataset in self.datasets.items():
            dataset[index] = np.asarray(row[name], dtype=dataset.dtype)
        self.sample_count += 1
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        if self.sample_count % self.FLUSH_EVERY == 0:
            self.file.flush()

    def finalize(self):
        """Close the uncompressed episode; -> its path, or None if empty.

        It stays `<episode>.partial.hdf5` with `complete` false: the episode
        is complete once episode_compress.py has written `<episode>.hdf5`.
        """
        for dataset in self.datasets.values():
            dataset.resize(self.sample_count, axis=0)
        self.file.attrs["virtual_cameras_json"] = json.dumps(
            [camera for camera in self.cameras if camera in self.virtual])
        self.file.attrs["late_samples"] = self.late_samples
        self.file.attrs["gaps"] = self.gaps
        if self.sample_count:
            duration_s = (self.last_timestamp_ns - self.first_timestamp_ns) / 1e9
            self.file.attrs["samples"] = self.sample_count
            self.file.attrs["duration_seconds"] = duration_s
        self.file.attrs["complete"] = False
        self.file.flush()
        self.file.close()
        return self.partial_path if self.sample_count else None


def _write_episode(kwargs, samples, result):
    """Child process: an EpisodeWriter fed from `samples` until None.

    With kwargs None it waits for them as the first item: forked before
    the episode is named, it opens the file only when recording starts.
    """
    import signal
    # Ctrl-C reaches the whole process group; the parent decides when the
    # episode ends and says so with None, after the last sample.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if kwargs is None:
        kwargs = samples.get()
        if kwargs is None:              # closed without ever recording
            return
    try:
        writer = EpisodeWriter(**kwargs)
    except Exception as error:      # reported to the parent, which raises it
        result.send(("error", repr(error)))
        return
    result.send(("ready", None))
    # A frame is sent once (BackgroundEpisodeWriter.append): None in its
    # place means the camera's previous one again.
    last = {}
    while True:
        sample = samples.get()
        if sample is None:
            break
        images = {}
        for camera, frames in sample[0].items():
            images[camera] = tuple(
                last[camera][k] if frame is None else frame
                for k, frame in enumerate(frames))
        last.update(images)
        writer.append(images, *sample[1:])
    path = writer.finalize()
    result.send(("done", str(path) if path else None))


class BackgroundEpisodeWriter:
    """An EpisodeWriter in its own process, behind a queue.

    The 50 Hz timer only hands each sample over; HDF5 and the disk -- whose
    writes stall now and then for tens of milliseconds, holding the GIL --
    never delay it. Started before rclpy, so the fork copies no ROS threads.
    """

    # Samples that may wait for the disk: 5 s, about 400 MB uncompressed.
    BACKLOG = 250

    def __init__(self, deferred=False, **kwargs):
        """`deferred`: fork the writer now, open the episode with open()
        later -- kwargs then leave out output_dir and episode_name."""
        import multiprocessing

        self.kwargs = dict(kwargs)
        self.opened = False
        self.partial_path = self.final_path = None
        self.joint_names = kwargs.get("joint_names") or DEFAULT_JOINT_NAMES
        self.frequency_hz = float(kwargs.get("frequency_hz", 50.0))
        self.img_height = int(kwargs.get("img_height", 480))
        self.img_width = int(kwargs.get("img_width", 640))
        self.depth_scale = float(kwargs.get("depth_scale", 0.001))
        self.depth_min_mm = int(kwargs.get("depth_min_mm", 200))
        self.depth_max_mm = int(kwargs.get("depth_max_mm", 1200))
        self.sample_count = 0
        self.cameras = tuple(kwargs.get("cameras", ("chest",)))
        self.sent = {}              # camera -> (rgb, depth) last put on the queue

        context = multiprocessing.get_context("fork")
        self.samples = context.Queue(maxsize=self.BACKLOG)
        self.result, child_end = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_write_episode,
            args=(None if deferred else kwargs, self.samples, child_end),
            name="episode-writer", daemon=False)
        self.process.start()
        if not deferred:
            self._opened(kwargs)

    def open(self, output_dir, episode_name):
        """Open the episode of a deferred writer; raises when it cannot."""
        kwargs = dict(self.kwargs, output_dir=Path(output_dir),
                      episode_name=episode_name)
        self.samples.put(kwargs)
        self._opened(kwargs)

    def _opened(self, kwargs):
        status, detail = self.result.recv()
        if status == "error":
            self.process.join()
            raise FileExistsError(detail) if "FileExists" in detail \
                else RuntimeError(detail)
        output_dir = Path(kwargs["output_dir"])
        episode = kwargs["episode_name"]
        self.partial_path = output_dir / f"{episode}.partial.hdf5"
        self.final_path = output_dir / f"{episode}.hdf5"
        self.opened = True

    def close(self):
        """End a deferred writer that never opened an episode."""
        if not self.opened and self.process.is_alive():
            self.samples.put(None)
            self.process.join(timeout=5)

    def append(self, images, qpos, qvel, effort, action, timestamp_ns,
               command=None, virtual=(), late=False):
        """As EpisodeWriter.append. A camera runs at 25 Hz and the samples
        at 50, so most frames come twice: the very same array again is sent
        as None, and the writer repeats the one it has -- half the bytes
        through the queue, which two cameras would otherwise double."""
        sent = {}
        for camera, frames in images.items():
            before = self.sent.get(camera, (None, None))
            sent[camera] = tuple(None if frame is before[k] else frame
                                 for k, frame in enumerate(frames))
            self.sent[camera] = frames
        self.samples.put((sent, qpos, qvel, effort, action, timestamp_ns,
                          command, tuple(virtual), bool(late)))
        self.sample_count += 1

    def finalize(self):
        """Wait for every sample to reach the file; -> its path or None."""
        if not self.opened:
            self.close()
            return None
        self.samples.put(None)
        status, path = self.result.recv()
        self.process.join()
        return Path(path) if path else None

    def compress_in_background(self):
        """Start episode_compress on the finished file, outliving this process
        and the dashboard's wait for it."""
        import subprocess
        return subprocess.Popen(
            [sys.executable, "-m", "openarm_ui.episode_compress",
             str(self.partial_path), str(self.final_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)


class ACTDataRecorder(Node):
    """
    ROS 2 Node thu thập dữ liệu OpenArm đồng bộ.
    Hỗ trợ 2 chế độ:
    - periodic_zoh (mặc định): Timer cố định 50Hz, đọc góc CAN bus tức thì và frame ảnh mới nhất từ bộ đệm (ZOH).
    - approximate: ApproximateTimeSynchronizer truyền thống chờ khung hình camera.
    """
    def __init__(self, args, writer: EpisodeWriter, recording=True):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 (rclpy) chưa được cài đặt trong môi trường này.")

        super().__init__("openarm_act_data_recorder")
        self.args = args
        self.writer = writer
        self.bridge = CvBridge()
        self.rejected_samples = 0
        self.last_timestamp_ns = -1
        self.next_slot_ns = None        # the grid's next sample time
        self.lock = threading.Lock()
        # False while armed in standby: the streams are followed, nothing
        # is written until begin().
        self.recording = recording
        # Set to end the episode (its length reached, or a standby never
        # started): main() stops spinning and saves. Shutting rclpy down from
        # inside the timer instead left the spin and the writer waiting on
        # each other, the episode never saved.
        self.stop = threading.Event()

        # Bộ đệm tin nhắn cảm biến: the newest frame of each camera,
        # "<camera>/<kind>" -> Image.
        self.cameras = tuple(args.cameras)
        self.latest_frames = {}
        self.latest_state_msg = None
        self.latest_arm_msg = None
        self.gripper_state = {}
        self.latest_command_msg = None
        self.warmup_logged = False

        # Cache giải mã ảnh nhằm tối ưu hoá CPU trong ZOH 50Hz: per
        # "<camera>/<kind>", (the message, its decoded array). The same array
        # again is what lets the writer send a frame only once.
        self._decoded = {}
        # Recorder depth is uint16 mm, 0 kept for no reading and the rest
        # clipped to the working range: as a lookup table, a frame is one
        # np.take (~2 ms at 640x480) instead of a float round trip (~9 ms),
        # which two cameras at 25 Hz could not afford beside the 50 Hz timer.
        self._depth_table = np.clip(np.arange(65536), writer.depth_min_mm,
                                    writer.depth_max_mm).astype(np.uint16)
        self._depth_table[0] = 0

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        joint_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # Commands are recorded whenever a topic is named; they become the
        # action, and sampling waits for the first, only without
        # --use-state-as-action.
        self.record_command = (
            bool(self.args.command_topic)
            and self.args.command_topic.strip().lower() not in ("none", "false")
        )
        self.has_command = (self.record_command
                            and not getattr(self.args, "use_state_as_action", False))
        self.command_time = None        # monotonic arrival of the last one

        prefix = args.camera_prefix.rstrip("/")
        topics = {f"{camera}/{kind}": f"{prefix}/{camera}/{kind}"
                  for camera in self.cameras for kind in ("rgb", "depth")}

        if self.args.sync_mode == "periodic_zoh":
            # 1. Đăng ký nhận tin tức thời từ các cảm biến độc lập
            for key, topic in topics.items():
                self.create_subscription(
                    Image, topic, lambda msg, key=key: self._on_frame(key, msg),
                    camera_qos)
            self.create_subscription(JointState, args.state_topic, self._on_state, joint_qos)
            # The arm joints and the grippers arrive on separate topics; the
            # dataset wants all sixteen together, so they are merged here.
            self.create_subscription(
                Float64MultiArray, args.left_gripper_topic,
                lambda msg: self._on_gripper(msg, "left_gripper"), joint_qos)
            self.create_subscription(
                Float64MultiArray, args.right_gripper_topic,
                lambda msg: self._on_gripper(msg, "right_gripper"), joint_qos)
            if self.record_command:
                self.create_subscription(JointState, args.command_topic, self._on_command, joint_qos)

            # 2. Timer chu kỳ lấy mẫu cố định (50.0Hz -> 20.0ms)
            timer_period = 1.0 / self.args.frequency_hz
            self.timer = self.create_timer(timer_period, self._on_periodic_timer)
            self.get_logger().info(
                f"[*] Khởi động ACT Data Recorder chế độ Periodic ZOH tại {self.args.frequency_hz} Hz (chu kỳ {timer_period*1000:.1f}ms), "
                f"cameras {', '.join(self.cameras)}. Nhấn Ctrl+C để kết thúc episode."
            )
        else:
            # Chế độ Approximate Time Synchronizer truyền thống
            # Every camera's rgb and depth, in topics order, then the state.
            subscribers = [
                message_filters.Subscriber(self, Image, topic, qos_profile=camera_qos)
                for topic in topics.values()
            ] + [
                message_filters.Subscriber(self, JointState, args.state_topic, qos_profile=joint_qos),
            ]
            if self.has_command:
                subscribers.append(
                    message_filters.Subscriber(self, JointState, args.command_topic, qos_profile=joint_qos)
                )

            self.sync = message_filters.ApproximateTimeSynchronizer(
                subscribers,
                queue_size=25,
                slop=args.sync_tolerance_ms / 1000.0,
            )
            self.sync.registerCallback(self._on_approx_sample)

            self.get_logger().info(
                f"[*] Khởi động ACT Data Recorder chế độ Approximate Synchronizer. Nhấn Ctrl+C để kết thúc episode."
            )

    def begin(self, output_dir, episode):
        """Start writing `episode` (standby); -> None, or why it cannot."""
        if self.recording:
            return "already recording"
        if not episode or not episode.replace("-", "").replace("_", "").isalnum():
            return "episode name may contain only letters, numbers, '-' and '_'"
        try:
            self.writer.open(output_dir, episode)
        except Exception as error:      # noqa: BLE001 -- reported to the parent
            return str(error)
        with self.lock:
            self.recording = True
        self.get_logger().info(f"recording {episode}")
        return None

    def _on_frame(self, key, msg):
        with self.lock:
            self.latest_frames[key] = msg
            recording = self.recording
        # Decoded as it arrives while recording, so the work is spread over
        # the frames rather than piled on the timer, which then only gathers
        # what is ready (and decodes what is not yet).
        if recording:
            self._decode(key, msg)

    def _on_state(self, msg):
        with self.lock:
            self.latest_arm_msg = msg
            self.latest_state_msg = self._merge_grippers(msg)

    def _on_gripper(self, msg, name):
        """Gripper state: data is [stroke in metres, stroke rate, effort]."""
        values = list(msg.data) + [0.0, 0.0, 0.0]
        with self.lock:
            self.gripper_state[name] = (float(values[0]), float(values[1]),
                                        float(values[2]))
            if self.latest_arm_msg is not None:
                self.latest_state_msg = self._merge_grippers(self.latest_arm_msg)

    def _merge_grippers(self, arm_msg):
        """One JointState carrying the arm joints and both grippers.

        The dataset contract is sixteen joints in a fixed order, so the split
        topics are put back together rather than changing the file format.
        """
        merged = JointState()
        merged.header = arm_msg.header
        merged.name = list(arm_msg.name)
        merged.position = list(arm_msg.position)
        merged.velocity = list(arm_msg.velocity)
        merged.effort = list(arm_msg.effort)
        for name, (position, velocity, effort) in self.gripper_state.items():
            merged.name.append(name)
            merged.position.append(position)
            merged.velocity.append(velocity)
            merged.effort.append(effort)
        return merged

    def _on_command(self, msg):
        with self.lock:
            self.latest_command_msg = msg
            self.command_time = time.monotonic()

    def _command_values(self, message) -> np.ndarray:
        """The command in the dataset's joint order; NaN for a joint it does
        not name (a source without grippers) or when it is stale."""
        values = np.full(len(self.writer.joint_names), np.nan, dtype=np.float32)
        if message is None or self.command_time is None or \
                time.monotonic() - self.command_time > COMMAND_STALE_S:
            return values
        by_name = dict(zip(message.name, message.position))
        for index, name in enumerate(self.writer.joint_names):
            value = by_name.get(name)
            if value is not None and np.isfinite(value):
                values[index] = value
        return values

    def _ordered_values(self, message: JointState, field: str) -> np.ndarray:
        raw_values = getattr(message, field, None)
        expected_joints = self.writer.joint_names
        expected_count = len(expected_joints)

        # Hỗ trợ trường hợp robot driver không gửi velocity hoặc effort
        if (raw_values is None or len(raw_values) == 0) and field in ("velocity", "effort"):
            return np.zeros(expected_count, dtype=np.float32)

        if len(raw_values) != expected_count:
            raise ValueError(
                f"expected {expected_count} {field} values, received {len(raw_values)}"
            )

        if not message.name:
            values = np.asarray(raw_values, dtype=np.float32)
        else:
            if len(message.name) != len(raw_values):
                raise ValueError(f"joint name and {field} lengths differ")
            by_name = dict(zip(message.name, raw_values))
            missing = [name for name in expected_joints if name not in by_name]
            if missing:
                raise ValueError(f"missing joints: {missing}")
            values = np.asarray([by_name[name] for name in expected_joints], dtype=np.float32)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"joint {field} values contain NaN or infinity")
        return values

    def _reject(self, reason: str):
        self.rejected_samples += 1
        self.get_logger().warning(reason, throttle_duration_sec=5.0)

    def _decode(self, key: str, msg: Image) -> np.ndarray:
        """A frame of "<camera>/<kind>", decoded once per message."""
        cached = self._decoded.get(key)
        if cached is not None and cached[0] is msg:
            return cached[1]
        array = (self._decode_rgb(msg) if key.endswith("/rgb")
                 else self._decode_depth(msg))
        self._decoded[key] = (msg, array)
        return array

    def _decode_rgb(self, rgb_msg: Image) -> np.ndarray:
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB image must have three channels, got {rgb.shape}")

        if rgb.shape[:2] != (self.writer.img_height, self.writer.img_width):
            rgb = cv2.resize(
                rgb,
                (self.writer.img_width, self.writer.img_height),
                interpolation=cv2.INTER_AREA,
            )

        return np.asarray(rgb, dtype=np.uint8)

    def _decode_depth(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.ndim != 2:
            raise ValueError(f"depth image must be single-channel, got {depth.shape}")

        if depth.shape != (self.writer.img_height, self.writer.img_width):
            depth = cv2.resize(
                depth,
                (self.writer.img_width, self.writer.img_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if depth.dtype == np.uint16:
            # The usual case (openarm_camera publishes 16UC1 mm).
            return np.take(self._depth_table, depth)

        # Chuẩn hoá đơn vị depth về milimét (uint16)
        if depth.dtype == np.float32:
            scale = self.writer.depth_scale if self.writer.depth_scale > 0 else 0.001
            depth_mm = np.nan_to_num(depth, nan=0.0) / scale
        else:
            depth_mm = depth.astype(np.float32)

        # Giữ 0 cho pixel mất tín hiệu / rỗng, kẹp trong dải làm việc quy định
        return np.where(
            depth_mm == 0,
            0,
            np.clip(depth_mm, self.writer.depth_min_mm, self.writer.depth_max_mm),
        ).astype(np.uint16, copy=False)

    def _process_and_record(
        self,
        frames: Dict[str, Image],
        state_msg: JointState,
        command_msg: Optional[JointState],
        timestamp_ns: int,
        late: bool = False,
    ):
        """`frames`: "<camera>/<kind>" -> Image, for every camera."""
        try:
            if timestamp_ns <= self.last_timestamp_ns:
                self._reject("Rejected duplicate or non-monotonic timestamp")
                return

            images = {camera: (self._decode(f"{camera}/rgb", frames[f"{camera}/rgb"]),
                               self._decode(f"{camera}/depth", frames[f"{camera}/depth"]))
                      for camera in self.cameras}
            # openarm_camera stamps a virtual (black) camera's frames so.
            virtual = [camera for camera in self.cameras
                       if frames[f"{camera}/rgb"].header.frame_id.endswith("_virtual")]

            qpos = self._ordered_values(state_msg, "position")
            qvel = self._ordered_values(state_msg, "velocity")
            effort = self._ordered_values(state_msg, "effort")

            if command_msg is not None and not getattr(self.args, "use_state_as_action", False):
                action = self._ordered_values(command_msg, "position")
            else:
                # Nếu không có command topic, sử dụng chính góc đo thực tế làm action
                action = qpos.copy()
            command = (self._command_values(command_msg)
                       if self.record_command else None)

            self.writer.append(
                images,
                qpos,
                qvel,
                effort,
                action,
                timestamp_ns,
                command,
                virtual,
                late,
            )
            self.last_timestamp_ns = timestamp_ns

            if self.writer.sample_count % int(self.writer.frequency_hz * 5) == 0:
                seconds = self.writer.sample_count / self.writer.frequency_hz
                self.get_logger().info(
                    f"Recorded {self.writer.sample_count} samples ({seconds:.1f} s @ {self.writer.frequency_hz:.1f}Hz)"
                )

            if self.args.max_duration > 0:
                elapsed = self.writer.sample_count / self.writer.frequency_hz
                if elapsed >= self.args.max_duration:
                    self.get_logger().info("Maximum duration reached; stopping episode")
                    self.stop.set()
        except Exception as error:
            self._reject(f"Rejected sample: {error}")

    def _on_periodic_timer(self):
        """Callback định kỳ kích hoạt chính xác theo chu kỳ frequency_hz (50Hz = 20ms)."""
        with self.lock:
            if not self.recording:
                return
            frames = dict(self.latest_frames)
            state_msg = self.latest_state_msg
            command_msg = self.latest_command_msg

        # Kiểm tra điều kiện khởi động luồng cảm biến (Warm-up check)
        need_command = self.has_command
        missing = [f"{camera} {kind}" for camera in self.cameras
                   for kind in ("rgb", "depth") if f"{camera}/{kind}" not in frames]
        if missing or state_msg is None or (need_command and command_msg is None):
            waiting = list(missing)
            if state_msg is None: waiting.append("Joint States")
            if need_command and command_msg is None: waiting.append("Joint Commands")
            self.get_logger().info(
                f"[*] Đang chờ luồng cảm biến xuất hiện: {', '.join(waiting)}...",
                throttle_duration_sec=2.0,
            )
            return

        if not self.warmup_logged:
            self.warmup_logged = True
            self.get_logger().info(f"[*] Tất cả luồng cảm biến đã sẵn sàng. Bắt đầu ghi dữ liệu tại {self.writer.frequency_hz} Hz...")

        # One sample per slot of a fixed grid: a tick a little early still
        # takes the slot it is nearest; one that comes late fills the slots
        # it missed, up to MAX_CATCH_UP, so a busy moment costs no samples.
        now_ns = self.get_clock().now().nanoseconds
        period_ns = round(1e9 / self.writer.frequency_hz)
        if self.next_slot_ns is None:
            self.next_slot_ns = now_ns
        overdue = (now_ns - self.next_slot_ns) // period_ns
        if overdue > MAX_CATCH_UP:
            self.get_logger().warning(
                f"the recorder stalled {overdue * period_ns / 1e6:.0f} ms: "
                f"{overdue - MAX_CATCH_UP} samples lost", throttle_duration_sec=2.0)
            self.next_slot_ns += (overdue - MAX_CATCH_UP) * period_ns
        first = True
        while self.next_slot_ns <= now_ns + period_ns // 2:
            self._process_and_record(frames, state_msg, command_msg,
                                     self.next_slot_ns, late=not first)
            self.next_slot_ns += period_ns
            first = False

    def _on_approx_sample(self, *msgs):
        """Callback đồng bộ approximate: every camera's rgb and depth, the
        state, then the command when there is a command topic."""
        if not self.recording:
            return
        keys = [f"{camera}/{kind}" for camera in self.cameras for kind in ("rgb", "depth")]
        frames = dict(zip(keys, msgs))
        state_msg = msgs[len(keys)]
        command_msg = msgs[len(keys) + 1] if len(msgs) > len(keys) + 1 else None
        first = msgs[0].header.stamp
        timestamp_ns = first.sec * 1_000_000_000 + first.nanosec
        self._process_and_record(frames, state_msg, command_msg, timestamp_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dataset", help="Thư mục lưu trữ HDF5")
    parser.add_argument("--episode", default=None, help="Tên episode (tùy chọn, mặc định sinh theo thời gian)")
    parser.add_argument("--frequency-hz", type=float, default=50.0, help="Tần số lấy mẫu ghi dữ liệu cố định (mặc định: 50.0 Hz)")
    parser.add_argument(
        "--sync-mode",
        type=str,
        default="periodic_zoh",
        choices=["periodic_zoh", "approximate"],
        help="Cơ chế đồng bộ: 'periodic_zoh' (khuyên dùng - 50Hz chuẩn ZOH) hoặc 'approximate' (chờ camera)",
    )
    parser.add_argument("--cameras", default="chest",
                        help="the cameras to record, chest,front for both (openarm_camera's roles)")
    parser.add_argument("--camera-prefix", default="/openarm/camera",
                        help="each camera is <prefix>/<name>/{rgb,depth}")
    parser.add_argument("--state-topic", default="/openarm/joint_states", help="ROS 2 topic trạng thái khớp tay")
    parser.add_argument("--left-gripper-topic", default="/openarm/left_gripper", help="ROS 2 topic trạng thái gripper trái")
    parser.add_argument("--right-gripper-topic", default="/openarm/right_gripper", help="ROS 2 topic trạng thái gripper phải")
    parser.add_argument(
        "--command-topic",
        default="none",
        help="ROS 2 topic lệnh góc điều khiển. Mặc định 'none': chỉ ghi từ "
             "joint states và camera, action lấy từ qpos. Trỏ vào "
             "/openarm/joint_commands nếu muốn action là lệnh đã gửi.",
    )
    parser.add_argument(
        "--use-state-as-action",
        action="store_true",
        help="Buộc dùng qpos làm action kể cả khi có command topic",
    )
    parser.add_argument("--sync-tolerance-ms", type=float, default=12.0, help="Dung sai đồng bộ cho mode approximate (ms)")
    parser.add_argument("--img-height", type=int, default=480, help="Chiều cao ảnh RGB-D (mặc định: 480)")
    parser.add_argument("--img-width", type=int, default=640, help="Chiều rộng ảnh RGB-D (mặc định: 640)")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Hệ số quy đổi depth (mặc định: 0.001)")
    parser.add_argument("--depth-min-mm", type=int, default=200, help="Khoảng cách depth tối thiểu hợp lệ mm (mặc định: 200)")
    parser.add_argument("--depth-max-mm", type=int, default=1200, help="Khoảng cách depth tối đa hợp lệ mm (mặc định: 1200)")
    parser.add_argument("--num-joints", type=int, default=16, help="Số lượng khớp robot (mặc định: 16)")
    parser.add_argument("--batch-size", type=int, default=10, help="HDF5 write batch size")
    parser.add_argument("--max-duration", type=float, default=25.0, help="Thời lượng ghi tối đa của 1 episode (giây, mặc định: 25.0s)")
    parser.add_argument("--sim", action="store_true", help="Đánh dấu episode là dữ liệu mô phỏng Simulation")
    parser.add_argument("--standby", action="store_true",
                        help="start armed and idle; record on `start <episode>` from stdin")

    args = parser.parse_args()

    if args.frequency_hz <= 0:
        parser.error("Tần số sampling frequency-hz phải > 0")
    if args.sync_tolerance_ms <= 0 or args.batch_size <= 0 or args.max_duration < 0:
        parser.error("invalid tolerance, batch size, or maximum duration")
    names = args.cameras.replace(",", " ").split()
    if not names or any(name not in CAMERAS for name in names):
        parser.error(f"--cameras: one or more of {', '.join(CAMERAS)}, e.g. chest,front")
    # In role order, chest first, whichever way they were typed.
    args.cameras = tuple(name for name in CAMERAS if name in names)

    if not HAS_ROS2:
        print("[X] Lỗi: Cần môi trường ROS 2 (rclpy) để khởi động node thu thập dữ liệu trên Robot/IPC.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser()
    episode_name = args.episode or datetime.now().strftime("episode_%Y%m%d_%H%M%S")
    if not episode_name.replace("-", "").replace("_", "").isalnum():
        parser.error("episode name may contain only letters, numbers, '-' and '_'")

    settings = dict(
        batch_size=args.batch_size,
        frequency_hz=args.frequency_hz,
        img_height=args.img_height,
        img_width=args.img_width,
        depth_scale=args.depth_scale,
        depth_min_mm=args.depth_min_mm,
        depth_max_mm=args.depth_max_mm,
        is_sim=args.sim,
        cameras=args.cameras,
        command_topic=(args.command_topic
                       if args.command_topic.strip().lower() not in ("none", "false")
                       else ""),
    )
    # Forked before rclpy, standby or not, so the writer has no ROS threads.
    if args.standby:
        writer = BackgroundEpisodeWriter(deferred=True, **settings)
    else:
        writer = BackgroundEpisodeWriter(
            output_dir=output_dir, episode_name=episode_name, **settings)

    rclpy.init()
    node = ACTDataRecorder(args, writer, recording=not args.standby)
    if args.standby:
        def control():
            """`start <episode> [<folder>]` from the dashboard -- the folder,
            the day's, under --output-dir; stdin closing first means it was
            never wanted."""
            for line in sys.stdin:
                words = line.split()
                if not words or words[0] != "start" or len(words) not in (2, 3):
                    continue
                episode = words[1]
                folder = words[2] if len(words) == 3 else ""
                if folder and not folder.replace("-", "").replace("_", "").isalnum():
                    print(f"@error bad folder name {folder!r}", flush=True)
                    continue
                problem = node.begin(output_dir / folder if folder else output_dir,
                                     episode)
                print(f"@error {problem}" if problem else f"@recording {episode}",
                      flush=True)
                if problem is None:
                    return
            if not node.recording:
                node.stop.set()
        threading.Thread(target=control, daemon=True).start()
        print("@ready", flush=True)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # Until the episode ends, or SIGINT shuts rclpy down.
        while rclpy.ok() and not node.stop.is_set():
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        partial_path = writer.finalize()
        if not writer.opened:
            print("[Recorder] standby closed without recording", flush=True)
        elif partial_path:
            writer.compress_in_background()
            print(
                f"[Recorder] Saved {writer.sample_count} samples to {partial_path} "
                f"({node.rejected_samples} rejected, Frequency: {args.frequency_hz}Hz); "
                f"compressing to {writer.final_path} in the background",
                flush=True,
            )
        else:
            print(
                f"[Recorder] No valid samples; incomplete file kept at {writer.partial_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
