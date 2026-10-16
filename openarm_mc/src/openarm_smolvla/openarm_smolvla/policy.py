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

"""SmolVLA as an openarm_act policy, asked of an openarm_smolvla server.

SmolVLA needs Python 3.12 and LeRobot, the robot computer has ROS Humble's
3.10 and maybe no GPU: the model runs in openarm_smolvla's
`scripts/serve.py`, and this asks it for each chunk over websocket.

    predict(rgb uint8 [H, W, 3], depth uint16 [H, W] mm, qpos float32 [16])
        -> float32 [chunk, 16]  absolute targets, left j1..j7, left gripper,
                                right j1..j7, right gripper; rad, grippers in m;
                                actions[0] for the step the observation was taken at

which is openarm_act's contract (openarm_act/policy.py). The frames go raw:
depth encoding, resizing, normalising and delta-to-absolute happen on the
server, in the code that trained the model.

No prompt goes with them. SmolVLA reads a task, but the model is trained on
one: every episode gets the same task from openarm_smolvla's
manifests/episodes.yaml, so the server fills in that one itself (its
default_prompt) and the model sees exactly what it was trained on. Another
instruction would only be text it never saw. A server whose model has
several tasks, and so no default, is refused.

The protocol is openpi's, as openarm_smolvla_client speaks it: on
connecting, the server sends its metadata (chunk size, frame size, cameras,
default prompt: the training task); then each observation out, msgpack
with NumPy arrays, is answered by {"actions", "server_timing"}, or by a
traceback as text on failure, after which the server closes the
connection. This reconnects on the next call.
"""

import contextlib
import os
import socket
import time
from urllib.parse import urlparse

import numpy as np
from websockets.exceptions import ConnectionClosed

from openarm_act.policy import NUM_JOINTS, Policy

from . import msgpack_numpy

JOINT_NAMES = (
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
)
# OpenArm V1 limits in JOINT_NAMES order: openarm_smolvla's V1_JOINT_LIMITS.
V1_JOINT_LIMITS = np.asarray([
    (-3.490659, 1.396263), (-3.316125, 0.174533), (-1.570796, 1.570796), (0.0, 2.443461),
    (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, 0.043),
    (-1.396263, 3.490659), (-0.174533, 3.316125), (-1.570796, 1.570796), (0.0, 2.443461),
    (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, 0.043),
], dtype=np.float32)
# The server names a camera's colour frames by its role (chest, front) and
# its depth frames by the role and this (openarm_smolvla's data configs).
DEPTH_SUFFIX = "_depth"
DEFAULT_PORT = 8000


def cameras_of(names):
    """The server's camera list -> (roles, roles with depth): ["chest",
    "chest_depth", "front"] -> (("chest", "front"), {"chest"}). A role with
    only depth listed still counts, its colour frames unsent."""
    names = list(names or ())
    roles = []
    for name in names:
        role = name[:-len(DEPTH_SUFFIX)] if name.endswith(DEPTH_SUFFIX) else name
        if role not in roles:
            roles.append(role)
    # The node feeds predict()'s rgb and depth from the first: chest.
    if "chest" in roles:
        roles.remove("chest")
        roles.insert(0, "chest")
    return tuple(roles), {role for role in roles if role + DEPTH_SUFFIX in names}


def parse_server(address):
    """"ws://host:port", "host:port" or "host" -> (host, port)."""
    address = str(address).strip()
    if not address:
        raise ValueError("no SmolVLA server: pass server:=ws://<host>:<port> "
                         "(or set OPENARM_SMOLVLA_SERVER)")
    parsed = urlparse(address if "://" in address else f"ws://{address}")
    try:
        port = parsed.port or DEFAULT_PORT
    except ValueError:
        port = None
    if parsed.scheme != "ws" or not parsed.hostname or port is None:
        raise ValueError(f"not a websocket server: {address!r}; "
                         f"expected ws://<host>:<port>")
    return parsed.hostname, port


class SmolVLAPolicy(Policy):
    """openarm_act's Policy, the model behind a websocket."""

    def __init__(self, checkpoint="", connect_timeout=10.0,
                 request_timeout=15.0, clip_to_limits=True, **_ignored):
        """`checkpoint` is the server, as openarm_act calls the model it loads."""
        self.host, self.port = parse_server(
            checkpoint or os.environ.get("OPENARM_SMOLVLA_SERVER", ""))
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout) or None
        self.clip = bool(clip_to_limits)
        self.ws = None
        self._exit = None
        self.metadata = self._connect()

        self.chunk_size = int(self.metadata["action_horizon"])
        # [height, width]; openarm_act refuses a camera that gives other frames.
        size = self.metadata.get("image_size")
        self.image_size = tuple(int(v) for v in size) if size else None
        # The cameras the model reads, chest first: the node subscribes to
        # them, and predict() is given every one (openarm_act's Policy).
        self.server_cameras = list(self.metadata.get("cameras") or ())
        self.cameras, self.with_depth = cameras_of(self.server_cameras)
        if not self.cameras:
            raise ValueError(f"the model at {self.server} reads no camera")
        self.server_ms = None           # the model's time on the server
        self.round_trip_ms = None       # the same plus the network and encoding

    @property
    def server(self):
        return f"ws://{self.host}:{self.port}"

    @property
    def task(self):
        """What the model was trained to do; the server feeds it this."""
        return self.metadata.get("default_prompt")

    def _connect(self):
        """Open the websocket; -> the server's metadata, checked. Refuses
        quickly when nothing listens rather than waiting for ever."""
        try:
            socket.create_connection((self.host, self.port),
                                     timeout=self.connect_timeout).close()
        except OSError as error:
            raise ConnectionError(
                f"no SmolVLA server at {self.server} ({error}); start "
                f"openarm_smolvla's scripts/serve.py there") from error
        from websockets.sync.client import connect

        exit_stack = contextlib.ExitStack()
        self.ws = exit_stack.enter_context(connect(
            self.server, compression=None, max_size=None,
            open_timeout=self.connect_timeout))
        self._exit = exit_stack
        try:
            metadata = dict(msgpack_numpy.unpackb(
                self.ws.recv(timeout=self.connect_timeout)))
        except Exception:
            self.close()
            raise
        joints = tuple(metadata.get("joint_names") or ())
        if metadata.get("action_dim") != NUM_JOINTS or joints != JOINT_NAMES:
            self.close()
            raise ValueError(
                f"the server at {self.server} is not an OpenArm policy "
                f"(action_dim {metadata.get('action_dim')}, joints {list(joints)})")
        if not metadata.get("default_prompt"):
            self.close()
            raise ValueError(
                f"the model at {self.server} has no single task to run (it was "
                f"trained on several); this node sends no prompt, so give the "
                f"server one: scripts/serve.py default_prompt=...")
        return metadata

    def _reconnect(self):
        """After a lost connection: the same model, or a refusal."""
        metadata = self._connect()
        if int(metadata["action_horizon"]) != self.chunk_size or \
                metadata.get("image_size") != self.metadata.get("image_size") or \
                metadata.get("cameras") != self.metadata.get("cameras"):
            self.close()
            raise RuntimeError(
                f"{self.server} now serves another model "
                f"({metadata.get('checkpoint')}); load it again")
        self.metadata = metadata

    def predict(self, rgb, depth, qpos, cameras=None):
        """`rgb`, `depth`: the first camera's (chest); `cameras`: the
        others the model reads, {role: (rgb, depth)}."""
        if self.ws is None:             # the last call lost the connection
            self._reconnect()
        frames = {self.cameras[0]: (rgb, depth), **(cameras or {})}
        missing = [role for role in self.cameras if role not in frames]
        if missing:
            raise ValueError(f"no frames from {', '.join(missing)}, which the "
                             f"model reads")
        images = {}
        for role in self.cameras:
            role_rgb, role_depth = frames[role]
            if role in self.server_cameras:
                images[role] = np.ascontiguousarray(role_rgb, dtype=np.uint8)
            if role in self.with_depth:
                images[role + DEPTH_SUFFIX] = np.ascontiguousarray(
                    role_depth, dtype=np.uint16)
        observation = {
            "images": images,
            "state": np.asarray(qpos, dtype=np.float32).reshape(NUM_JOINTS),
        }
        started = time.perf_counter()
        try:
            self.ws.send(msgpack_numpy.packb(observation))
            # Bounded: a server that hangs must not hang the node's worker.
            response = self.ws.recv(timeout=self.request_timeout)
        except TimeoutError as error:
            self.close()
            raise TimeoutError(f"{self.server} did not answer within "
                               f"{self.request_timeout:g} s") from error
        except ConnectionClosed as error:
            self.close()
            raise ConnectionError(f"lost the connection to {self.server}; "
                                  f"the next inference reconnects") from error
        except Exception:
            self.close()
            raise
        if isinstance(response, str):   # the server's traceback; it hangs up
            self.close()
            last = response.strip().splitlines()[-1] if response.strip() else ""
            raise RuntimeError(f"the SmolVLA server failed: {last}")
        result = msgpack_numpy.unpackb(response)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
            raise ValueError(f"the server returned actions of shape "
                             f"{actions.shape}, expected [chunk, {NUM_JOINTS}]")
        self.server_ms = (result.get("server_timing") or {}).get("infer_ms")
        self.round_trip_ms = (time.perf_counter() - started) * 1e3
        if self.clip:
            actions = np.clip(actions, V1_JOINT_LIMITS[:, 0], V1_JOINT_LIMITS[:, 1])
        return actions[:self.chunk_size]

    def close(self):
        if self._exit is not None:
            try:
                self._exit.close()
            except Exception:           # noqa: BLE001 -- already broken
                pass
        self._exit = None
        self.ws = None

    def describe(self):
        meta = self.metadata
        step = f" step {meta['step']}" if meta.get("step") is not None else ""
        size = (f"{self.image_size[1]}x{self.image_size[0]} "
                if self.image_size else "")
        cameras = ", ".join(
            f"{role} {'RGB-D' if role in self.with_depth else 'RGB'}"
            for role in self.cameras)
        return (f"SmolVLAPolicy: {meta.get('model')}/{meta.get('finetune')}"
                f"{step} at {self.server}, chunk {self.chunk_size} at "
                f"{meta.get('fps')} Hz, {size}{cameras}, task {self.task!r}"
                + ("" if self.clip else ", V1 limits NOT enforced"))
