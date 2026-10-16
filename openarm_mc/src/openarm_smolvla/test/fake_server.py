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

"""A stand-in for openarm_smolvla's scripts/serve.py: its protocol, no model.

It answers every observation with the chunk that holds the observed pose,
as openarm_act's HoldPolicy predicts, so the node's plumbing can be tried
without a GPU (in the simulator, say):

    python3 src/openarm_smolvla/test/fake_server.py --port 8000
    ros2 launch openarm_smolvla smolvla.launch.py server:=ws://localhost:8000

The tests run it in a thread and look at what it was sent.
"""

import argparse
import threading
import time

import numpy as np
from websockets.sync.server import serve

from openarm_smolvla import msgpack_numpy
from openarm_smolvla.policy import JOINT_NAMES

METADATA = {
    "robot": "openarm_v1_bimanual",
    "joint_names": list(JOINT_NAMES),
    "action_dim": len(JOINT_NAMES),
    "action_horizon": 50,
    "fps": 50,
    "model": "smolvla",
    "finetune": "fake",
    "cameras": ["chest", "chest_depth"],
    "delta_actions": False,
    "depth_as": "channel",
    "default_prompt": "pick up the green box and place it on the white paper",
    "image_size": [480, 640],
    "checkpoint": "fake",
    "step": 0,
}


class FakeServer:
    """openarm_smolvla's PolicyServer, with `act(obs) -> actions` for a model."""

    def __init__(self, metadata=None, act=None, port=0, fail_next=0, delay=0.0):
        self.metadata = {**METADATA, **(metadata or {})}
        self.act = act or self.hold
        self.fail_next = fail_next      # this many answers are tracebacks
        self.delay = delay              # seconds before each answer
        self.received = []              # every observation, decoded
        self.connections = 0
        self.live = []                  # open connections, for drop()
        self.server = serve(self._handle, "127.0.0.1", port,
                            compression=None, max_size=None)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def hold(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32)
        return np.tile(state, (self.metadata["action_horizon"], 1))

    def drop(self):
        """Hang up on every client, as a server that went away does."""
        for websocket in list(self.live):
            websocket.close()

    def _handle(self, websocket):
        self.connections += 1
        self.live.append(websocket)
        websocket.send(msgpack_numpy.packb(self.metadata))
        for message in websocket:
            obs = msgpack_numpy.unpackb(message)
            self.received.append(obs)
            if self.delay:
                time.sleep(self.delay)
            if self.fail_next:
                self.fail_next -= 1
                websocket.send("Traceback (most recent call last):\n"
                               "ValueError: no image from camera(s) ['chest']")
                websocket.close()
                return
            actions = np.asarray(self.act(obs), dtype=np.float32)
            websocket.send(msgpack_numpy.packb(
                {"actions": actions, "server_timing": {"infer_ms": 12.5}}))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.05,
                        help="seconds per chunk, as a model would take")
    args = parser.parse_args()
    with FakeServer(port=args.port, delay=args.delay) as server:
        print(f"fake SmolVLA server on {server.url} (holds the observed pose)")
        try:
            server.thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
