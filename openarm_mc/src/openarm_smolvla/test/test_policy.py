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

"""SmolVLAPolicy against a fake server speaking openarm_smolvla's protocol."""

import os
import socket
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fake_server import FakeServer  # noqa: E402

from openarm_smolvla.policy import (  # noqa: E402
    V1_JOINT_LIMITS, SmolVLAPolicy, parse_server,
)


def observation(qpos=None):
    rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth = np.full((480, 640), 600, dtype=np.uint16)
    if qpos is None:
        qpos = np.linspace(0.1, 0.4, 16, dtype=np.float32)
        qpos[7] = qpos[15] = 0.02       # grippers, m
    return rgb, depth, qpos


@pytest.fixture(autouse=True)
def no_environment(monkeypatch):
    monkeypatch.delenv("OPENARM_SMOLVLA_SERVER", raising=False)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ address

@pytest.mark.parametrize("address, expected", [
    ("ws://gpu:9000", ("gpu", 9000)),
    ("gpu:9000", ("gpu", 9000)),
    ("gpu", ("gpu", 8000)),
    (" ws://10.0.0.2:8000 ", ("10.0.0.2", 8000)),
])
def test_server_addresses(address, expected):
    assert parse_server(address) == expected


@pytest.mark.parametrize("address", ["", "http://gpu:8000", "ws://", "ws://gpu:port",
                                     "/home/me/policy.onnx"])
def test_what_is_not_a_server(address):
    with pytest.raises(ValueError):
        parse_server(address)


def test_no_server_is_refused_quickly():
    with pytest.raises(ConnectionError, match="no SmolVLA server"):
        SmolVLAPolicy(checkpoint=f"ws://127.0.0.1:{free_port()}", connect_timeout=1.0)


# ----------------------------------------------------------------- metadata

def test_the_server_says_what_the_model_is():
    with FakeServer() as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        assert policy.chunk_size == 50
        assert policy.image_size == (480, 640)      # what ActNode checks
        assert policy.cameras == ("chest",) and policy.with_depth == {"chest"}
        assert "fake" in policy.describe() and "640x480" in policy.describe()
        policy.close()


def test_a_server_for_another_robot_is_refused():
    with FakeServer(metadata={"action_dim": 14, "joint_names": ["j"] * 14}) as server:
        with pytest.raises(ValueError, match="not an OpenArm policy"):
            SmolVLAPolicy(checkpoint=server.url)


def test_the_server_comes_from_the_environment(monkeypatch):
    with FakeServer() as server:
        monkeypatch.setenv("OPENARM_SMOLVLA_SERVER", server.url)
        assert SmolVLAPolicy().port == server.port


# --------------------------------------------------------------------- task

def test_no_prompt_is_sent_the_server_gives_its_task():
    """The model is trained on one task; the server fills it in."""
    with FakeServer() as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        policy.predict(*observation())
        assert "prompt" not in server.received[-1]
        assert policy.task == server.metadata["default_prompt"]
        assert repr(policy.task) in policy.describe()


def test_a_model_of_several_tasks_is_refused():
    with FakeServer(metadata={"default_prompt": None}) as server:
        with pytest.raises(ValueError, match="no single task"):
            SmolVLAPolicy(checkpoint=server.url)


# ------------------------------------------------------------------ predict

def test_the_observation_goes_raw():
    with FakeServer() as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        rgb, depth, qpos = observation()
        actions = policy.predict(rgb, depth, qpos)
        sent = server.received[-1]
        np.testing.assert_array_equal(sent["images"]["chest"], rgb)
        np.testing.assert_array_equal(sent["images"]["chest_depth"], depth)
        assert sent["images"]["chest_depth"].dtype == np.uint16
        np.testing.assert_allclose(sent["state"], qpos)
        # The fake holds the pose: the chunk is the observation, 50 times.
        assert actions.shape == (50, 16) and actions.dtype == np.float32
        np.testing.assert_allclose(actions, np.tile(qpos, (50, 1)), atol=1e-6)
        assert policy.server_ms == 12.5 and policy.round_trip_ms > 0


def test_an_rgb_model_is_not_sent_depth():
    with FakeServer(metadata={"cameras": ["chest"]}) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        policy.predict(*observation())
        assert list(server.received[-1]["images"]) == ["chest"]


def test_targets_are_clipped_to_the_joint_limits():
    wild = np.full((50, 16), 10.0, dtype=np.float32)
    with FakeServer(act=lambda obs: wild) as server:
        actions = SmolVLAPolicy(checkpoint=server.url).predict(*observation())
        np.testing.assert_allclose(actions[0], V1_JOINT_LIMITS[:, 1])
        unclipped = SmolVLAPolicy(checkpoint=server.url, clip_to_limits=False)
        assert unclipped.predict(*observation()).max() == 10.0


def test_a_wrong_shape_is_refused():
    with FakeServer(act=lambda obs: np.zeros((50, 14))) as server:
        with pytest.raises(ValueError, match="shape"):
            SmolVLAPolicy(checkpoint=server.url).predict(*observation())


# ------------------------------------------------------------- failures

def test_a_server_error_is_raised_then_it_reconnects():
    with FakeServer(fail_next=1) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        with pytest.raises(RuntimeError, match="no image from camera"):
            policy.predict(*observation())
        assert policy.ws is None
        assert policy.predict(*observation()).shape == (50, 16)
        assert server.connections == 2


def test_a_server_that_went_away_is_said_plainly():
    with FakeServer() as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        server.drop()
        with pytest.raises(ConnectionError, match="lost the connection"):
            policy.predict(*observation())
        assert policy.ws is None
        assert policy.predict(*observation()).shape == (50, 16)


def test_a_server_that_hangs_times_out():
    with FakeServer(delay=1.0) as server:
        policy = SmolVLAPolicy(checkpoint=server.url, request_timeout=0.2)
        with pytest.raises(TimeoutError, match="did not answer"):
            policy.predict(*observation())
        assert policy.ws is None


def test_another_model_behind_the_same_address_is_refused():
    with FakeServer() as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        policy.close()
        server.metadata["action_horizon"] = 30
        with pytest.raises(RuntimeError, match="another model"):
            policy.predict(*observation())


# ---------------------------------------------------------- two cameras

TWO_CAMERAS = {"cameras": ["chest", "chest_depth", "front", "front_depth"]}


def test_a_two_camera_model_reads_both_chest_first():
    with FakeServer(metadata={"cameras": ["front", "front_depth", "chest", "chest_depth"]}) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        assert policy.cameras == ("chest", "front")
        assert "front RGB-D" in policy.describe()


def test_every_camera_goes_by_its_name():
    with FakeServer(metadata=TWO_CAMERAS) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        rgb, depth, qpos = observation()
        front_rgb = np.full_like(rgb, 7)
        front_depth = np.full_like(depth, 900)
        policy.predict(rgb, depth, qpos, cameras={"front": (front_rgb, front_depth)})
        sent = server.received[-1]["images"]
        assert sorted(sent) == ["chest", "chest_depth", "front", "front_depth"]
        np.testing.assert_array_equal(sent["chest"], rgb)
        np.testing.assert_array_equal(sent["front"], front_rgb)
        np.testing.assert_array_equal(sent["front_depth"], front_depth)


def test_a_camera_the_model_reads_cannot_be_left_out():
    with FakeServer(metadata=TWO_CAMERAS) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        with pytest.raises(ValueError, match="no frames from front"):
            policy.predict(*observation())


def test_a_camera_with_colour_only():
    with FakeServer(metadata={"cameras": ["chest", "chest_depth", "front"]}) as server:
        policy = SmolVLAPolicy(checkpoint=server.url)
        rgb, depth, qpos = observation()
        policy.predict(rgb, depth, qpos, cameras={"front": (rgb, depth)})
        assert sorted(server.received[-1]["images"]) == ["chest", "chest_depth", "front"]
