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

"""The smoother and the ensembling, against act_pipeline's own code."""

import math

import numpy as np

from openarm_act.chunking import ActionChunker
from openarm_act.smoothing import TrajectorySmoother


class ReferenceSmoother:
    """tools/infer_robot.py's TrajectorySmoother, copied as it is."""

    def __init__(self, vel_scale=1.0, dt=0.02):
        self.dt = dt
        self.scale = max(0.1, min(2.0, vel_scale))
        base_v_lim = np.array([1.0, 1.0, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5], dtype=np.float32)
        base_a_lim = np.array([3.0, 3.0, 5.0, 5.0, 8.0, 8.0, 8.0, 12.0], dtype=np.float32)
        self.v_lim = np.concatenate([base_v_lim, base_v_lim]) * self.scale
        self.a_lim = np.concatenate([base_a_lim, base_a_lim]) * self.scale
        self.q_cmd = None
        self.v_cmd = np.zeros(16, dtype=np.float32)

    def reset(self, initial_qpos):
        self.q_cmd = np.array(initial_qpos, dtype=np.float32).copy()
        self.v_cmd = np.zeros(16, dtype=np.float32)

    def step(self, target_qpos):
        if self.q_cmd is None:
            self.reset(target_qpos)
            return self.q_cmd.copy()
        diff = target_qpos - self.q_cmd
        desired_vel = diff / self.dt
        desired_vel = np.clip(desired_vel, -self.v_lim, self.v_lim)
        max_dv = self.a_lim * self.dt
        dv = np.clip(desired_vel - self.v_cmd, -max_dv, max_dv)
        self.v_cmd += dv
        self.q_cmd += self.v_cmd * self.dt
        return self.q_cmd.copy()


class ReferenceEnsemble:
    """act_pipeline.utils.temporal_ensemble.TemporalEnsemblePolicy, copied."""

    def __init__(self, chunk_size=50, action_dim=16, ensemble_m=0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.weights = np.exp(-ensemble_m * np.arange(chunk_size))
        self.action_buffer = []
        self.current_step = 0

    def update(self, action_chunk):
        target_len = self.current_step + self.chunk_size
        while len(self.action_buffer) < target_len:
            self.action_buffer.append([])
        for offset in range(self.chunk_size):
            step = self.current_step + offset
            self.action_buffer[step].append((action_chunk[offset], self.weights[offset]))
        total_weight = 0.0
        weighted = np.zeros(self.action_dim, dtype=np.float32)
        for action, weight in self.action_buffer[self.current_step]:
            weighted += action * weight
            total_weight += weight
        self.current_step += 1
        return weighted / max(total_weight, 1e-6)


def test_smoother_matches_infer_robot_step_for_step():
    rng = np.random.default_rng(0)
    for scale in (0.5, 1.0, 1.7):
        mine, ref = TrajectorySmoother(scale, 0.02), ReferenceSmoother(scale, 0.02)
        start = rng.uniform(-1, 1, 16).astype(np.float32)
        mine.reset(start)
        ref.reset(start)
        for _ in range(200):
            # Jumps as large as a bad prediction, so the limits bite.
            target = rng.uniform(-2, 2, 16).astype(np.float32)
            np.testing.assert_allclose(mine.step(target), ref.step(target), atol=1e-6)


def test_smoother_holds_its_limits():
    smoother = TrajectorySmoother(1.0, 0.02)
    smoother.reset(np.zeros(16))
    previous, velocity = np.zeros(16), np.zeros(16)
    for _ in range(100):
        command = smoother.step(np.full(16, 3.0))
        new_velocity = (command - previous) / 0.02
        assert np.all(np.abs(new_velocity) <= smoother.v_lim + 1e-5)
        assert np.all(np.abs(new_velocity - velocity) / 0.02 <= smoother.a_lim + 1e-3)
        previous, velocity = command, new_velocity


def test_ensembling_one_chunk_a_step_is_temporal_ensemble_policy():
    # The synchronous loop: a chunk filed at every step, the step's action taken.
    rng = np.random.default_rng(1)
    for m in (0.01, 0.1):
        chunker = ActionChunker(50, temporal_ensemble=True, ensemble_k=m, weighting="newest")
        ref = ReferenceEnsemble(50, 16, m)
        for step in range(120):
            chunk = rng.normal(size=(50, 16)).astype(np.float32)
            chunker.add(step, chunk)
            chunker.prune(step)
            np.testing.assert_allclose(chunker.action(step), ref.update(chunk), atol=1e-5)


def test_newest_chunk_weighs_most():
    chunker = ActionChunker(50, ensemble_k=0.1)
    chunker.add(0, np.zeros((50, 1)))
    chunker.add(10, np.ones((50, 1)))
    weight_new, weight_old = 1.0, math.exp(-0.1 * 10)
    expected = weight_new / (weight_new + weight_old)
    assert abs(chunker.action(10)[0] - expected) < 1e-9
