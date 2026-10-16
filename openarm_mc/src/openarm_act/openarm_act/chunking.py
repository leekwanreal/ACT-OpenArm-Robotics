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

"""Turn overlapping action chunks into one action per control step.

The policy predicts `chunk_size` actions at a time, the first one for the step
its observation was taken at. Inference takes longer than a step, so by the
time a chunk arrives its first few actions are already in the past; they are
simply never asked for. A new chunk is requested every few steps, so several
chunks usually cover the same step.

With temporal ensembling every chunk that covers a step votes, weighted
exp(-k * age), age being how many steps ago the chunk was predicted -- the
scheme act_pipeline's TemporalEnsemblePolicy uses, and so what the policy
was evaluated with: the newest prediction weighs most. `weighting="oldest"`
gives the ACT paper's opposite order, which damps a fresh chunk harder.
Without ensembling, the newest chunk alone decides.

Kept free of ROS and torch so it can be tested on its own.
"""

import math

import numpy as np


class ActionChunker:
    """Holds the chunks still in play and answers `action(step)`."""

    def __init__(self, chunk_size, temporal_ensemble=True, ensemble_k=0.01,
                 weighting="newest"):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if weighting not in ("newest", "oldest"):
            raise ValueError("weighting must be newest or oldest")
        self.chunk_size = int(chunk_size)
        self.temporal_ensemble = bool(temporal_ensemble)
        self.ensemble_k = float(ensemble_k)
        self.weighting = weighting
        # (start_step, actions[chunk_size, dof]), oldest first.
        self.chunks = []

    def reset(self):
        self.chunks.clear()

    def add(self, start_step, actions):
        """Record a chunk whose actions[i] is meant for step start_step + i."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"a chunk must be [steps, dof], got {actions.shape}")
        self.chunks.append((int(start_step), actions))
        self.chunks.sort(key=lambda chunk: chunk[0])

    def prune(self, step):
        """Drop chunks that end before `step`; nothing will ask for them."""
        self.chunks = [(start, actions) for start, actions in self.chunks
                       if start + len(actions) > step]

    def covers(self, step):
        return any(start <= step < start + len(actions)
                   for start, actions in self.chunks)

    def action(self, step):
        """The action for `step`, or None when no chunk reaches it."""
        covering = [(start, actions) for start, actions in self.chunks
                    if start <= step < start + len(actions)]
        if not covering:
            return None
        votes = np.stack([actions[step - start] for start, actions in covering])
        if not self.temporal_ensemble:
            return votes[-1].copy()
        if self.weighting == "newest":
            # act_pipeline: weight by how long ago the chunk was predicted.
            weights = np.asarray([math.exp(-self.ensemble_k * (step - start))
                                  for start, _ in covering])
        else:
            # The ACT paper: by rank, the oldest chunk first.
            weights = np.asarray([math.exp(-self.ensemble_k * i)
                                  for i in range(len(covering))])
        weights /= weights.sum()
        return np.tensordot(weights, votes, axes=1)

    def latest(self):
        """(start_step, actions) of the newest chunk, or None."""
        return self.chunks[-1] if self.chunks else None

    def remaining(self, step):
        """Steps from `step` until the newest chunk runs out."""
        if not self.chunks:
            return 0
        return max(0, max(start + len(actions)
                          for start, actions in self.chunks) - step)
