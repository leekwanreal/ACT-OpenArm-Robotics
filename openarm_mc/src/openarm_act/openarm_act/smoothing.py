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

"""Per-joint velocity and acceleration limits on the policy's commands.

act_pipeline's tools/infer_robot.py (and sim/model_inference_engine.py)
put every ensembled action through this before it reaches the motors, so
the policy is evaluated with it: a sudden jump in the prediction becomes a
ramp the motor can follow. The limits follow each motor:

    DM8009  J1 J2     1.0 rad/s   3.0 rad/s^2
    DM4340  J3 J4     1.5 rad/s   5.0 rad/s^2
    DM4310  J5 J6 J7  2.5 rad/s   8.0 rad/s^2
    DM4310  gripper   3.5 /s     12.0 /s^2   (in the gripper's unit)

all scaled by vel_scale (0.1 to 2.0). Kept free of ROS so it can be tested
on its own.
"""

import numpy as np

ARM_V_LIMIT = np.array([1.0, 1.0, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5], dtype=np.float32)
ARM_A_LIMIT = np.array([3.0, 3.0, 5.0, 5.0, 8.0, 8.0, 8.0, 12.0], dtype=np.float32)


class TrajectorySmoother:
    """One command per control step, its speed and acceleration clamped."""

    def __init__(self, vel_scale=1.0, dt=0.02):
        self.dt = float(dt)
        self.scale = max(0.1, min(2.0, float(vel_scale)))
        # Left J1..gripper, then right J1..gripper: the dataset's order.
        self.v_lim = np.concatenate([ARM_V_LIMIT, ARM_V_LIMIT]) * self.scale
        self.a_lim = np.concatenate([ARM_A_LIMIT, ARM_A_LIMIT]) * self.scale
        self.q_cmd = None
        self.v_cmd = np.zeros(16, dtype=np.float32)

    def reset(self, initial_qpos):
        """Start from `initial_qpos`, at rest."""
        self.q_cmd = np.array(initial_qpos, dtype=np.float32).copy()
        self.v_cmd = np.zeros(16, dtype=np.float32)

    def step(self, target_qpos):
        """The next command towards `target_qpos`."""
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        if self.q_cmd is None:
            self.reset(target_qpos)
            return self.q_cmd.copy()
        desired_vel = (target_qpos - self.q_cmd) / self.dt
        desired_vel = np.clip(desired_vel, -self.v_lim, self.v_lim)
        max_dv = self.a_lim * self.dt
        self.v_cmd += np.clip(desired_vel - self.v_cmd, -max_dv, max_dv)
        self.q_cmd += self.v_cmd * self.dt
        return self.q_cmd.copy()
