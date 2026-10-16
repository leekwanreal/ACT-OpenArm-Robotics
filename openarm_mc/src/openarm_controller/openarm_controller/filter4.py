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

"""The other way from sparse targets to the 400 Hz setpoint: a filter.

`interpolator: filter4` on the controller, instead of spline.py's quintic
segments. Each joint's newest target u drives a critically damped
fourth-order filter, four poles at -w, stepped every control tick with
semi-implicit Euler:

    q''''_d = w^4 (u - q_d) - 4 w^3 q'_d - 6 w^2 q''_d - 4 w q'''_d

and its q_d and q'_d are what the MIT frame carries as the position and the
velocity the motor's PD tracks. Position through jerk stay continuous however
the targets step, a smooth stream is followed about 4/w seconds behind, and a
step settles without overshoot (95 % in about 7.75/w).

q'_d is kept within the joint's speed limit (spline.KINEMATIC_LIMITS): a step
too wide for the joint is crossed at that speed instead of the filter's own,
which for a step of A peaks near 0.224 w A. Its acceleration is not limited:
a wide step starts and ends briskly (about 90 rad/s^2 for 1 rad at w = 30).
"""

DEFAULT_BANDWIDTH = 30.0          # rad/s: w of the four poles

# Close enough to the target, and slow enough, to stop commanding.
SETTLED_POSITION = 1e-4           # rad
SETTLED_VELOCITY = 1e-3           # rad/s


class FourthOrderFilter:
    """One joint's filter state; the same calls as spline.JointInterpolator."""

    def __init__(self, v_max=2.0, a_max=6.0, initial_q=0.0, w=DEFAULT_BANDWIDTH):
        self.v_max = abs(float(v_max))
        self.a_max = abs(float(a_max))          # unused: the filter sets the pace
        self.w = float(w)
        self.set_state(initial_q)

    def set_state(self, q, v=0.0, a=0.0):
        """Snap onto a measured position (and speed), with nowhere to go."""
        self.q_d = float(q)
        self.q_dot_d = float(v)
        self.q_dotdot_d = float(a)
        self.q_dotdotdot_d = 0.0
        self.target_q = float(q)

    def update_target(self, target_q, dt_target=None):
        """The newest command u. The filter, not the interval, sets the pace;
        `dt_target` is accepted so it drops in for the quintic layer."""
        self.target_q = float(target_q)

    @property
    def current_q(self):
        return self.q_d

    @property
    def current_v(self):
        return self.q_dot_d

    @property
    def settled(self):
        """At the target and at rest: nothing left worth commanding."""
        return (abs(self.target_q - self.q_d) < SETTLED_POSITION
                and abs(self.q_dot_d) < SETTLED_VELOCITY)

    def step(self, dt):
        """Advance dt seconds towards the command (semi-implicit Euler).

        -> (q_d, q'_d): the setpoint and the velocity the MIT frame carries.
        """
        w = self.w
        w2 = w * w
        w3 = w2 * w
        w4 = w3 * w
        q_d4 = (w4 * (self.target_q - self.q_d) - 4.0 * w3 * self.q_dot_d
                - 6.0 * w2 * self.q_dotdot_d - 4.0 * w * self.q_dotdotdot_d)
        self.q_dotdotdot_d += dt * q_d4
        self.q_dotdot_d += dt * self.q_dotdotdot_d
        self.q_dot_d += dt * self.q_dotdot_d
        if abs(self.q_dot_d) > self.v_max:
            # At the joint's speed: hold it there, and drop the acceleration
            # and jerk pushing past it so they do not wind up meanwhile.
            self.q_dot_d = self.v_max if self.q_dot_d > 0 else -self.v_max
            if self.q_dotdot_d * self.q_dot_d > 0:
                self.q_dotdot_d = 0.0
            if self.q_dotdotdot_d * self.q_dot_d > 0:
                self.q_dotdotdot_d = 0.0
        self.q_d += dt * self.q_dot_d
        return self.q_d, self.q_dot_d
