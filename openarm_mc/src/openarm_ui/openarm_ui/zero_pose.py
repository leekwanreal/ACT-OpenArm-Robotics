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

"""Take the arm to its zero pose from wherever it is, whoever drove it there.

The dashboard's Zero pose button. It runs as the `ui` source: the page cuts
the old source off at the proxy, then streams this motion from the measured
pose to every arm joint at 0 along a fifth-degree smoothstep (zero speed and
acceleration at both ends), its farthest joint averaging `speed`. The
grippers stay where they are: closing one on an object, or opening it under
one, is not what "zero" should do by surprise.

Several things may want the `ui` targets at once -- the sliders of any open
page, a second click, a source switch from a terminal -- so the motion is a
small state machine guarded by its owner's lock (ros_node.py):

    idle -> starting    claimed: a second claim is refused, sliders ignored
    starting -> moving  the proxy took ui, the pose is fresh: the clock starts
    moving -> finishing the motion ended; the source is handed back
    any -> idle         done, or cancelled (Stop, another source, no state)

Each claim gets a token, so a step that comes back late from a service call
cannot act on a later claim, or on one that was cancelled meanwhile.

Kept free of ROS so it can be tested on its own.
"""

IDLE = "idle"
STARTING = "starting"
MOVING = "moving"
FINISHING = "finishing"


def smoothstep(t):
    """0 -> 1 with zero speed and acceleration at both ends."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


class ZeroPoseMotion:
    """origin {id: value} -> goal {id: value}, as a function of time."""

    def __init__(self, origin, goal, start, speed=0.5, min_duration=1.5,
                 arm_ids=None):
        self.origin = dict(origin)
        # Whatever the goal leaves out stays where it is.
        self.goal = {motor_id: goal.get(motor_id, value)
                     for motor_id, value in self.origin.items()}
        self.start = float(start)
        arm = set(arm_ids) if arm_ids is not None else set(self.origin)
        gap = max((abs(self.goal[i] - self.origin[i]) for i in self.origin
                   if i in arm), default=0.0)
        self.duration = max(float(min_duration), gap / max(float(speed), 1e-6))

    def progress(self, now):
        return min(1.0, max(0.0, (now - self.start) / self.duration))

    def remaining(self, now):
        return max(0.0, self.start + self.duration - now)

    def sample(self, now):
        """-> ({id: value}, done)."""
        t = self.progress(now)
        blend = smoothstep(t)
        targets = {motor_id: origin + (self.goal[motor_id] - origin) * blend
                   for motor_id, origin in self.origin.items()}
        return targets, t >= 1.0


class ZeroPoseState:
    """Who owns the ui targets, and what the last attempt came to.

    Not thread-safe by itself: every call is made under the owner's lock.
    """

    def __init__(self):
        self.phase = IDLE
        self.token = 0
        self.motion = None
        self.restore = None     # source to hand the arm to once at zero
        self.result_id = 0
        self.result = None      # (ok, message) of the last attempt

    @property
    def busy(self):
        return self.phase != IDLE

    def claim(self):
        """-> (token, None), or (None, why not)."""
        if self.busy:
            return None, "the arm is already going to the zero pose"
        self.token += 1
        self.phase = STARTING
        self.motion = None
        return self.token, None

    def begin(self, token, motion, restore):
        """Start the clock on a claim; -> None, or why it cannot."""
        if token != self.token or self.phase != STARTING:
            return "the zero pose was cancelled"
        self.phase = MOVING
        self.motion = motion
        self.restore = restore
        return None

    def arrived(self):
        """The motion ended: -> the token, to hand the source back with."""
        self.phase = FINISHING
        return self.token

    def finish(self, token, ok, message):
        """End the claim `token` (None: whichever is running)."""
        if not self.busy or (token is not None and token != self.token):
            return False
        self.phase = IDLE
        self.motion = None
        self.restore = None
        self.result_id += 1
        self.result = (ok, message)
        return True

    def snapshot(self, now):
        motion = self.motion if self.phase == MOVING else None
        return {
            "phase": self.phase,
            "progress": motion.progress(now) if motion else
                        (1.0 if self.phase == FINISHING else 0.0),
            "remaining_s": motion.remaining(now) if motion else 0.0,
            "last": None if self.result is None else {
                "id": self.result_id, "ok": self.result[0],
                "message": self.result[1]},
        }
