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

"""Hand the arm to a new source without a jump.

A source starts from wherever it is: a leader arm in another pose, a page
whose sliders were set before the robot moved. Forwarded as is, its first
command is a step the controller takes as fast as its limits allow. So the
first command after a switch is compared with where the robot really is
(its joint states), and when they are apart the output eases from the
measured pose onto the source's targets -- which keep moving meanwhile --
along a fifth-degree smoothstep, its farthest arm joint averaging `speed`.
Close enough, and the source passes straight through: a policy or a plan
that starts on the robot is not bent.

"Where the robot is" is the setpoint the controller holds, when it holds
one (hold_pose): with no gravity compensation the measured pose hangs a
little below it, and a first command at the measured pose lets every joint
drop by that much in one 20 ms segment -- a visible jerk. The measured pose
is the fallback, for motors that are off or have not been commanded yet.

Kept free of ROS so it can be tested on its own.
"""


# The controller's states in which its setpoint is what the motors hold.
HOLDING_STATES = ("commanding", "holding")


def hold_pose(measured, setpoint=None, setpoint_state=None, setpoint_age=None,
              max_age=0.25):
    """{id: value} the arm is held at: the controller's setpoint over the
    measured pose, when that setpoint is fresh and actually on the motors."""
    pose = dict(measured)
    if setpoint and setpoint_state in HOLDING_STATES and \
            setpoint_age is not None and setpoint_age <= max_age:
        pose.update(setpoint)
    return pose


def smoothstep(t):
    """0 -> 1 with zero speed and acceleration at both ends."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


class SwitchBlend:
    """Per joint id: measured pose -> source targets, after a switch."""

    def __init__(self, arm_ids, threshold=0.02, speed=0.5, min_duration=0.5):
        self.arm_ids = set(arm_ids)
        self.threshold = float(threshold)   # rad: closer needs no easing
        self.speed = float(speed)           # rad/s, farthest arm joint
        self.min_duration = float(min_duration)
        self.pending = False                # a switch waits for its first command
        self.start = None
        self.duration = 0.0
        self.origin = {}                    # id -> measured value at the start

    @property
    def active(self):
        return self.start is not None

    def switched(self):
        """A new source was selected: ease its first command in if needed."""
        self.pending = True
        self.start = None
        self.origin = {}

    def output(self, targets, measured, now):
        """What to send for `targets` ({id: value}) at `now`.

        `measured` is the robot's pose ({id: value}). Returns the targets
        themselves once no easing is (or is any longer) needed.
        """
        if self.pending:
            self.pending = False
            gaps = [abs(value - measured[motor_id])
                    for motor_id, value in targets.items()
                    if motor_id in self.arm_ids and motor_id in measured]
            gap = max(gaps, default=0.0)
            if gap >= self.threshold:
                self.start = now
                self.duration = max(self.min_duration, gap / max(self.speed, 1e-6))
                self.origin = {}
        if self.start is None:
            return dict(targets)

        blend = smoothstep((now - self.start) / self.duration)
        out = {}
        for motor_id, value in targets.items():
            if motor_id not in self.origin:
                # Joints joining later (a gripper's first message) ease in
                # from where they are too, on the same clock.
                self.origin[motor_id] = measured.get(motor_id, value)
            origin = self.origin[motor_id]
            out[motor_id] = origin + (value - origin) * blend
        if blend >= 1.0:
            self.start = None
        return out
