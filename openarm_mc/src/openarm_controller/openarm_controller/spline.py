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

"""Quintic smoothing between sparse joint targets and the 400 Hz CAN loop.

Targets arrive on /openarm/joint_commands at whatever rate the publisher runs
-- 50 Hz from an ACT policy, slower and irregular from teleop. Sending those
straight to the motors steps the setpoint, which the arm shows as jerk. Each
new target instead starts a quintic segment from the joint's current position,
velocity and acceleration, so position, velocity and acceleration all stay
continuous across targets and the 400 Hz samples come out smooth.

A stream of targets (teleop, a policy) is followed, not stopped at: a segment
ends at the target with the speed the targets themselves move at, and lasts
as long as the stream's own interval, stretched only when that curve would
break the joint's limits. A lone target (a slider, a zero pose) is
approached from rest to rest within the limits. When a stream stops while
the joint still moves, it brakes within a_max, by at most BRAKE_OVERSHOOT
past the last target. (Ending every segment at rest, and sizing each from
rest, made a 1 rad/s leader arm trail by 100-170 ms on the shoulders.)

Ported from openarm_can/sim/spline_interpolator.py, keeping its coefficients
and its per-joint kinematic limits. filter4.py is the alternative
(`interpolator: filter4` on the controller).
"""

import math

from .filter4 import DEFAULT_BANDWIDTH, FourthOrderFilter

# Per-joint limits (v_max rad/s, a_max rad/s^2), indexed by joint 1..8 within
# an arm. A segment is stretched when the requested move would exceed them.
KINEMATIC_LIMITS = {
    1: (1.2, 3.5),   # J1 shoulder, DM8009
    2: (1.2, 3.5),   # J2 shoulder, DM8009
    3: (2.0, 6.0),   # J3 elbow, DM4340
    4: (2.0, 6.0),   # J4 elbow, DM4340
    5: (3.0, 10.0),  # J5 wrist, DM4310
    6: (3.0, 10.0),  # J6 wrist, DM4310
    7: (3.0, 10.0),  # J7 wrist, DM4310
    8: (2.5, 10.0),  # J8 gripper, DM4310
}


# Targets closer together than this are one stream; further apart, the next
# one starts from rest.
STREAM_GAP_S = 0.1
# How far past the last target a joint may run while braking when a stream
# stops mid-motion; it brakes harder rather than run further.
BRAKE_OVERSHOOT = 0.03
# Samples of a planned segment checked against the limits.
LIMIT_SAMPLES = 12
# For its acceleration, a stream's segment is stretched to at most this many
# of its intervals: further, it would plan to stay behind the stream, and
# the lag would grow with every target. Within it the arm follows the
# stream's own motion, which a leader arm or a policy's smoother keeps
# within reason. Speed is never let past v_max.
STREAM_STRETCH = 1.5


class QuinticSegment:
    """q(t) = c0 + c1 t + ... + c5 t^5 through both endpoints' q, v and a."""

    def __init__(self, q0, v0, a0, q1, v1, a1, duration):
        self.T = max(1e-4, float(duration))
        self.q1 = float(q1)

        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        dq = float(q1) - float(q0)

        self.c0 = float(q0)
        self.c1 = float(v0)
        self.c2 = 0.5 * float(a0)
        self.c3 = (20.0 * dq - (8.0 * v1 + 12.0 * v0) * T
                   - (3.0 * a0 - a1) * T2) / (2.0 * T3)
        self.c4 = (30.0 * (-dq) + (14.0 * v1 + 16.0 * v0) * T
                   + (3.0 * a0 - 2.0 * a1) * T2) / (2.0 * T4)
        # + (a1 - a0) T^2: with the minus sign the port carried, a segment
        # planned while the joint accelerates ends off its target, by
        # (a0 - a1) T^2 / 2, and step() then snapped the setpoint onto it.
        self.c5 = (12.0 * dq - 6.0 * (v1 + v0) * T
                   + (a1 - a0) * T2) / (2.0 * T5)

    def evaluate(self, t):
        """Return (position, velocity, acceleration) at t, clamped to [0, T]."""
        t = min(max(t, 0.0), self.T)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        pos = (self.c0 + self.c1 * t + self.c2 * t2
               + self.c3 * t3 + self.c4 * t4 + self.c5 * t5)
        vel = (self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * t2
               + 4.0 * self.c4 * t3 + 5.0 * self.c5 * t4)
        acc = (2.0 * self.c2 + 6.0 * self.c3 * t
               + 12.0 * self.c4 * t2 + 20.0 * self.c5 * t3)
        return pos, vel, acc


class JointInterpolator:
    """One joint's running quintic state, stepped by the CAN loop."""

    def __init__(self, v_max=2.0, a_max=6.0, initial_q=0.0):
        self.v_max = abs(float(v_max))
        self.a_max = abs(float(a_max))
        self.current_q = float(initial_q)
        self.current_v = 0.0
        self.current_a = 0.0
        self.target_q = float(initial_q)
        self.segment = None
        self.segment_time = 0.0
        self.segment_duration = 0.02
        # Seconds since the last new target: tells a stream from a lone one.
        self.since_target = float("inf")
        # The stream's own speed at its last target, for its acceleration.
        self.stream_v = 0.0

    def set_state(self, q, v=0.0, a=0.0):
        """Snap the interpolator onto a measured position.

        Used when the motor is first enabled or resynchronised, so the first
        segment starts from where the arm actually is rather than from 0.
        """
        self.current_q = float(q)
        self.current_v = float(v)
        self.current_a = float(a)
        self.target_q = float(q)
        self.segment = None
        self.segment_time = 0.0
        self.since_target = float("inf")
        self.stream_v = 0.0

    def update_target(self, target_q, dt_target=0.02):
        """Plan a new segment from the current state to `target_q`.

        In a stream (the previous target came less than STREAM_GAP_S ago)
        the segment lasts the stream's own interval and ends moving at the
        targets' speed, so the arm keeps pace instead of stopping at each
        one. A lone target is reached from rest to rest, in `dt_target` or
        as fast as the limits allow. Either way a move too fast for the
        joint stretches the segment instead of being clipped: the arm then
        trails a fast-moving target rather than snapping to it.

        The same target again -- another topic of the same command naming
        this joint too -- changes nothing.
        """
        target_q = float(target_q)
        if self.segment is not None and abs(target_q - self.target_q) < 1e-9:
            return
        elapsed = self.since_target
        if elapsed <= STREAM_GAP_S:
            duration = max(0.005, elapsed)
            end_v = (target_q - self.target_q) / duration
            end_v = max(-self.v_max, min(self.v_max, end_v))
            # Speeding up or slowing down as the stream does: ending each
            # segment at zero acceleration would make it swing in between.
            end_a = (end_v - self.stream_v) / duration
            end_a = max(-self.a_max, min(self.a_max, end_a))
        else:
            duration = max(0.005, float(dt_target))
            end_v = end_a = 0.0
            diff = abs(target_q - self.current_q)
            duration = max(duration, diff / max(self.v_max, 1e-3),
                           math.sqrt(2.0 * diff / max(self.a_max, 1e-3)))
        self.target_q = target_q
        self.since_target = 0.0
        self.stream_v = end_v
        self._plan(target_q, end_v, duration, end_a,
                   longest=STREAM_STRETCH * duration if elapsed <= STREAM_GAP_S
                   else None)

    def _plan(self, target_q, end_v, duration, end_a=0.0, longest=None):
        """A segment to (target_q, end_v, end_a), `duration` long or
        stretched until its speed and acceleration stay within the limits
        -- for a stream's acceleration, to `longest` at most."""
        planned = duration
        for _ in range(16):
            # Arriving faster than twice the segment's average speed, or
            # moving back the way it came, makes the curve swing out and
            # back -- what a stretched segment chasing a lagging stream
            # would do. Bound the end speed by the distance to cover.
            average = (target_q - self.current_q) / planned
            if end_v * average <= 0.0:
                arrive, arrive_a = 0.0, 0.0
            else:
                arrive = math.copysign(min(abs(end_v), 2.0 * abs(average)), end_v)
                arrive_a = end_a if planned <= duration else 0.0
            segment = QuinticSegment(
                self.current_q, self.current_v, self.current_a,
                target_q, arrive, arrive_a, planned)
            speed_ok, accel_ok = self._within_limits(segment)
            if speed_ok and (accel_ok or (longest is not None and planned >= longest)):
                break
            grown = planned * 1.25
            if speed_ok and longest is not None:
                grown = min(longest, grown)
            planned = grown
        self.segment = segment
        self.segment_time = 0.0
        self.segment_duration = planned

    def _within_limits(self, segment):
        """-> (speed within v_max, acceleration within a_max). A little
        slack: the segment starts from the state the last one left, which
        the clamps in step() may already hold at a limit."""
        speed_ok = accel_ok = True
        for index in range(1, LIMIT_SAMPLES + 1):
            _, vel, acc = segment.evaluate(segment.T * index / LIMIT_SAMPLES)
            speed_ok = speed_ok and abs(vel) <= self.v_max * 1.02
            accel_ok = accel_ok and abs(acc) <= self.a_max * 1.05
        return speed_ok, accel_ok

    def _brake(self):
        """The stream stopped with the joint moving: come to rest within
        a_max, at most BRAKE_OVERSHOOT past where it is."""
        speed = abs(self.current_v)
        duration = max(0.02, 1.875 * speed / max(self.a_max, 1e-3))
        run = min(BRAKE_OVERSHOOT, 0.5 * speed * duration)
        duration = max(0.02, 2.0 * run / max(speed, 1e-6))
        stop = self.current_q + math.copysign(run, self.current_v)
        self.target_q = stop
        self.segment = QuinticSegment(
            self.current_q, self.current_v, self.current_a, stop, 0.0, 0.0,
            duration)
        self.segment_time = 0.0
        self.segment_duration = duration

    @property
    def settled(self):
        """True once the planned segment has run to its end.

        The control loop asks this before it stops commanding: a motion must
        finish even when no further target arrives, or it is abandoned
        wherever the stream happened to stop.
        """
        return self.segment is None or self.segment_time >= self.segment_duration

    def step(self, dt):
        """Advance by `dt` and return the (position, velocity) to command."""
        self.since_target += dt
        if self.segment is None:
            if abs(self.target_q - self.current_q) < 1e-6:
                self.current_q = self.target_q
                self.current_v = 0.0
                self.current_a = 0.0
                return self.current_q, 0.0
            self.update_target(self.target_q)

        self.segment_time += dt
        pos, vel, acc = self.segment.evaluate(self.segment_time)

        if abs(vel) > self.v_max:
            vel = math.copysign(self.v_max, vel)
        if abs(acc) > self.a_max:
            acc = math.copysign(self.a_max, acc)

        self.current_q = pos
        self.current_v = vel
        self.current_a = acc

        if self.segment_time >= self.segment_duration:
            if abs(vel) > 1e-6:
                # Ran off the end of a stream's segment still moving: the
                # next target is late or not coming. Brake, do not stop dead.
                self._brake()
            else:
                # Ended at rest: settled on the target.
                self.current_q = self.target_q
                self.current_v = 0.0
                self.current_a = 0.0

        return self.current_q, self.current_v


# How targets become the 400 Hz setpoint: this module's quintic segments,
# or filter4.py's critically damped fourth-order filter.
KINDS = ("quintic", "filter4")


class BimanualInterpolator:
    """A JointInterpolator (or FourthOrderFilter) per motor, by motor id."""

    def __init__(self, motor_ids, initial_positions=None, kind="quintic",
                 w=DEFAULT_BANDWIDTH):
        if kind not in KINDS:
            raise ValueError(f"interpolator must be one of {KINDS}, not {kind!r}")
        self.kind = kind
        self.w = float(w)
        self.joints = {}
        for motor_id in motor_ids:
            initial = (initial_positions or {}).get(motor_id, 0.0)
            self.joints[motor_id] = self._make(motor_id, initial)

    def _make(self, motor_id, initial):
        joint_idx = ((motor_id - 1) % 8) + 1
        v_max, a_max = KINEMATIC_LIMITS.get(joint_idx, (2.0, 6.0))
        if self.kind == "filter4":
            return FourthOrderFilter(v_max, a_max, initial, self.w)
        return JointInterpolator(v_max, a_max, initial)

    def set_kind(self, kind):
        """Switch every joint to the other interpolator where it stands:
        same position, speed and target, so the setpoint does not jump."""
        if kind not in KINDS:
            raise ValueError(f"interpolator must be one of {KINDS}, not {kind!r}")
        if kind == self.kind:
            return
        self.kind = kind
        for motor_id, old in list(self.joints.items()):
            new = self._make(motor_id, old.current_q)
            new.set_state(old.current_q, old.current_v)
            new.update_target(old.target_q)
            self.joints[motor_id] = new

    def set_bandwidth(self, w):
        """filter4's w, while running; the quintic layer has none."""
        self.w = float(w)
        for joint in self.joints.values():
            if isinstance(joint, FourthOrderFilter):
                joint.w = self.w

    def set_state(self, motor_id, q, v=0.0):
        joint = self.joints.get(motor_id)
        if joint is not None:
            joint.set_state(q, v)

    def update_target(self, motor_id, target_q, dt_target=0.02):
        joint = self.joints.get(motor_id)
        if joint is not None:
            joint.update_target(target_q, dt_target)

    def step(self, motor_id, dt):
        joint = self.joints.get(motor_id)
        if joint is None:
            return 0.0, 0.0
        return joint.step(dt)

    def settled(self):
        """True once every joint has finished the segment it was given."""
        return all(joint.settled for joint in self.joints.values())
