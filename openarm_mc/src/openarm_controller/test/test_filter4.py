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

"""What `interpolator: filter4` guarantees, and switching to and from it."""

import pytest

from openarm_controller.filter4 import DEFAULT_BANDWIDTH, FourthOrderFilter
from openarm_controller.spline import BimanualInterpolator, JointInterpolator

V_MAX, A_MAX = 1.2, 3.5          # J1/J2 limits
CONTROL_DT = 1.0 / 400.0
COMMAND_DT = 0.02                # 50 Hz, an ACT policy's rate
TICKS = int(round(COMMAND_DT / CONTROL_DT))


def response(step, v_max=V_MAX, seconds=2.0, w=DEFAULT_BANDWIDTH):
    joint = FourthOrderFilter(v_max, A_MAX, 0.0, w)
    joint.update_target(step)
    return [joint.step(CONTROL_DT) for _ in range(int(seconds / CONTROL_DT))]


def test_a_step_settles_without_overshoot():
    for step in (0.05, 0.3, -1.0):
        samples = response(step)
        reach = [q * (1 if step > 0 else -1) for q, _ in samples]
        assert max(reach) <= abs(step) + 1e-9
        assert abs(samples[-1][0] - step) < 1e-4


def test_a_small_step_takes_the_filters_own_time():
    # Four poles at -w: 95 % of a step in about 7.75 / w.
    samples = response(0.05)
    t95 = next(i for i, (q, _) in enumerate(samples) if q >= 0.95 * 0.05) * CONTROL_DT
    assert abs(t95 - 7.75 / DEFAULT_BANDWIDTH) < 0.02


def test_velocity_is_what_the_position_does():
    # The MIT frame gets both: they must agree, or the PD fights itself.
    # Semi-implicit Euler: each tick moves the position by the new velocity.
    samples = response(0.4)
    for (q0, _), (q1, v1) in zip(samples, samples[1:]):
        assert abs((q1 - q0) / CONTROL_DT - v1) < 1e-9


def test_a_wide_step_is_crossed_at_the_joints_speed():
    samples = response(1.0, seconds=3.0)
    assert max(abs(v) for _, v in samples) <= V_MAX + 1e-9
    assert abs(samples[-1][0] - 1.0) < 1e-4


def test_a_smooth_stream_is_followed_about_4_over_w_behind():
    joint = FourthOrderFilter(V_MAX, A_MAX, 0.0)
    target, lags = 0.0, []
    for command in range(100):               # 2 s of a 0.5 rad/s ramp at 50 Hz
        target += 0.5 * COMMAND_DT
        joint.update_target(target, COMMAND_DT)
        for _ in range(TICKS):
            q, _ = joint.step(CONTROL_DT)
        if command >= 50:
            lags.append((target - q) / 0.5)
    # Sampled at the end of each command period, half a period after the
    # target's average: the filter's own lag is 4 / w.
    lag = sum(lags) / len(lags) + COMMAND_DT / 2
    assert abs(lag - 4.0 / DEFAULT_BANDWIDTH) < 0.005


def test_settles_and_rebases_like_the_quintic_layer():
    joint = FourthOrderFilter(V_MAX, A_MAX, 0.0)
    assert joint.settled
    joint.update_target(-1.2)
    assert not joint.settled
    for _ in range(int(0.5 / CONTROL_DT)):     # a typical command timeout
        joint.step(CONTROL_DT)
    assert not joint.settled
    for _ in range(int(2.0 / CONTROL_DT)):
        joint.step(CONTROL_DT)
    assert joint.settled
    joint.set_state(1.0)
    assert joint.step(CONTROL_DT) == (1.0, 0.0)


def test_quintic_stays_the_default():
    interp = BimanualInterpolator(range(1, 17))
    assert interp.kind == "quintic"
    assert all(isinstance(j, JointInterpolator) for j in interp.joints.values())


def test_switching_mid_move_does_not_jump():
    for start, end in (("quintic", "filter4"), ("filter4", "quintic")):
        interp = BimanualInterpolator([1], kind=start)
        interp.update_target(1, 0.6, COMMAND_DT)
        for _ in range(40):
            q_before, v_before = interp.step(1, CONTROL_DT)
        interp.set_kind(end)
        q_after, v_after = interp.step(1, CONTROL_DT)
        assert abs(q_after - q_before) < V_MAX * CONTROL_DT * 1.5
        for _ in range(int(3.0 / CONTROL_DT)):
            q, _ = interp.step(1, CONTROL_DT)
        assert abs(q - 0.6) < 1e-3                  # the target came along


def test_an_unknown_interpolator_is_refused():
    with pytest.raises(ValueError):
        BimanualInterpolator([1], kind="cubic")
    interp = BimanualInterpolator([1])
    with pytest.raises(ValueError):
        interp.set_kind("cubic")
