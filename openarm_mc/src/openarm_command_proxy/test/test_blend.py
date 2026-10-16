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

"""Switching sources eases from the measured pose, and only when needed."""

import pytest

from openarm_command_proxy.blend import SwitchBlend, hold_pose, smoothstep

ARM = [1, 2, 3]
GRIPPER = 8


def test_a_source_that_starts_on_the_robot_passes_straight_through():
    blend = SwitchBlend(ARM)
    blend.switched()
    measured = {1: 0.5, 2: -0.2, 3: 1.0}
    targets = {1: 0.505, 2: -0.2, 3: 1.0}       # 5 mrad off: within threshold
    assert blend.output(targets, measured, now=0.0) == targets
    assert not blend.active


def test_a_far_source_starts_at_the_measured_pose_and_arrives_smoothly():
    blend = SwitchBlend(ARM, speed=0.5, min_duration=0.5)
    blend.switched()
    measured = {1: 0.0, 2: 0.0, 3: 0.0}
    targets = {1: 1.0, 2: -0.5, 3: 0.0}         # 1 rad away: 2 s at 0.5 rad/s
    first = blend.output(targets, measured, now=10.0)
    assert first == pytest.approx(measured)     # no step at all
    assert blend.duration == pytest.approx(2.0)
    half = blend.output(targets, measured, now=11.0)
    assert half[1] == pytest.approx(0.5) and half[2] == pytest.approx(-0.25)
    done = blend.output(targets, measured, now=12.0)
    assert done == pytest.approx(targets)
    assert not blend.active
    # Afterwards the source passes through untouched.
    assert blend.output({1: 2.0, 2: 0.0, 3: 0.0}, measured, now=12.1)[1] == 2.0


def test_a_target_that_moves_meanwhile_is_followed():
    blend = SwitchBlend(ARM)
    blend.switched()
    measured = {1: 0.0, 2: 0.0, 3: 0.0}
    blend.output({1: 1.0, 2: 0.0, 3: 0.0}, measured, now=0.0)
    out = blend.output({1: 1.4, 2: 0.0, 3: 0.0}, measured, now=blend.duration / 2)
    assert out[1] == pytest.approx(0.7)         # halfway to where it is now


def test_a_gripper_joining_later_eases_from_its_own_measurement():
    blend = SwitchBlend(ARM)
    blend.switched()
    measured = {1: 0.0, 2: 0.0, 3: 0.0, GRIPPER: 0.01}
    blend.output({1: 1.0, 2: 0.0, 3: 0.0}, measured, now=0.0)
    out = blend.output({1: 1.0, 2: 0.0, 3: 0.0, GRIPPER: 0.04}, measured,
                       now=blend.duration / 2)
    assert out[GRIPPER] == pytest.approx(0.025)


def test_only_a_switch_arms_the_easing():
    blend = SwitchBlend(ARM)
    measured = {1: 0.0, 2: 0.0, 3: 0.0}
    far = {1: 2.0, 2: 0.0, 3: 0.0}
    assert blend.output(far, measured, now=0.0) == far


def test_smoothstep_ends_at_rest():
    assert smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    h = 1e-4
    assert (smoothstep(h) - smoothstep(0.0)) / h < 1e-3
    assert (smoothstep(1.0) - smoothstep(1.0 - h)) / h < 1e-3


def test_held_pose_is_the_controllers_setpoint_while_the_motors_hold_it():
    measured = {1: 0.48, 2: -0.2, 8: 0.01}      # sagging under gravity
    setpoint = {1: 0.5, 2: -0.2}
    assert hold_pose(measured, setpoint, "holding", 0.01) == \
        {1: 0.5, 2: -0.2, 8: 0.01}
    assert hold_pose(measured, setpoint, "commanding", 0.01)[1] == 0.5
    # Motors off, nothing sent yet, or a controller gone quiet: measured.
    for state, age in (("disabled", 0.01), ("unsent", 0.01), ("holding", 1.0)):
        assert hold_pose(measured, setpoint, state, age) == measured
    assert hold_pose(measured) == measured


def test_a_switch_onto_the_setpoint_does_not_move_the_arm():
    blend = SwitchBlend(ARM)
    blend.switched()
    held = hold_pose({1: 0.47, 2: 0.0, 3: 0.0}, {1: 0.5, 2: 0.0, 3: 0.0},
                     "holding", 0.01)
    # A source that starts where the controller holds the arm: no easing
    # from the sagged measured pose, so nothing drops.
    targets = {1: 0.5, 2: 0.0, 3: 0.0}
    assert blend.output(targets, held, now=0.0) == targets
    assert not blend.active
