# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest

from openarm_ui.zero_pose import (
    FINISHING, IDLE, MOVING, STARTING, ZeroPoseMotion, ZeroPoseState)


def test_motion_starts_on_the_arm_and_ends_at_zero():
    origin = {1: 1.0, 2: -0.5, 8: 0.03}     # 8, a gripper, has no goal
    motion = ZeroPoseMotion(origin, {1: 0.0, 2: 0.0}, start=10.0,
                            speed=0.5, min_duration=1.0, arm_ids=[1, 2])
    assert motion.duration == pytest.approx(2.0)    # 1 rad at 0.5 rad/s
    first, done = motion.sample(10.0)
    assert first == pytest.approx(origin) and not done
    middle, _ = motion.sample(11.0)
    assert middle[1] == pytest.approx(0.5) and middle[8] == pytest.approx(0.03)
    last, done = motion.sample(12.5)
    assert done and last[1] == 0.0 and last[2] == 0.0 and last[8] == 0.03


def test_a_short_motion_still_takes_min_duration():
    motion = ZeroPoseMotion({1: 0.01}, {1: 0.0}, start=0.0, speed=0.5,
                            min_duration=1.5)
    assert motion.duration == pytest.approx(1.5)


def test_one_claim_at_a_time():
    state = ZeroPoseState()
    token, problem = state.claim()
    assert problem is None and state.phase == STARTING and state.busy
    again, problem = state.claim()
    assert again is None and "already" in problem


def test_a_cancelled_claim_cannot_start():
    state = ZeroPoseState()
    token, _ = state.claim()
    assert state.finish(None, False, "zero pose stopped: teleop selected")
    assert state.phase == IDLE
    motion = ZeroPoseMotion({1: 1.0}, {1: 0.0}, start=0.0)
    assert state.begin(token, motion, "none") == "the zero pose was cancelled"
    assert state.phase == IDLE


def test_a_late_step_cannot_end_a_newer_claim():
    state = ZeroPoseState()
    old, _ = state.claim()
    state.finish(old, False, "cancelled")
    new, _ = state.claim()
    assert not state.finish(old, True, "late")
    assert state.phase == STARTING and state.token == new


def test_arrival_then_finish_reports_once():
    state = ZeroPoseState()
    token, _ = state.claim()
    motion = ZeroPoseMotion({1: 1.0}, {1: 0.0}, start=0.0)
    assert state.begin(token, motion, "none") is None
    assert state.phase == MOVING and state.restore == "none"
    assert state.arrived() == token and state.phase == FINISHING
    assert state.busy           # sliders stay refused while handing back
    assert state.finish(token, True, "at the zero pose")
    assert not state.finish(token, True, "again")
    snapshot = state.snapshot(0.0)
    assert snapshot["phase"] == IDLE
    assert snapshot["last"] == {"id": 1, "ok": True,
                                "message": "at the zero pose"}
