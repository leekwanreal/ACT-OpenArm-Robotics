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

"""What the quintic layer guarantees between sparse commands and the CAN loop."""

from openarm_controller import damiao
from openarm_controller.spline import JointInterpolator

V_MAX, A_MAX = 1.2, 3.5          # J1/J2 limits
CONTROL_DT = 1.0 / 400.0
COMMAND_DT = 0.02                # 50 Hz, an ACT policy's rate
TICKS = int(round(COMMAND_DT / CONTROL_DT))


def stream(delta, commands=50, initial=0.0):
    """Feed `commands` targets `delta` apart and sample the 400 Hz output."""
    interp = JointInterpolator(V_MAX, A_MAX, initial)
    positions, velocities = [], []
    target = initial
    for _ in range(commands):
        target += delta
        interp.update_target(target, COMMAND_DT)
        for _ in range(TICKS):
            q, v = interp.step(CONTROL_DT)
            positions.append(q)
            velocities.append(v)
    return positions, velocities, target, interp


def test_output_is_far_smoother_than_the_raw_command():
    delta = 0.02
    positions, _, _, _ = stream(delta)
    biggest = max(abs(b - a) for a, b in zip(positions, positions[1:]))
    # A raw command would step the whole delta in one tick.
    assert biggest < delta / 4


def test_velocity_stays_within_the_joint_limit():
    _, velocities, _, _ = stream(0.02)
    assert max(abs(v) for v in velocities) <= V_MAX + 1e-6


def test_an_impossible_command_rate_is_slewed_not_followed():
    # 0.3 rad every 20 ms is 15 rad/s, far past V_MAX: the joint must refuse
    # to track it rather than command a jump.
    _, velocities, _, _ = stream(0.3)
    assert max(abs(v) for v in velocities) <= V_MAX + 1e-6


def test_a_settled_stream_reaches_its_target():
    # One target, then enough ticks to finish the planned segment.
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    interp.update_target(0.1, COMMAND_DT)
    for _ in range(400):
        q, _ = interp.step(CONTROL_DT)
    assert abs(q - 0.1) < 1e-9


def test_a_slow_publisher_still_moves_smoothly():
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    interp.update_target(0.5, 0.2)       # one target, 5 Hz
    samples = [interp.step(CONTROL_DT)[0] for _ in range(80)]
    assert max(abs(b - a) for a, b in zip(samples, samples[1:])) < 0.01
    assert max(samples) <= 0.5 + 1e-6    # no overshoot


def test_set_state_rebases_without_a_jump():
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    interp.set_state(1.0)
    q, v = interp.step(CONTROL_DT)
    assert q == 1.0 and v == 0.0


def test_gripper_stroke_round_trips():
    for stroke in (0.0, 0.02, damiao.GRIPPER_MAX_STROKE_M):
        rad = damiao.gripper_stroke_to_rad(stroke)
        assert abs(damiao.gripper_rad_to_stroke(rad) - stroke) < 1e-9


def test_mit_frame_round_trips():
    p_max, v_max, t_max = 12.5, 45.0, 54.0
    data = damiao.encode_mit(0.8, 1.5, 18.0, 2.0, 0.0, p_max, v_max, t_max)
    assert len(data) == 8
    q_uint = (data[0] << 8) | data[1]
    assert abs(damiao.uint_to_double(q_uint, -p_max, p_max, 16) - 0.8) < 1e-3


def test_state_frame_decodes_what_a_motor_sends():
    p_max, v_max, t_max = 12.5, 45.0, 54.0
    q, dq, tau = 0.6137, -1.234, 3.75
    # Build the frame the way the motor firmware does.
    q_uint = damiao.double_to_uint(q, -p_max, p_max, 16)
    dq_uint = damiao.double_to_uint(dq, -v_max, v_max, 12)
    tau_uint = damiao.double_to_uint(tau, -t_max, t_max, 12)
    frame = bytes([
        (1 << 4) | 0x01,
        (q_uint >> 8) & 0xFF, q_uint & 0xFF,
        (dq_uint >> 4) & 0xFF,
        ((dq_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F),
        tau_uint & 0xFF, 34, 31,
    ])
    got = damiao.decode_state(frame, p_max, v_max, t_max)
    assert abs(got["q"] - q) < 1e-3
    assert abs(got["dq"] - dq) < 2e-2
    assert abs(got["tau"] - tau) < 5e-2
    assert got["error_code"] == 1


def test_two_bus_hardware_and_shared_bus_simulator_addressing():
    hardware = damiao.build_motors("can1", "can0")
    assert hardware[1].send_id == 0x01 and hardware[1].interface == "can1"
    # Real arms sit on separate buses, so both use the same ids.
    assert hardware[9].send_id == 0x01 and hardware[9].interface == "can0"

    shared = damiao.build_motors("vcan0", "vcan0", right_id_offset=0x20)
    assert shared[1].send_id == 0x01
    assert shared[9].send_id == 0x21 and shared[9].recv_id == 0x31


def test_a_segment_is_not_settled_until_it_has_run_out():
    # The control loop asks `settled` before it stops commanding. A wide
    # swing takes far longer than any command timeout, so saying "settled"
    # early is what abandons the move half way.
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    assert interp.settled                      # nothing planned yet

    interp.update_target(-1.2, COMMAND_DT)
    assert not interp.settled

    # Half a second in -- a typical command timeout -- it is still moving.
    for _ in range(int(0.5 / CONTROL_DT)):
        interp.step(CONTROL_DT)
    assert not interp.settled

    for _ in range(int(1.5 / CONTROL_DT)):
        interp.step(CONTROL_DT)
    assert interp.settled
    assert abs(interp.current_q - (-1.2)) < 1e-9


def test_a_wide_swing_outlasts_the_default_command_timeout():
    # Why the loop cannot use "no command for 0.5 s" alone as its stop rule.
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    interp.update_target(-1.2, COMMAND_DT)
    assert interp.segment_duration > 0.5


def test_a_segment_ends_where_it_was_planned_to_while_accelerating():
    # The port's c5 had the acceleration term's sign flipped: a segment
    # planned mid-acceleration ended off its target, and the setpoint was
    # then snapped onto it -- a step at every re-plan.
    from openarm_controller.spline import QuinticSegment
    segment = QuinticSegment(0.0, 0.5, 3.0, 0.02, 0.7, 1.0, 0.05)
    q, v, a = segment.evaluate(0.05)
    assert abs(q - 0.02) < 1e-9 and abs(v - 0.7) < 1e-9 and abs(a - 1.0) < 1e-9


def ramp(speed, hz, seconds, interp=None):
    """A leader moving at `speed` from rest, `hz` targets a second."""
    interp = interp or JointInterpolator(V_MAX, A_MAX, 0.0)
    ticks = int(round(1.0 / hz / CONTROL_DT))
    samples, target = [], 0.0
    for index in range(int(seconds * hz)):
        # Ease into the speed over 0.3 s, as an arm does.
        target += speed * min(1.0, index / hz / 0.3) / hz
        interp.update_target(target, 1.0 / hz)
        samples += [interp.step(CONTROL_DT) for _ in range(ticks)]
    return samples, target, interp


def test_a_stream_is_followed_without_falling_behind():
    # A leader arm at 0.8 rad/s, 50 Hz: the shoulder keeps within about one
    # interval of it, at its speed, instead of stopping at every target.
    samples, target, interp = ramp(0.8, 50, 2.0)
    lag_s = (target - interp.current_q) / 0.8
    assert lag_s < 0.04
    assert abs(samples[-1][1] - 0.8) < 0.1


def test_the_other_topics_of_one_command_do_not_replan():
    interp = JointInterpolator(V_MAX, A_MAX, 0.0)
    interp.update_target(0.1, COMMAND_DT)
    segment = interp.segment
    interp.update_target(0.1, 0.005)     # the gripper's message, same target
    assert interp.segment is segment


def test_a_stream_that_stops_mid_motion_brakes_smoothly():
    from openarm_controller.spline import BRAKE_OVERSHOOT
    samples, target, interp = ramp(0.8, 50, 1.0)
    rest = [interp.step(CONTROL_DT) for _ in range(400)]
    positions = [q for q, _ in samples + rest]
    steps = [abs(b - a) for a, b in zip(positions, positions[1:])]
    assert max(steps) < V_MAX * CONTROL_DT * 1.05      # no jump
    assert abs(rest[-1][1]) < 1e-9                      # at rest
    assert rest[-1][0] - target <= BRAKE_OVERSHOOT + 1e-9
    assert interp.settled
