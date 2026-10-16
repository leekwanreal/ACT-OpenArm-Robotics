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

"""The joint ordering this proxy has to get right to drive openarm_controller."""

from openarm_command_proxy.command_proxy import (
    ARM_JOINT_IDS,
    JOINT_NAMES,
    LEFT_GRIPPER_ID,
    RIGHT_GRIPPER_ID,
)


def names_for(motor_ids):
    """What _publish puts in JointState.name for these motor ids."""
    return [JOINT_NAMES[motor_id - 1] for motor_id in sorted(motor_ids)]


def test_the_fourteen_arm_values_land_on_the_arm_joints():
    # A source sends left j1..j7 then right j1..j7, and nothing else.
    assert len(ARM_JOINT_IDS) == 14
    assert names_for(ARM_JOINT_IDS) == [
        "left_j1", "left_j2", "left_j3", "left_j4",
        "left_j5", "left_j6", "left_j7",
        "right_j1", "right_j2", "right_j3", "right_j4",
        "right_j5", "right_j6", "right_j7",
    ]


def test_grippers_are_the_two_ids_the_arm_array_skips():
    assert LEFT_GRIPPER_ID not in ARM_JOINT_IDS
    assert RIGHT_GRIPPER_ID not in ARM_JOINT_IDS
    assert JOINT_NAMES[LEFT_GRIPPER_ID - 1] == "left_gripper"
    assert JOINT_NAMES[RIGHT_GRIPPER_ID - 1] == "right_gripper"
    # Together they account for every joint, exactly once.
    assert sorted(ARM_JOINT_IDS + [LEFT_GRIPPER_ID, RIGHT_GRIPPER_ID]) == list(
        range(1, 17))


def test_arm_values_keep_their_order_through_the_merge():
    # The i-th value of the array must reach the i-th arm joint.
    data = [float(i) for i in range(14)]
    targets = dict(zip(ARM_JOINT_IDS, data))
    motor_ids = sorted(targets)
    positions = [targets[motor_id] for motor_id in motor_ids]
    assert names_for(motor_ids)[0] == "left_j1" and positions[0] == 0.0
    assert names_for(motor_ids)[6] == "left_j7" and positions[6] == 6.0
    assert names_for(motor_ids)[7] == "right_j1" and positions[7] == 7.0
    assert names_for(motor_ids)[13] == "right_j7" and positions[13] == 13.0


def test_a_gripper_sorts_into_place_rather_than_onto_the_end():
    # left_gripper is motor 8, so it belongs between left_j7 and right_j1.
    targets = dict.fromkeys(ARM_JOINT_IDS, 0.0)
    targets[LEFT_GRIPPER_ID] = 0.04
    names = names_for(targets)
    assert names.index("left_gripper") == names.index("left_j7") + 1
    assert names.index("left_gripper") == names.index("right_j1") - 1


def test_partial_commands_name_only_what_arrived():
    # Before any gripper message, the command carries the 14 arm joints only.
    names = names_for(ARM_JOINT_IDS)
    assert "left_gripper" not in names
    assert "right_gripper" not in names
    assert len(names) == 14
