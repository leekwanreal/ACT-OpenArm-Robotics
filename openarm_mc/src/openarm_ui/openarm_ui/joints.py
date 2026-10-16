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

"""Joint order and motor ids, as openarm_can numbers them."""

JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]

# The fourteen the arm topic carries, in its order: left j1..j7 then right.
ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
LEFT_GRIPPER_ID = 8
RIGHT_GRIPPER_ID = 16

JOINT_NAME_TO_ID = {name: index + 1 for index, name in enumerate(JOINT_NAMES)}
