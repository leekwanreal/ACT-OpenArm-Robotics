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

"""What the camera view shows for a given frame."""

import cv2
import numpy as np

from openarm_ui.camera import colorize_depth, encode_depth, encode_rgb


def test_rgb_survives_the_round_trip_in_rgb_order():
    image = np.zeros((48, 64, 3), np.uint8)
    image[..., 0] = 220                         # red in RGB order
    decoded = cv2.imdecode(np.frombuffer(encode_rgb(image, 95), np.uint8),
                           cv2.IMREAD_COLOR)    # BGR
    assert decoded.shape == (48, 64, 3)
    assert decoded[..., 2].mean() > 200 and decoded[..., 0].mean() < 30


def test_depth_colour_is_fixed_by_range_not_by_frame():
    near = np.full((4, 4), 200, np.uint16)
    far = np.full((4, 4), 1200, np.uint16)
    both = np.hstack([near, far])
    # The same distance gets the same colour whatever else is in the frame.
    assert np.array_equal(colorize_depth(near)[0, 0], colorize_depth(both)[0, 0])
    assert np.array_equal(colorize_depth(far)[0, 0], colorize_depth(both)[0, -1])
    # Near is red, far is blue (BGR).
    assert colorize_depth(near)[0, 0, 2] > colorize_depth(near)[0, 0, 0]
    assert colorize_depth(far)[0, 0, 0] > colorize_depth(far)[0, 0, 2]


def test_no_reading_is_black_and_metres_are_understood():
    depth = np.array([[0.0, 0.2, np.nan]], np.float32)       # metres
    colored = colorize_depth(depth)
    assert not colored[0, 0].any() and not colored[0, 2].any()
    assert np.array_equal(colored[0, 1],
                          colorize_depth(np.array([[200]], np.uint16))[0, 0])
    assert encode_depth(depth)[:2] == b"\xff\xd8"           # a JPEG
