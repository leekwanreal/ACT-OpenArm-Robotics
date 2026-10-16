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

"""Turn camera frames into JPEGs a browser can show.

The page gets each camera's RGB-D pair as openarm_camera prepares it, which
is what the policies and the recorder see, so what is on screen is what they
get. A stream is "<camera>/<kind>": chest/rgb, front/depth.
Depth is coloured over the recorder's working range rather than scaled to
whatever the frame holds, so the same colour means the same distance from one
frame to the next; pixels with no reading are black.
"""

import cv2
import numpy as np

KINDS = ("rgb", "depth")


def kind_of(stream):
    """"front/depth" -> "depth"."""
    return str(stream).rsplit("/", 1)[-1]

_bridge = None


def encode_message(stream, msg, quality=70, min_mm=200, max_mm=1200):
    """A sensor_msgs/Image from `stream` -> (JPEG bytes, width, height)."""
    global _bridge
    if _bridge is None:
        from cv_bridge import CvBridge
        _bridge = CvBridge()
    if kind_of(stream) == "rgb":
        image = _bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        jpeg = encode_rgb(image, quality)
    else:
        image = _bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        jpeg = encode_depth(image, min_mm, max_mm, quality=quality)
    return jpeg, int(msg.width), int(msg.height)


def encode_rgb(image, quality=70):
    """RGB uint8 [H, W, 3] -> JPEG bytes."""
    bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return _jpeg(bgr, quality)


def depth_to_mm(depth, depth_scale=0.001):
    """A depth image in whatever unit it came in -> float millimetres."""
    if depth.dtype in (np.float32, np.float64):
        return np.nan_to_num(depth, nan=0.0).astype(np.float32) / depth_scale
    return depth.astype(np.float32)


def colorize_depth(depth, min_mm=200, max_mm=1200, depth_scale=0.001):
    """Depth [H, W] -> BGR uint8, near warm and far cold, no reading black."""
    depth_mm = depth_to_mm(np.asarray(depth), depth_scale)
    span = max(1.0, float(max_mm - min_mm))
    near = np.clip((depth_mm - min_mm) / span, 0.0, 1.0)
    # TURBO runs blue to red; flip it so near is red.
    shade = ((1.0 - near) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
    colored[depth_mm <= 0] = 0
    return colored


def encode_depth(depth, min_mm=200, max_mm=1200, depth_scale=0.001,
                 quality=70):
    """Depth [H, W] -> JPEG bytes of its colour map."""
    return _jpeg(colorize_depth(depth, min_mm, max_mm, depth_scale), quality)


def _jpeg(bgr, quality):
    ok, buffer = cv2.imencode(".jpg", bgr,
                              [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()
