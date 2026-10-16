#!/usr/bin/env python3
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

"""A camera that is not there: black RGB-D frames on a real camera's topics.

camera.launch.py starts one for a camera whose RealSense is not plugged in,
in place of its driver and preprocessor, so the recorder, the dashboard and
the policies run as with the real camera -- in the simulator, say:

    /openarm/camera/<name>/rgb     rgb8, every pixel black
    /openarm/camera/<name>/depth   16UC1, every pixel 0 (no reading)

at the preprocessor's size and rate, stamped the way it stamps, with
frame_id <name>_virtual so whoever reads a frame can tell (cameras.is_virtual).

    ros2 run openarm_camera virtual_camera --name front
"""

import argparse
import sys

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
except ImportError:
    print("[virtual camera] ROS 2 not found.")
    sys.exit(0)

from openarm_camera import cameras


def black_frames(width, height):
    """(rgb, depth) Image messages, all zeros, without a header yet."""
    rgb = Image(height=height, width=width, encoding="rgb8", is_bigendian=0,
                step=width * 3, data=bytes(width * height * 3))
    depth = Image(height=height, width=width, encoding="16UC1", is_bigendian=0,
                  step=width * 2, data=bytes(width * height * 2))
    return rgb, depth


class VirtualCamera(Node):
    def __init__(self, args):
        super().__init__("openarm_virtual_camera")
        self.name = args.name
        self.frame_id = f"{args.name}{cameras.VIRTUAL_SUFFIX}"
        self.period_ns = round(1e9 / args.rate)
        # The same two messages every time: only their stamp changes.
        self.rgb, self.depth = black_frames(args.width, args.height)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE)
        self.rgb_pub = self.create_publisher(Image, args.rgb_output, output_qos)
        self.depth_pub = self.create_publisher(Image, args.depth_output, output_qos)
        self.create_timer(1.0 / args.rate, self._publish)
        self.get_logger().warning(
            f"{self.name}: no RealSense -- a virtual camera, black "
            f"{args.width}x{args.height} @ {args.rate:g} Hz on "
            f"{args.rgb_output} + {args.depth_output}")

    def _publish(self):
        # On the output-rate slot, as the preprocessor stamps real frames.
        now_ns = self.get_clock().now().nanoseconds
        slot = ((now_ns + self.period_ns // 2) // self.period_ns) * self.period_ns
        stamp = Time(sec=slot // 1_000_000_000, nanosec=slot % 1_000_000_000)
        for msg, publisher in ((self.rgb, self.rgb_pub), (self.depth, self.depth_pub)):
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="chest", choices=cameras.CAMERAS)
    parser.add_argument("--rgb-output", default=None)
    parser.add_argument("--depth-output", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--rate", type=float, default=25.0)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    if args.width <= 0 or args.height <= 0 or args.rate <= 0:
        parser.error("width, height, and rate must be positive")
    args.rgb_output = args.rgb_output or cameras.rgb_topic(args.name)
    args.depth_output = args.depth_output or cameras.depth_topic(args.name)

    rclpy.init()
    node = VirtualCamera(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
