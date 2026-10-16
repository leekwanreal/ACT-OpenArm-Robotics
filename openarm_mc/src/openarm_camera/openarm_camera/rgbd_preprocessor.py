#!/usr/bin/env python3
"""Synchronize, resize, and rate-limit one RealSense camera's RGB-D frames.

One per camera (camera.launch.py): the driver's colour and aligned depth in,
an RGB-D pair out at the dataset's size and rate, stamped on the output
clock, for the recorder, the dashboard and the policies.

Its `source` parameter is the camera whose driver it reads, its own role
at first. Setting it to the other role swaps the two cameras while they
run, with no driver restarted -- setup_camera.sh switch and the dashboard
do it for both preprocessors at once, when chest and front were assigned
the wrong way round:

    ros2 param set /openarm_chest_preprocessor source front
"""

import argparse
import sys
import threading

import cv2
try:
    import message_filters
    import rclpy
    from builtin_interfaces.msg import Time
    from cv_bridge import CvBridge
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import ExternalShutdownException
    from rclpy.executors import MultiThreadedExecutor
    from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
except ImportError:
    import sys
    print("[RGB-D Preprocessor] ROS 2 dependencies not found. Preprocessor disabled.")
    sys.exit(0)

from openarm_camera import cameras


class RGBDPreprocessor(Node):
    def __init__(self, args):
        super().__init__("openarm_rgbd_preprocessor")
        self.name = args.name
        self.bridge = CvBridge()
        self.width = args.width
        self.height = args.height
        self.period = 1.0 / args.rate
        self.latest_pair = None
        self.latest_pair_received_ns = None
        self.last_output_timestamp_ns = None
        self.pair_count = 0
        self.frame_lock = threading.Lock()
        self.input_callback_group = MutuallyExclusiveCallbackGroup()
        self.output_callback_group = MutuallyExclusiveCallbackGroup()

        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.rgb_pub = self.create_publisher(Image, args.rgb_output, output_qos)
        self.depth_pub = self.create_publisher(Image, args.depth_output, output_qos)

        self.rgb_sub = self.depth_sub = self.sync = None
        self.source = None
        self.declare_parameter("source", self.name, ParameterDescriptor(
            description="the camera whose driver this reads: its own role, "
                        "or the other one to swap the two"))
        self._subscribe(self.name, args.rgb_input, args.depth_input)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.publish_timer = self.create_timer(
            self.period,
            self._publish_latest,
            callback_group=self.output_callback_group,
        )

        self.get_logger().info(
            f"{self.name}: {args.rgb_input} -> {self.width}x{self.height} @ "
            f"{args.rate:.1f} Hz on {args.rgb_output} + {args.depth_output}"
        )

    def _subscribe(self, source, rgb_input=None, depth_input=None):
        """Read the driver of camera `source`: its raw topics, unless named."""
        default_rgb, default_depth = cameras.raw_topics(source)
        rgb_input, depth_input = rgb_input or default_rgb, depth_input or default_depth
        for sub in (self.rgb_sub, self.depth_sub):
            if sub is not None:
                self.destroy_subscription(sub.sub)
        with self.frame_lock:
            # The other camera's last pair must not go out as this one's.
            self.latest_pair = None
            self.latest_pair_received_ns = None
        self.rgb_sub = message_filters.Subscriber(
            self, Image, rgb_input, qos_profile=self.sensor_qos,
            callback_group=self.input_callback_group)
        self.depth_sub = message_filters.Subscriber(
            self, Image, depth_input, qos_profile=self.sensor_qos,
            callback_group=self.input_callback_group)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=8, slop=0.01)
        self.sync.registerCallback(self._on_rgbd)
        self.source = source

    def _on_set_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name != "source":
                continue
            source = str(parameter.value)
            if source not in cameras.CAMERAS:
                return SetParametersResult(
                    successful=False,
                    reason=f"source must be one of {', '.join(cameras.CAMERAS)}")
            if source != self.source:
                self._subscribe(source)
                self.get_logger().info(
                    f"{self.name}: now reading the {source} driver "
                    f"({cameras.raw_topics(source)[0]})")
        return SetParametersResult(successful=True)

    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image):
        with self.frame_lock:
            self.latest_pair = (rgb_msg, depth_msg)
            self.latest_pair_received_ns = self.get_clock().now().nanoseconds

    def _publish_latest(self):
        with self.frame_lock:
            if self.latest_pair is None:
                return
            rgb_msg, depth_msg = self.latest_pair
            received_ns = self.latest_pair_received_ns

        try:
            now_ns = self.get_clock().now().nanoseconds
            if received_ns is None or now_ns - received_ns > 100_000_000:
                self.get_logger().warning(
                    f"{self.name}: no fresh RGB-D pair from the driver; "
                    f"pausing its output",
                    throttle_duration_sec=5.0,
                )
                return

            if (
                rgb_msg.width == self.width
                and rgb_msg.height == self.height
                and rgb_msg.encoding.lower() == "rgb8"
                and depth_msg.width == self.width
                and depth_msg.height == self.height
                and depth_msg.encoding.lower() == "16uc1"
            ):
                # Fast path for the production profile: avoid two full image
                # conversions and resizes when the source already matches the target.
                rgb_out = rgb_msg
                depth_out = depth_msg
            else:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
                depth = self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding="passthrough",
                )
                rgb_small = cv2.resize(
                    rgb,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )
                depth_small = cv2.resize(
                    depth,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )
                rgb_out = self.bridge.cv2_to_imgmsg(rgb_small, encoding="rgb8")
                depth_out = self.bridge.cv2_to_imgmsg(depth_small, encoding="16UC1")

            # RealSense reports a hardware-clock timestamp while the backend
            # uses the ROS/system clock. Stamp both images on the nearest output-rate
            # ROS-time slot so camera and robot messages can be synchronized.
            period_ns = round(self.period * 1_000_000_000)
            target_timestamp_ns = ((now_ns + period_ns // 2) // period_ns) * period_ns
            if self.last_output_timestamp_ns is None:
                next_timestamp_ns = target_timestamp_ns
            else:
                next_timestamp_ns = self.last_output_timestamp_ns + period_ns
            if next_timestamp_ns > target_timestamp_ns:
                return

            # Catch up short scheduler delays without creating gaps in the
            # dataset clock. Longer stalls remain visible to the validator.
            oldest_allowed_ns = target_timestamp_ns - 4 * period_ns
            next_timestamp_ns = max(next_timestamp_ns, oldest_allowed_ns)
            while next_timestamp_ns <= target_timestamp_ns:
                output_stamp = Time(
                    sec=next_timestamp_ns // 1_000_000_000,
                    nanosec=next_timestamp_ns % 1_000_000_000,
                )
                rgb_out.header.frame_id = rgb_msg.header.frame_id
                rgb_out.header.stamp = output_stamp
                depth_out.header.frame_id = depth_msg.header.frame_id
                depth_out.header.stamp = output_stamp
                self.rgb_pub.publish(rgb_out)
                self.depth_pub.publish(depth_out)
                self.pair_count += 1
                self.last_output_timestamp_ns = next_timestamp_ns
                next_timestamp_ns += period_ns
        except Exception as error:
            self.get_logger().warning(
                f"{self.name}: could not preprocess an RGB-D pair: {error}",
                throttle_duration_sec=5.0,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="chest",
                        help="the camera's role, for the log")
    parser.add_argument(
        "--rgb-input",
        default="/camera/chest/color/image_raw",
    )
    parser.add_argument(
        "--depth-input",
        default="/camera/chest/aligned_depth_to_color/image_raw",
    )
    parser.add_argument("--rgb-output", default="/openarm/camera/chest/rgb")
    parser.add_argument("--depth-output", default="/openarm/camera/chest/depth")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--rate", type=float, default=25.0)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    if args.width <= 0 or args.height <= 0 or args.rate <= 0:
        parser.error("width, height, and rate must be positive")

    rclpy.init()
    node = RGBDPreprocessor(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
