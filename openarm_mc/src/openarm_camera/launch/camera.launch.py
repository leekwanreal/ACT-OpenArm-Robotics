"""The cameras, chest and front: real where plugged in, virtual where not.

Which RealSense is which comes from cameras.yaml at the workspace root
(./setup_camera.sh writes it; openarm_camera/cameras.py):

    ros2 launch openarm_camera camera.launch.py                 # both cameras
    ros2 launch openarm_camera camera.launch.py cameras:=chest  # one of them
    ros2 launch openarm_camera camera.launch.py virtual:=none   # real ones only

For each camera <name> whose RealSense is plugged in:

    realsense2_camera  /camera/<name>/{color,aligned_depth_to_color}/image_raw,
                       the device chosen by its serial number
    rgbd_preprocessor  /openarm/camera/<name>/{rgb,depth}: paired, 640x480 @ 25 Hz

and for one that is not, virtual_camera: black frames on the same two
topics, frame_id <name>_virtual. With no cameras.yaml yet, a single
RealSense is the chest camera, as before.
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from openarm_camera import cameras as camera_config
from openarm_camera.launch_env import PROFILES_ENV, fast_dds_actions


def _cameras(context):
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    start_driver = value("start_driver").lower() == "true"
    prefix = value("topic_prefix")
    # The driver runs elsewhere (or a test feeds the raw topics): every
    # camera is a preprocessor on them, nothing is virtual.
    plugged = camera_config.plugged_serials() if start_driver else None
    try:
        if start_driver:
            chosen = camera_config.plan(value("cameras"), value("virtual"),
                                        plugged=plugged)
        else:
            requested = value("cameras")
            names = camera_config.CAMERAS if requested.strip() in ("", "auto") \
                else camera_config.parse_names(requested)
            chosen = [(name, "", "external") for name in names]
    except ValueError as error:
        raise RuntimeError(f"openarm_camera: {error}") from error
    if not chosen:
        return [LogInfo(msg="openarm_camera: no cameras")]

    actions = [LogInfo(msg="openarm_camera: " + ", ".join(
        f"{name} {'RealSense ' + serial if source == 'realsense' else source}"
        for name, serial, source in chosen))]
    if plugged is None and start_driver:
        actions.append(LogInfo(msg="openarm_camera: cannot list the RealSense "
                                   "cameras (no rs-enumerate-devices)"))
    profile = context.environment.get(PROFILES_ENV, "")
    for name, serial, source in chosen:
        rgb_input, depth_input = camera_config.raw_topics(name)
        if source == "virtual":
            actions.append(Node(
                package="openarm_camera",
                executable="virtual_camera",
                name=f"openarm_{name}_virtual_camera",
                output="screen",
                arguments=[
                    "--name", name,
                    "--rgb-output", camera_config.rgb_topic(name, prefix),
                    "--depth-output", camera_config.depth_topic(name, prefix),
                    "--width", value("width"),
                    "--height", value("height"),
                    "--rate", value("rate"),
                ],
            ))
            continue
        if source == "realsense":
            # rs_launch.py sees only its own arguments: it warns about every
            # launch argument it does not know, and the ~100 it declares --
            # config_file among them -- must not leak out of this group.
            actions.append(GroupAction(
                # The group does not forward the environment: the driver,
                # which publishes the frames, needs the Fast DDS profile most.
                ([SetEnvironmentVariable(PROFILES_ENV, profile)] if profile else []) +
                [IncludeLaunchDescription(PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare("realsense2_camera"),
                                          "launch", "rs_launch.py"])))],
                scoped=True,
                forwarding=False,
                launch_configurations={
                    "camera_namespace": "camera",
                    "camera_name": name,
                    # "_" keeps an all-digit serial a string; the driver
                    # strips it. Empty: the first RealSense it finds.
                    "serial_no": f"_{serial}" if serial else "''",
                    "initial_reset": value("initial_reset"),
                    "enable_color": "true",
                    "enable_depth": "true",
                    "enable_sync": "true",
                    "align_depth.enable": "true",
                    "rgb_camera.color_profile": value("color_profile"),
                    "depth_module.depth_profile": value("depth_profile"),
                },
            ))
        actions.append(Node(
            package="openarm_camera",
            executable="rgbd_preprocessor",
            name=f"openarm_{name}_preprocessor",
            output="screen",
            arguments=[
                "--name", name,
                "--rgb-input", rgb_input,
                "--depth-input", depth_input,
                "--rgb-output", camera_config.rgb_topic(name, prefix),
                "--depth-output", camera_config.depth_topic(name, prefix),
                "--width", value("width"),
                "--height", value("height"),
                "--rate", value("rate"),
            ],
        ))
    return actions


def generate_launch_description():
    env = os.environ.get
    args = [
        DeclareLaunchArgument(
            "cameras", default_value=env("OPENARM_CAMERAS", "auto"),
            description="auto (chest and front), or a list: chest, front, "
                        "chest,front"),
        DeclareLaunchArgument(
            "virtual", default_value=env("OPENARM_CAMERAS_VIRTUAL", "auto"),
            description="auto: a camera not plugged in is a virtual (black) "
                        "one; none: only real ones; all; or a list of roles"),
        DeclareLaunchArgument(
            "start_driver", default_value="true",
            description="start realsense2_camera (false when the driver runs "
                        "elsewhere, or a test publishes the raw topics)"),
        DeclareLaunchArgument(
            "initial_reset", default_value=env("OPENARM_CAMERA_INITIAL_RESET", "true"),
            description="reset each camera as its driver starts: a few seconds "
                        "slower, but one stopped uncleanly (Ctrl+C, a replug) "
                        "otherwise fails with 'Depth stream start failure' "
                        "and sends no RGB-D pair"),
        DeclareLaunchArgument(
            "color_profile", default_value=env("OPENARM_CAMERA_COLOR_PROFILE",
                                               "640x480x30")),
        DeclareLaunchArgument(
            "depth_profile", default_value=env("OPENARM_CAMERA_DEPTH_PROFILE",
                                               "640x480x30")),
        DeclareLaunchArgument(
            "topic_prefix", default_value=camera_config.TOPIC_PREFIX,
            description="the prepared frames go to <prefix>/<camera>/{rgb,depth}"),
        DeclareLaunchArgument("width", default_value=env("OPENARM_CAMERA_WIDTH", "640")),
        DeclareLaunchArgument("height", default_value=env("OPENARM_CAMERA_HEIGHT", "480")),
        DeclareLaunchArgument("rate", default_value=env("OPENARM_CAMERA_FREQUENCY_HZ", "25")),
    ]
    return LaunchDescription(fast_dds_actions() + args
                             + [OpaqueFunction(function=_cameras)])
