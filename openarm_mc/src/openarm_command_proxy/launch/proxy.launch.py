"""Command source selection for OpenArm.

Which sources exist and where they publish is config, not launch arguments:
see config/sources.yaml, or point `config_file` at your own copy. Only the
runtime choice is a launch argument, because it is the one thing that changes
between runs.

    ros2 launch openarm_command_proxy proxy.launch.py
    ros2 launch openarm_command_proxy proxy.launch.py active_source:=ui
    ros2 param set /openarm_command_proxy active_source teleop

It starts on `none`, so nothing reaches the arm until a source is chosen.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("openarm_command_proxy"), "config", "sources.yaml"])

    args = [
        DeclareLaunchArgument(
            "config_file", default_value=default_config,
            description="YAML listing the sources and their topics"),
        DeclareLaunchArgument(
            "active_source", default_value="none",
            description="which source reaches the arm at startup; "
                        "'none' blocks all"),
    ]

    proxy = Node(
        package="openarm_command_proxy",
        executable="command_proxy",
        name="openarm_command_proxy",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            # After the file, so the launch argument wins.
            {"active_source": LaunchConfiguration("active_source")},
        ],
    )

    return LaunchDescription(args + [proxy])
