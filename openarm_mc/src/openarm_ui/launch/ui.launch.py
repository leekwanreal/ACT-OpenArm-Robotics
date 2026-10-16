"""The OpenArm web dashboard.

    ros2 launch openarm_ui ui.launch.py
    # then open http://localhost:8080

The page publishes as the `ui` source of the command proxy, so its sliders
only reach the arm once that source is selected -- from the page itself, or
with `ros2 param set /openarm_command_proxy active_source ui`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("http_port", default_value="8080",
                              description="port serving the page and its WebSocket"),
        DeclareLaunchArgument("source_name", default_value="ui",
                              description="which proxy source the sliders publish as"),
        DeclareLaunchArgument("telemetry_rate", default_value="30.0",
                              description="Hz the page is updated at"),
        DeclareLaunchArgument("command_rate", default_value="50.0",
                              description="Hz slider targets are published at"),
    ]

    ui = Node(
        package="openarm_ui",
        executable="ui_server",
        name="openarm_ui",
        output="screen",
        parameters=[{
            "http_port": LaunchConfiguration("http_port"),
            "source_name": LaunchConfiguration("source_name"),
            "telemetry_rate": LaunchConfiguration("telemetry_rate"),
            "command_rate": LaunchConfiguration("command_rate"),
        }],
    )

    return LaunchDescription(args + [ui])
