"""SmolVLA policy inference: preview, then execute, as openarm_act.

The model runs on a GPU machine, in openarm_smolvla:

    uv run scripts/serve.py checkpoint=releases/<name> port=8000

and this node asks it for each chunk:

    ros2 launch openarm_smolvla smolvla.launch.py server:=ws://gpu-host:8000
    ros2 service call /openarm_smolvla/generate std_srvs/srv/Trigger
    ros2 param set /openarm_command_proxy active_source smolvla
    ros2 param set /openarm_smolvla mode execute

Until the server answers, the node exits saying so and is started again
every 5 s (respawn:=false stops that).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _node(context):
    respawn = LaunchConfiguration("respawn").perform(context).lower() == "true"
    return [Node(
        package="openarm_smolvla",
        executable="smolvla_node",
        name="openarm_smolvla",
        output="screen",
        respawn=respawn,
        respawn_delay=5.0,
        parameters=[
            LaunchConfiguration("config_file"),
            # After the file, so the launch argument wins; a string as given.
            {
                "checkpoint": ParameterValue(LaunchConfiguration("server"),
                                             value_type=str),
            },
        ],
    )]


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("openarm_smolvla"), "config", "smolvla.yaml"])

    args = [
        DeclareLaunchArgument("config_file", default_value=default_config,
                              description="parameter YAML"),
        DeclareLaunchArgument(
            "server",
            default_value=os.environ.get("OPENARM_SMOLVLA_SERVER",
                                         "ws://localhost:8000"),
            description="the openarm_smolvla server, ws://<host>:<port>; by "
                        "default $OPENARM_SMOLVLA_SERVER, else localhost:8000"),
        DeclareLaunchArgument("respawn", default_value="true",
                              description="start the node again every 5 s "
                                          "while the server is not there"),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=_node)])
