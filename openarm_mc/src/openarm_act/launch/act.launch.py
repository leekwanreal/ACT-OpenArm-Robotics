"""ACT policy inference: preview on the dashboard, then execute.

    ros2 launch openarm_act act.launch.py      # newest .onnx in checkpoint/onnx/
    ros2 launch openarm_act act.launch.py checkpoint:=~/act/policy.onnx
    # on http://localhost:8080: Generate trajectory, check the arrows, then
    # Run trajectory; or by hand:
    ros2 service call /openarm_act/generate std_srvs/srv/Trigger
    ros2 param set /openarm_act mode execute
    ros2 param set /openarm_command_proxy active_source act

The training checkpoint itself, without exporting:

    ros2 launch openarm_act act.launch.py \
        policy:=openarm_act.policy:ActPipelinePolicy \
        checkpoint:=$PWD/checkpoint/pth/best_checkpoint.pth \
        act_pipeline_path:=~/Project/OpenArm_MC

Without a model, to check the camera and state plumbing in simulation:

    ros2 launch openarm_act act.launch.py policy:=openarm_act.policy:HoldPolicy
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from openarm_act.models import find_model


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("openarm_act"), "config", "act.yaml"])

    args = [
        DeclareLaunchArgument("config_file", default_value=default_config,
                              description="parameter YAML"),
        DeclareLaunchArgument("checkpoint", default_value=find_model(),
                              description="model: .onnx, or .pth for "
                                          "ActPipelinePolicy; by default "
                                          "$OPENARM_ACT_MODEL, else the newest "
                                          ".onnx under checkpoint/onnx/"),
        DeclareLaunchArgument("stats_file", default_value="",
                              description="act_pipeline's dataset_stats.pkl (or .json); "
                                          "empty when the .onnx carries them"),
        DeclareLaunchArgument("policy",
                              default_value="openarm_act.policy:OnnxPolicy",
                              description="policy class as module:Class"),
        DeclareLaunchArgument("device", default_value="auto",
                              description="auto (cuda when the runtime has "
                                          "it, else cpu), cuda, cuda:<n> or cpu"),
        DeclareLaunchArgument("act_pipeline_path", default_value="",
                              description="directory holding act_pipeline"),
    ]

    node = Node(
        package="openarm_act",
        executable="act_node",
        name="openarm_act",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            # After the file, so the launch arguments win.
            {
                "checkpoint": LaunchConfiguration("checkpoint"),
                "stats_file": LaunchConfiguration("stats_file"),
                "policy": LaunchConfiguration("policy"),
                "device": LaunchConfiguration("device"),
                "act_pipeline_path": LaunchConfiguration("act_pipeline_path"),
            },
        ],
    )

    return LaunchDescription(args + [node])
