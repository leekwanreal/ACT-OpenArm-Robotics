"""Top-level OpenArm bringup.

Each subsystem keeps its launch file in the package that owns it; this one
only picks which of them come up, through an enable_<subsystem> argument.
Arguments not declared here fall through to the included launch files:

  ros2 launch bringup bringup.launch.py enable_sim:=true use_sim:=true

ACT comes up when there is a model to run: $OPENARM_ACT_MODEL, or an .onnx
under checkpoint/onnx/ (enable_act:=false keeps it off). SmolVLA comes up
with enable_smolvla:=true server:=ws://<gpu-host>:8000.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from openarm_act.models import find_model
from openarm_camera.launch_env import fast_dds_actions

# (name, package that owns the launch file, file, enabled by default)
SUBSYSTEMS = [
    ("camera", "openarm_camera", "camera.launch.py", "true"),
    ("controller", "openarm_controller", "controller.launch.py", "true"),
    ("proxy", "openarm_command_proxy", "proxy.launch.py", "true"),
    ("ui", "openarm_ui", "ui.launch.py", "true"),
    # The MuJoCo simulator stands in for the robot, so it is off by default.
    ("sim", "openarm_mujoco", "sim.launch.py", "false"),
    # On when there is a model; checkpoint:= and stats_file:= pass through.
    ("act", "openarm_act", "act.launch.py", "true" if find_model() else "false"),
    # Needs an openarm_smolvla server; server:= passes through.
    ("smolvla", "openarm_smolvla", "smolvla.launch.py", "false"),
]


def generate_launch_description():
    # First, so every node below gets the camera frames whole.
    actions = fast_dds_actions()
    for name, package, launch_file, default in SUBSYSTEMS:
        enable = "enable_" + name
        actions.append(DeclareLaunchArgument(
            enable, default_value=default,
            description=f"Start the {name} subsystem ({package})"))
        # Scoped: what one subsystem declares (act and proxy both have a
        # config_file) stays in it, while bringup's arguments still reach all.
        actions.append(GroupAction(
            [IncludeLaunchDescription(PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare(package), "launch", launch_file])))],
            scoped=True,
            condition=IfCondition(LaunchConfiguration(enable)),
        ))
    return LaunchDescription(actions)
