"""OpenArm CAN controller, against the real robot or the MuJoCo simulator.

`use_sim` picks where the motors are. The real robot gives each arm its own
bus and both arms answer on send ids 0x01-0x08; the simulator serves both on
one virtual bus, so the right arm moves up by 0x20 to keep the ids apart.

    ros2 launch openarm_controller controller.launch.py                 # robot
    ros2 launch openarm_controller controller.launch.py use_sim:=true   # MuJoCo

Either default can still be overridden one at a time, e.g.
`use_sim:=true left_interface:=vcan1`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

default_gains = PathJoinSubstitution(
    [FindPackageShare("openarm_controller"), "config", "gains.yaml"])

SIM = {"left": "vcan0", "right": "vcan0", "offset": "32"}
ROBOT = {"left": "can1", "right": "can0", "offset": "0"}


def _pick(use_sim, key):
    """Choose the simulator's value or the robot's, at launch time."""
    return PythonExpression(
        ["'", SIM[key], "' if '", use_sim, "'.lower() in ('true', '1') else '",
         ROBOT[key], "'"])


def generate_launch_description():
    use_sim = LaunchConfiguration("use_sim")

    args = [
        DeclareLaunchArgument(
            "use_sim", default_value="false",
            description="true drives the MuJoCo simulator on vcan0, "
                        "false the robot on can0/can1"),
        DeclareLaunchArgument(
            "left_interface", default_value=_pick(use_sim, "left"),
            description="SocketCAN interface carrying the left arm"),
        DeclareLaunchArgument(
            "right_interface", default_value=_pick(use_sim, "right"),
            description="SocketCAN interface carrying the right arm"),
        DeclareLaunchArgument(
            "right_id_offset", default_value=_pick(use_sim, "offset"),
            description="added to the right arm's CAN ids; "
                        "32 when both arms share one bus"),
        DeclareLaunchArgument("arms", default_value="both",
                              description="both, left or right"),
        DeclareLaunchArgument("control_rate", default_value="400.0",
                              description="CAN read/write rate (Hz)"),
        DeclareLaunchArgument("publish_rate", default_value="100.0",
                              description="/openarm/joint_states rate (Hz)"),
        DeclareLaunchArgument(
            "auto_enable", default_value="false",
            description="true arms the motors as the controller starts; "
                        "false leaves them limp until something enables "
                        "them, which is what the dashboard's connect does"),
        DeclareLaunchArgument(
            "gains_file", default_value=default_gains,
            description="YAML holding the PD set the controller starts with"),
    ]

    parameters = [{
        "left_interface": LaunchConfiguration("left_interface"),
        "right_interface": LaunchConfiguration("right_interface"),
        "arms": LaunchConfiguration("arms"),
        "right_id_offset": LaunchConfiguration("right_id_offset"),
        "control_rate": LaunchConfiguration("control_rate"),
        "publish_rate": LaunchConfiguration("publish_rate"),
        "auto_enable": LaunchConfiguration("auto_enable"),
    }]

    controller = Node(
        package="openarm_controller", executable="controller",
        name="openarm_controller", output="screen",
        # Gains first, so an explicit launch argument still wins over the file.
        parameters=[LaunchConfiguration("gains_file")] + parameters,
    )

    return LaunchDescription(args + [controller])
