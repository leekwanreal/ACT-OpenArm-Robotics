"""MuJoCo standing in for the robot's motors on a CAN bus.

The node answers Damiao CAN traffic with this repository's physics, so a
controller drives the simulation exactly as it drives the arm. It publishes
no topics -- the bus carries the robot's data -- but it is an ordinary node,
so it can be retuned while it runs:

    ros2 param set /openarm_mujoco_sim paused true
    ros2 param set /openarm_mujoco_sim realtime 0.25
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "interface", default_value="vcan0",
            description="SocketCAN interface the simulated motors answer on"),
        DeclareLaunchArgument(
            "right_id_offset", default_value="32",
            description="added to the right arm's CAN ids; 32 keeps both arms "
                        "on one bus, as one simulated robot does"),
        DeclareLaunchArgument(
            "model", default_value="",
            description="MJCF path; empty uses this checkout's v1/scene.xml"),
        DeclareLaunchArgument(
            "keyframe", default_value="home",
            description="keyframe to start from, empty for qpos 0"),
        DeclareLaunchArgument(
            "control_mode", default_value="mit",
            description="mit honours the kp/kd sent over CAN; "
                        "position uses the model's tuned actuators"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="open the MuJoCo window"),
        DeclareLaunchArgument("realtime", default_value="1.0",
                              description="speed against the wall clock"),
    ]

    simulator = Node(
        package="openarm_mujoco",
        executable="can_bridge",
        name="openarm_mujoco_sim",
        output="screen",
        parameters=[{
            "interface": LaunchConfiguration("interface"),
            "right_id_offset": LaunchConfiguration("right_id_offset"),
            "model": LaunchConfiguration("model"),
            "keyframe": LaunchConfiguration("keyframe"),
            "control_mode": LaunchConfiguration("control_mode"),
            "viewer": LaunchConfiguration("viewer"),
            "realtime": LaunchConfiguration("realtime"),
        }],
    )

    return LaunchDescription(args + [simulator])
