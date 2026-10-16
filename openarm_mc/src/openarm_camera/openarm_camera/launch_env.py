"""The environment every node of a launch needs for the camera frames.

Fast DDS, Jazzy's default RMW, drops most 640x480 colour frames with its
default transports (config/fastdds.xml says why); its profile file fixes
that. A launch file adds these actions first, so every process it starts
inherits them:

    LaunchDescription(fast_dds_actions() + [...])

A profile file already named in the environment, or another RMW, is left
alone.
"""

import os

from launch.actions import LogInfo, SetEnvironmentVariable

PROFILES_ENV = "FASTRTPS_DEFAULT_PROFILES_FILE"


def fast_dds_profile():
    """The path of openarm_camera's Fast DDS profile."""
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("openarm_camera"),
                        "config", "fastdds.xml")


def fast_dds_actions():
    rmw = os.environ.get("RMW_IMPLEMENTATION", "")
    if os.environ.get(PROFILES_ENV) or (rmw and "fastrtps" not in rmw):
        return []
    profile = fast_dds_profile()
    return [SetEnvironmentVariable(PROFILES_ENV, profile),
            LogInfo(msg=f"Fast DDS: large frames over shared memory ({profile})")]
