#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve the OpenArm dashboard and bridge it to ROS 2.

    ros_node.py    the ROS half: joint_states in, ui commands out, and the
                   parameters of the controller and the command proxy
    web_server.py  the browser half: the page over HTTP, state and actions
                   over a WebSocket on the same port
    recorder.py    one episode of data_recorder.py per recording
    joints.py      joint order and motor ids

Everything the page can change is a parameter on another node, so the same
things are reachable with `ros2 param` -- the page is one way in, not the
only one.
"""

import asyncio
import sys
import threading

import rclpy

from .ros_node import UiServer
from .web_server import Dashboard


def main(argv=None):
    rclpy.init(args=argv)
    node = UiServer()

    # ROS spins beside the server: the asyncio loop owns the main thread.
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    dashboard = Dashboard(node)
    status = 0
    try:
        asyncio.run(dashboard.run())
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        # Something the operator can act on, such as the port being taken.
        node.get_logger().error(str(error))
        status = 1
    finally:
        dashboard.shutdown_recorders()
        # Shut the context down first so the spin thread leaves the node
        # alone; destroying it underneath a running spin aborts the process.
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
    return status


if __name__ == "__main__":
    sys.exit(main())
