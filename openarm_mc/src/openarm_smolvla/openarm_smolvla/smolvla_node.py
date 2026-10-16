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

"""Run a SmolVLA policy: preview what it would do, then let it drive.

openarm_act's node under another name, with SmolVLAPolicy in it, so the run
is the same in every way that matters for the arm: idle first, one shot or
continuous, chunks ensembled at 50 Hz, the ease-in onto the first step,
the refusals, Stop, Reset and the way home. What differs:

    model      an openarm_smolvla server (scripts/serve.py) on a GPU machine,
               named by the `checkpoint` parameter: ws://<host>:<port>.
               Setting it while idle connects to another one. No prompt:
               the model is trained on one task, which the server gives
               it (policy.py).
    topics     the `smolvla` source of the command proxy:
               /openarm/smolvla/{joint_commands, left_gripper, right_gripper,
               preview, status}, its chunks on /openarm/smolvla/trajectory.
    services   /openarm_smolvla/generate, /openarm_smolvla/reset

    ros2 launch openarm_smolvla smolvla.launch.py server:=ws://gpu-host:8000
    ros2 service call /openarm_smolvla/generate std_srvs/srv/Trigger
    ros2 param set /openarm_command_proxy active_source smolvla
    ros2 param set /openarm_smolvla mode execute     # run the trajectory once
    ros2 param set /openarm_smolvla running true     # or: continuously, previewing
    ros2 param set /openarm_smolvla mode execute     # after watching the preview
    ros2 param set /openarm_smolvla mode preview     # stop commanding; arm holds
"""

from rcl_interfaces.msg import ParameterDescriptor

from openarm_act.act_node import ActNode, main as run_node

from .policy import parse_server


class SmolVLANode(ActNode):
    """ActNode with a SmolVLA server as its model."""

    def __init__(self):
        super().__init__("openarm_smolvla")

    # ------------------------------------------------------------- the hooks

    def _extra_policy_options(self):
        """Declared here, before ActNode loads the policy with them."""
        self.declare_parameter("connect_timeout", 10.0, ParameterDescriptor(
            read_only=True, description="seconds to reach the server"))
        self.declare_parameter("request_timeout", 15.0, ParameterDescriptor(
            read_only=True,
            description="seconds a chunk may take before the server counts "
                        "as gone (it reconnects on the next); 0 waits for ever"))
        self.declare_parameter("clip_to_limits", True, ParameterDescriptor(
            read_only=True,
            description="clip the predicted targets to OpenArm V1's joint limits"))
        value = lambda name: self.get_parameter(name).value  # noqa: E731
        return {
            "connect_timeout": float(value("connect_timeout")),
            "request_timeout": float(value("request_timeout")),
            "clip_to_limits": bool(value("clip_to_limits")),
        }

    def _model_problem(self, path):
        """A server, not a file."""
        try:
            parse_server(path)
        except ValueError as error:
            return str(error)
        return None

    def _models(self):
        """No files to pick from: the model is whatever the server serves."""
        return []

    def _status(self):
        status = super()._status()
        policy = self.policy
        status["smolvla"] = {
            "server": policy.server,
            "task": policy.task,
            "server_ms": policy.server_ms,
            "round_trip_ms": policy.round_trip_ms,
            "connected": policy.ws is not None,
            "model": policy.metadata.get("model"),
            "finetune": policy.metadata.get("finetune"),
            "step": policy.metadata.get("step"),
            "checkpoint": policy.metadata.get("checkpoint"),
        }
        return status


def main(argv=None):
    return run_node(argv, node_class=SmolVLANode, name="openarm_smolvla")


if __name__ == "__main__":
    main()
