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

"""Start and stop the controller and the simulator, so connect can pick a mode.

Which bus the controller drives is fixed when its CAN threads come up:
`left_interface` and `right_interface` are read-only parameters. Switching
between the robot and MuJoCo is therefore not a parameter write but a
restart, and that is what this module does.

The two modes are a pair, not two independent switches. `real` wants the
simulator gone, because a MuJoCo process left answering on vcan0 is a second
robot nobody is watching; `sim` wants it running, because without it the
controller talks to an empty bus. So each mode says what the simulator should
be doing, and `plan` works out the difference from what is running now.

Processes are started in their own session and stopped by signalling the whole
group. `ros2 launch` does not reliably take its nodes down with it -- the same
reason the README tells you to check `pgrep -af "lib/openarm"` -- and a node
that outlives its launch keeps the bus open and breaks the next start.

Nodes this dashboard did not start are still stopped when they are in the way:
a controller from `ros2 launch bringup` is the normal case, not an exception.
Those are signalled one process at a time, never by group, because they share
a group with the rest of their launch -- this dashboard included.
A controller already running in the mode being asked for is left alone, since
restarting it would disarm an arm that is holding its position for no reason.
"""

import collections
import os
import signal
import subprocess
import threading
import time

# What each mode needs: the controller's launch arguments, and whether the
# MuJoCo simulator belongs on the bus.
MODES = {
    "real": {
        "controller_args": ["use_sim:=false"],
        "wants_sim": False,
        "label": "the robot",
    },
    "sim": {
        "controller_args": ["use_sim:=true"],
        "wants_sim": True,
        "label": "the simulator",
    },
}

# How a process of each kind is recognised: the tail of the installed
# executable's path, so a node is found whoever started it. The `ros2 launch`
# wrapper around it does not match, and does not need to -- a launch whose
# only node has exited stops on its own.
CONTROLLER_PATTERN = "lib/openarm_controller/controller"
SIM_PATTERN = "lib/openarm_mujoco/can_bridge"

# A MuJoCo window and a model load take noticeably longer than a controller.
CONTROLLER_START_TIMEOUT = 20.0
SIM_START_TIMEOUT = 30.0
STOP_TIMEOUT = 10.0

# How much of a child's output to keep for the message on the page.
TAIL_LINES = 50


class SupervisorError(RuntimeError):
    """A mode change that could not be carried out."""


def _pids(suffix):
    """PIDs running the executable whose path ends with `suffix`.

    This reads /proc rather than shelling out to `pgrep -f`, because a
    substring match over whole command lines also finds processes that merely
    mention the path -- a shell running pgrep against it, or this dashboard's
    own launch arguments. A false positive here is not cosmetic: it makes
    `plan` believe the simulator is already up and skip starting it.

    So a match has to be an argument that is an absolute path ending in the
    executable, which is what a launched node's argv holds and what a pattern
    typed on a command line does not.
    """
    mine = os.getpid()
    found = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == mine:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                argv = handle.read().split(b"\0")
        except OSError:          # it exited, or it is not ours to read
            continue
        for argument in argv:
            text = argument.decode("utf-8", "replace")
            if text.startswith("/") and text.endswith(suffix):
                found.append(pid)
                break
    return found


def _signal_group(pid, number):
    """Signal a whole process group. Only for groups we created ourselves.

    Never call this on a process this module did not start. A controller that
    came up under `ros2 launch bringup` shares its group with the rest of that
    launch -- the dashboard included -- so signalling the group would take
    down the page that asked for the mode change.
    """
    try:
        os.killpg(os.getpgid(pid), number)
    except (ProcessLookupError, PermissionError):
        _signal_pid(pid, number)


def _signal_pid(pid, number):
    """Signal one process and nothing else around it."""
    try:
        os.kill(pid, number)
    except (ProcessLookupError, PermissionError):
        pass


def _gone(pattern, timeout):
    """Wait for every process matching `pattern` to leave."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pids(pattern):
            return True
        time.sleep(0.1)
    return not _pids(pattern)


def _stop_pattern(pattern, owned=None, timeout=STOP_TIMEOUT):
    """Stop everything matching `pattern`: SIGINT, then SIGTERM, then SIGKILL.

    SIGINT first because that is the shutdown a ROS node handles, which lets
    the controller disable the motors on its way out rather than leaving them
    armed with nothing driving them.

    `owned` is the launch this module started for that pattern, if any. Its
    group is signalled, which takes the `ros2 launch` wrapper down with its
    node; a process that belongs to somebody else's launch is signalled on
    its own, so that launch is left to notice and carry on.
    """
    if not _pids(pattern) and owned is None:
        return True
    own_pid = owned.pid if owned is not None and owned.poll() is None else None
    own_group = None
    if own_pid is not None:
        try:
            own_group = os.getpgid(own_pid)
        except (ProcessLookupError, PermissionError):
            own_group = None

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if own_pid is not None and owned.poll() is None:
            _signal_group(own_pid, number)
        for pid in _pids(pattern):
            # Anything already covered by the group signal above.
            if own_group is not None:
                try:
                    if os.getpgid(pid) == own_group:
                        continue
                except (ProcessLookupError, PermissionError):
                    continue
            _signal_pid(pid, number)
        # SIGKILL cannot be caught, so the wait after it is only reaping.
        if _gone(pattern, 3.0 if number is signal.SIGKILL else timeout / 2):
            if owned is not None:
                try:
                    owned.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            return True
    return False


class Supervisor:
    """Owns the controller and simulator processes behind the connect button."""

    def __init__(self, logger=None):
        self._logger = logger
        self.controller = None
        self.sim = None
        # The mode the last successful start was for, so status can say what
        # this dashboard believes it brought up.
        self.mode = None

    def _log(self, message):
        if self._logger is not None:
            self._logger.info(f"supervisor: {message}")

    # -- what is running ----------------------------------------------------

    def controller_running(self):
        return bool(_pids(CONTROLLER_PATTERN))

    def sim_running(self):
        return bool(_pids(SIM_PATTERN))

    def plan(self, mode, current_mode):
        """What has to change to be connected in `mode`.

        `current_mode` is what the running controller reports, which is read
        from its interfaces rather than from anything stored here -- a
        controller started by bringup is as real as one started from the page.
        """
        spec = MODES[mode]
        controller_ok = self.controller_running() and current_mode == mode
        return {
            "restart_controller": not controller_ok,
            "start_sim": spec["wants_sim"] and not self.sim_running(),
            "stop_sim": not spec["wants_sim"] and self.sim_running(),
        }

    # -- starting and stopping ----------------------------------------------

    def _spawn(self, command, pattern, timeout, what, tag):
        """Launch `command` and wait for its node to appear."""
        self._log(f"starting {what}: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
                # Its own session, so stopping it later can signal the group
                # and take the node with the launch.
                start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except OSError as error:
            raise SupervisorError(f"cannot start {what}: {error}") from error

        tail = collections.deque(maxlen=TAIL_LINES)
        threading.Thread(target=self._drain, args=(process, tail, tag),
                         daemon=True).start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _pids(pattern):
                return process
            if process.poll() is not None:
                # Let the reader catch up with what it printed on the way out.
                time.sleep(0.3)
                raise SupervisorError(
                    f"{what} exited immediately: {_last(tail, process)}")
            time.sleep(0.2)

        _stop_pattern(pattern, owned=process)
        raise SupervisorError(f"{what} did not come up within {timeout:.0f} s")

    def _drain(self, process, tail, tag):
        """Forward a child's output, and keep the last of it.

        The pipe has to be read whether or not anyone wants the lines: it
        holds 64 KiB, and a child whose output fills it blocks on its next
        write. For the controller that is the CAN loop stopping, which would
        look like the robot freezing for no reason.

        Forwarding also puts the controller's own messages -- how many motors
        answered, feedback lost and restored -- in the dashboard's log, where
        they would otherwise be swallowed by the pipe.
        """
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if self._logger is not None:
                    self._logger.info(f"[{tag}] {line}")
        except (OSError, ValueError):     # closed under us on shutdown
            pass
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def start_sim(self):
        if self.sim_running():
            return
        self.sim = self._spawn(
            ["ros2", "launch", "openarm_mujoco", "sim.launch.py"],
            SIM_PATTERN, SIM_START_TIMEOUT, "the simulator", "sim")

    def stop_sim(self):
        if not _stop_pattern(SIM_PATTERN, owned=self.sim):
            raise SupervisorError("the simulator would not stop")
        self.sim = None
        self._log("simulator stopped")

    def start_controller(self, mode):
        command = ["ros2", "launch", "openarm_controller",
                   "controller.launch.py"]
        command += MODES[mode]["controller_args"]
        # The page arms the motors itself once the node is up, so that a
        # failure to start is never a robot that has already gone stiff.
        command.append("auto_enable:=false")
        self.controller = self._spawn(
            command, CONTROLLER_PATTERN, CONTROLLER_START_TIMEOUT,
            "the controller", "controller")

    def stop_controller(self):
        if not _stop_pattern(CONTROLLER_PATTERN, owned=self.controller):
            raise SupervisorError("the controller would not stop")
        self.controller = None
        self._log("controller stopped")

    def shutdown(self):
        """Stop whatever this dashboard started, on its way out.

        Only its own children: a controller that was already running when the
        page opened should outlive the page, the same as before.
        """
        mine = [process for process in (self.controller, self.sim)
                if process is not None and process.poll() is None]
        for process in mine:
            _signal_group(process.pid, signal.SIGINT)
        for process in mine:
            try:
                process.wait(timeout=STOP_TIMEOUT / 2)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
        self.controller = None
        self.sim = None


def _last(tail, process, lines=3):
    """The last of a dead process's output, for the message on the page."""
    recent = list(tail)[-lines:]
    if not recent:
        return f"exit code {process.returncode}"
    return " / ".join(recent)
