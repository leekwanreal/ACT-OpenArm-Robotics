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

"""Run one episode of data_recorder.py per recording, armed in advance.

The recorder writes its HDF5 file when it stops, so an episode is a process
that lives exactly as long as the recording: stop sends it SIGINT and it
saves. That is also how a recording survives the dashboard being closed
mid-episode -- the file is still written.

Starting that process takes about two seconds (rclpy, h5py, cv2, then the
subscriptions finding their topics), which is what made Start recording lag.
So the next episode's process is started ahead of time with --standby: it
follows the streams and writes nothing until told `start <episode>`, which
takes a few tens of milliseconds. The dashboard arms one at startup and
another as each recording stops -- not while one runs, which it would slow.
"""

import collections
import os
import queue
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# <workspace>/dataset/, one folder per day: dataset/2026-10-08/episode_...hdf5.
# Found through this file's real path, which a --symlink-install build keeps
# in src/; OPENARM_DATASET_DIR names another root.
DEFAULT_OUTPUT_DIR = str(Path(os.path.realpath(__file__)).parents[3] / "dataset")
# What the proxy sends the controller -- teleop's targets while it drives --
# kept beside each sample as joint_commands. `action` stays qpos.
DEFAULT_COMMAND_TOPIC = "/openarm/joint_commands"
# The first standby also waits out its imports and DDS discovery.
READY_TIMEOUT_S = 15.0
START_TIMEOUT_S = 5.0


def output_dir():
    return os.environ.get("OPENARM_DATASET_DIR", DEFAULT_OUTPUT_DIR)


def episode_name():
    return datetime.now().strftime("episode_%Y%m%d_%H%M%S")


def day_folder():
    """The day's folder under the root, decided when an episode starts --
    not when its recorder was armed, which may have been the day before."""
    return datetime.now().strftime("%Y-%m-%d")


class Recorder:
    """One data_recorder.py process: armed, then recording, then saved.

    `cameras`: the ones it records (chest, or chest and front). It follows
    their streams from the moment it is armed, so a recorder armed for one
    set records that set: another choice arms another recorder."""

    def __init__(self, directory=None, extra_args=None, cameras=("chest",)):
        self.cameras = tuple(cameras)
        self.directory = directory or output_dir()
        os.makedirs(self.directory, exist_ok=True)
        command = [
            sys.executable, "-u", "-m", "openarm_ui.data_recorder",
            "--standby", "--output-dir", self.directory,
            # 0 means no limit: the episode ends when the dashboard says stop.
            "--max-duration", "0",
            "--command-topic", os.environ.get(
                "OPENARM_RECORD_COMMAND_TOPIC", DEFAULT_COMMAND_TOPIC),
            "--use-state-as-action",
            "--cameras", ",".join(self.cameras),
        ] + list(extra_args or [])
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"))
        self.episode = None
        self.path = None                # where the episode's .hdf5 lands
        # Its last lines, for an error, and its answers (lines with @). The
        # output is read all the time: a full pipe would block its logging.
        self.lines = collections.deque(maxlen=40)
        self.answers = queue.Queue()
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            line = line.rstrip()
            if line.startswith("@"):
                if line == "@ready":
                    self.ready.set()
                else:
                    self.answers.put(line)
            else:
                self.lines.append(line)
        self.answers.put(None)          # it exited
        self.ready.set()

    def alive(self):
        return self.process.poll() is None

    @property
    def recording(self):
        return self.episode is not None and self.alive()

    def _why(self):
        last = next((line for line in reversed(self.lines) if line.strip()), "")
        return last or f"exited with code {self.process.poll()}"

    def start(self, episode=None):
        """Begin the episode; -> its name, or raises RuntimeError."""
        episode = episode or episode_name()
        day = day_folder()
        if not self.ready.wait(READY_TIMEOUT_S) or not self.alive():
            raise RuntimeError(f"recorder failed to start: {self._why()}")
        try:
            self.process.stdin.write(f"start {episode} {day}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError):
            raise RuntimeError(f"recorder failed to start: {self._why()}")
        try:
            answer = self.answers.get(timeout=START_TIMEOUT_S)
        except queue.Empty:
            answer = "@error no answer from the recorder"
        if answer is None:
            raise RuntimeError(f"recorder failed to start: {self._why()}")
        if not answer.startswith("@recording"):
            raise RuntimeError(f"cannot record: {answer[len('@error '):]}")
        self.episode = episode
        self.path = os.path.join(self.directory, day, f"{episode}.hdf5")
        return episode

    def stop(self, timeout=60.0):
        """Save the episode: SIGINT, then wait for every sample to reach
        the disk (up to the writer's backlog). -> True when it exited."""
        if not self.alive():
            return True
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            self.process.kill()
            return False

    def close(self):
        """Dismiss an armed recorder that never recorded: no file."""
        if self.episode is None and self.alive():
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start_recorder(episode=None, output_dir=None, extra_args=None,
                   cameras=("chest",)):
    """One recorder, started now; -> (Recorder, episode name). Kept for
    callers that do not arm one in advance."""
    recorder = Recorder(output_dir, extra_args, cameras)
    try:
        name = recorder.start(episode)
    except RuntimeError:
        recorder.close()
        raise
    return recorder, name
