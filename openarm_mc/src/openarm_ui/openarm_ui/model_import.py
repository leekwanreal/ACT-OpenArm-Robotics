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

"""A model picked on the page, brought into checkpoint/ and loaded.

The browser cannot hand over a path, only the file, so the page sends it
over its socket in chunks (web_server.py). Then:

    .onnx  kept as checkpoint/onnx/<name>/<name>.onnx
    .pth   kept as checkpoint/pth/<name>.pth, exported by utils/export_onnx.py
           to checkpoint/onnx/<name>/policy.onnx (its stats beside it)

and the .onnx is loaded into openarm_act. A file already there, byte for
byte, is not kept twice: the copy in place is used.

One import at a time, from any page; the page that sends it owns it, and a
page that goes away mid-upload takes its partial file with it. Everything
lands in checkpoint/.incoming/ first and is renamed into place only once
whole, so a half-written model is never listed or loaded.

The pure helpers (safe_stem, unique_dir, same_file) are tested on their own.
"""

import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid

EXTENSIONS = (".onnx", ".pth")
MAX_BYTES = 4 << 30                 # 4 GB: a checkpoint is a few hundred MB
EXPORT_TIMEOUT_S = 600.0
LOG_LINES = 12


def safe_stem(filename):
    """A folder name from a file name: "best_checkpoint (1).pth" ->
    "best_checkpoint_1"."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "model"


def unique_dir(parent, stem, suffix=""):
    """parent/stem+suffix, or parent/stem_2+suffix, _3... when taken."""
    candidate, index = os.path.join(parent, stem + suffix), 2
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{stem}_{index}{suffix}")
        index += 1
    return candidate


def file_digest(path, chunk=1 << 20):
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def same_file(directory, suffix, size, digest):
    """A file under `directory` with this size and content, or None."""
    for folder, _dirs, names in os.walk(directory):
        if ".incoming" in folder.split(os.sep):
            continue
        for name in names:
            path = os.path.join(folder, name)
            if not name.endswith(suffix):
                continue
            try:
                if os.path.getsize(path) == size and file_digest(path) == digest:
                    return path
            except OSError:
                continue
    return None


class ModelImport:
    """The upload, export and load of one model at a time."""

    def __init__(self, workspace, load, log):
        """`load(path)` -> (ok, message) loads an .onnx into openarm_act,
        blocking; `log(text)` reports to the node's log."""
        self.workspace = workspace
        self.checkpoint = os.path.join(workspace, "checkpoint")
        self.pth_dir = os.path.join(self.checkpoint, "pth")
        self.onnx_dir = os.path.join(self.checkpoint, "onnx")
        self.incoming = os.path.join(self.checkpoint, ".incoming")
        self.export_script = os.path.join(workspace, "utils", "export_onnx.py")
        self._load = load
        self._log = log
        self.owner = None           # the connection uploading
        self.phase = "idle"         # receiving, saving, exporting, loading
        self.name = None
        self.size = 0
        self.received = 0
        self.part = None            # (path, open file, digest)
        self.lines = []             # the export's last lines
        self.started = None
        self.result_id = 0
        self.result = None          # (ok, message, path or None)
        self.task = None

    # ------------------------------------------------------------- the page

    def snapshot(self):
        return {
            "phase": self.phase, "name": self.name, "size": self.size,
            "received": self.received, "log": self.lines[-LOG_LINES:],
            "elapsed_s": (time.monotonic() - self.started
                          if self.started and self.phase != "idle" else None),
            "last": None if self.result is None else {
                "id": self.result_id, "ok": self.result[0],
                "message": self.result[1], "path": self.result[2]},
        }

    def begin(self, connection, name, size):
        """-> None, or why the upload cannot start."""
        if self.phase != "idle":
            return f"already importing {self.name}"
        name = os.path.basename(str(name or ""))
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in EXTENSIONS:
            return "pick an .onnx or a .pth"
        try:
            size = int(size)
        except (TypeError, ValueError):
            return "no file size given"
        if not 0 < size <= MAX_BYTES:
            return f"{name} is {size / 1e9:.1f} GB; at most {MAX_BYTES / 1e9:.0f} GB"
        if suffix == ".pth" and not os.path.isfile(self.export_script):
            return f"no {self.export_script} to export a .pth with"
        os.makedirs(self.incoming, exist_ok=True)
        path = os.path.join(self.incoming, f"{uuid.uuid4().hex}{suffix}.part")
        self.part = (path, open(path, "wb"), hashlib.blake2b(digest_size=16))
        self.owner, self.phase, self.name = connection, "receiving", name
        self.size, self.received, self.lines = size, 0, []
        self.started = time.monotonic()
        self._log(f"receiving {name} ({size / 1e6:.0f} MB)")
        return None

    def chunk(self, connection, data):
        """One piece of the file, in order; -> None, or why it was refused."""
        if connection is not self.owner or self.phase != "receiving":
            return "no upload in progress from this page"
        if self.received + len(data) > self.size:
            self.abort(connection, "more bytes than the file has")
            return "more bytes than announced; upload aborted"
        _path, handle, digest = self.part
        handle.write(data)
        digest.update(data)
        self.received += len(data)
        return None

    def end(self, connection):
        """The last chunk is in: keep, export and load, in the background."""
        if connection is not self.owner or self.phase != "receiving":
            return "no upload in progress from this page"
        if self.received != self.size:
            got = self.received
            self.abort(connection, "incomplete upload")
            return f"received {got} of {self.size} bytes"
        path, handle, digest = self.part
        handle.close()
        self.part = None
        self.owner = None
        self.phase = "saving"
        self.task = asyncio.get_running_loop().create_task(
            self._finish(path, digest.hexdigest()))
        return None

    def abort(self, connection, why):
        """Drop the upload of `connection` (a page that closed, or an error)."""
        if connection is not self.owner or self.phase != "receiving":
            return
        path, handle, _digest = self.part
        handle.close()
        try:
            os.remove(path)
        except OSError:
            pass
        self.part = None
        self.owner = None
        self._done(False, f"upload of {self.name} stopped: {why}", None)

    # ------------------------------------------------------------ the work

    def _done(self, ok, message, path):
        self.phase = "idle"
        self.result_id += 1
        self.result = (ok, message, path)
        self._log(message)

    async def _finish(self, part, digest):
        try:
            ok, message, path = await self._keep_export_load(part, digest)
        except Exception as error:      # noqa: BLE001 -- report, never hang
            ok, message, path = False, f"import failed: {error}", None
        finally:
            if os.path.exists(part):
                os.remove(part)
        self._done(ok, message, path)

    async def _keep_export_load(self, part, digest):
        loop = asyncio.get_running_loop()
        stem = safe_stem(self.name)
        if self.name.lower().endswith(".pth"):
            # The checkpoint itself is kept too: it is what the export
            # comes from, and what a later export with other options would.
            pth = await loop.run_in_executor(
                None, same_file, self.pth_dir, ".pth", self.size, digest)
            if pth is None:
                os.makedirs(self.pth_dir, exist_ok=True)
                pth = unique_dir(self.pth_dir, stem, ".pth")
                os.replace(part, pth)
            onnx = os.path.join(unique_dir(self.onnx_dir, stem), "policy.onnx")
            self.phase = "exporting"
            ok, why = await self._export(pth, onnx)
            if not ok:
                shutil.rmtree(os.path.dirname(onnx), ignore_errors=True)
                return False, f"export of {self.name} failed: {why}", None
            kept = f"{os.path.relpath(pth, self.checkpoint)} exported to " \
                   f"{os.path.relpath(onnx, self.checkpoint)}"
        else:
            onnx = await loop.run_in_executor(
                None, same_file, self.onnx_dir, ".onnx", self.size, digest)
            if onnx is not None:
                kept = f"already in checkpoint/ as {os.path.relpath(onnx, self.checkpoint)}"
            else:
                folder = unique_dir(self.onnx_dir, stem)
                os.makedirs(folder)
                onnx = os.path.join(folder, stem + ".onnx")
                os.replace(part, onnx)
                kept = f"saved as {os.path.relpath(onnx, self.checkpoint)}"

        self.phase = "loading"
        ok, why = await loop.run_in_executor(None, self._load, onnx)
        if not ok:
            return True, f"{kept}; not loaded: {why}", onnx
        return True, f"{kept}; loading it", onnx

    def _export_command(self, pth, onnx):
        """The uv environment pyproject.toml describes, which has torch."""
        venv = os.path.join(self.workspace, ".venv", "bin", "python")
        script = [self.export_script, "--checkpoint", pth, "--output", onnx]
        if os.path.isfile(venv):
            return [venv, *script]
        uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
        if os.path.isfile(uv):
            return [uv, "run", "--project", self.workspace, "python", *script]
        return None

    async def _export(self, pth, onnx):
        command = self._export_command(pth, onnx)
        if command is None:
            return False, "no export environment: run ./utils/setup_env.sh in the workspace"
        self.lines = [" ".join(os.path.basename(c) for c in command[:2]) + " ..."]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        # ROS's sourced paths would put the system's packages first.
        env.pop("PYTHONPATH", None)
        process = await asyncio.create_subprocess_exec(
            *command, cwd=self.workspace, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

        async def read():
            async for raw in process.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line.strip(" =-"):
                    self.lines.append(line)
                    del self.lines[:-200]
        try:
            await asyncio.wait_for(asyncio.gather(read(), process.wait()),
                                   EXPORT_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False, f"took longer than {EXPORT_TIMEOUT_S:.0f} s"
        if process.returncode != 0 or not os.path.isfile(onnx):
            last = next((line for line in reversed(self.lines) if line.strip()),
                        "no output")
            return False, f"exit {process.returncode}: {last}"
        return True, None
