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

"""The browser half: static files over HTTP, state and commands over a socket.

Both share one port, because the page and its socket then share an origin and
the only thing to configure is the URL. Every request arrives at
`_serve_file`; a WebSocket upgrade is passed through, anything else is a file.
"""

import asyncio
import base64
import json
import os
import signal
import threading
import time
from email.utils import formatdate
from http import HTTPStatus

import websockets
from websockets.asyncio.server import serve

from . import camera, mujoco_visuals
from .model_import import ModelImport
from .joints import JOINT_NAMES
from .static_cache import StaticCache
from .supervisor import MODES, Supervisor, SupervisorError

# After the controller's process exists, its parameter services still take a
# moment to advertise; and joint_states only flows once the CAN threads have
# heard from the motors.
CONTROLLER_READY_TIMEOUT = 15.0
# Long enough for MuJoCo to load its model and start answering, which is the
# slow case; a failure only costs this much waiting.
FEEDBACK_TIMEOUT = 15.0


def _expected_motors(arms):
    """How many motors the controller drives, given its `arms` setting.

    Eight per arm, the gripper included, so a controller launched with
    `arms:=left` is complete at eight rather than looking half broken.
    """
    return len(JOINT_NAMES) // 2 if arms in ("left", "right") else len(JOINT_NAMES)


def set_header(headers, name, value):
    """Replace a header rather than add one.

    websockets' Headers is a multidict, so plain assignment appends. The
    response respond() hands back already carries its own Content-Type and
    Content-Length, and a browser rejects a reply that repeats either.
    """
    try:
        del headers[name]
    except KeyError:
        pass
    headers[name] = value


def default_web_root():
    """The page's files, installed beside this package or in its source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory("openarm_ui"), "web")
        if os.path.isdir(share):
            return share
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


class Dashboard:
    """The browser half: static files over HTTP, state and commands over WS."""

    def __init__(self, node):
        self.node = node
        self.clients = set()
        # The Recorder writing an episode, and the one armed for the next
        # (recorder.py): Start recording then only tells it to begin.
        self.recorder = None
        self.standby = None
        self._record_lock = threading.Lock()
        self.recording_started = None   # monotonic, so the page can show elapsed
        self.episode = None
        # The cameras the next episode records, as the page picks them;
        # the armed recorder follows them (recorder.py).
        self.record_cameras = ("chest",)
        # The episode being compressed after its recording stopped (see
        # _watch_compression), and how the last one ended, for the page:
        # nothing new is recorded until the compression is done, so it
        # never takes CPU from an episode being recorded.
        self.compressing = None         # {"episode", "final", "partial", "since"}
        self.last_saved = None          # {"id", "ok", "message"}
        self.web_root = node.web_root or default_web_root()
        self.static = StaticCache()
        # The simulator's meshes and materials, so the 3D view looks like
        # MuJoCo (mujoco_visuals.py); None leaves the page on its COLLADA.
        self.mujoco_manifest, self.mujoco_root = mujoco_visuals.manifest()
        if self.mujoco_manifest is None:
            node.get_logger().warning(
                "openarm_mujoco's MJCF not found: the 3D view keeps the "
                "URDF's black-and-white meshes")
        self.supervisor = Supervisor(node.get_logger())
        # The mode a connect is working towards, so the page can say what is
        # happening while the controller is down between stop and start.
        self.connecting = None
        # One mode change at a time: the second would stop the controller the
        # first is waiting for.
        self._connect_lock = threading.Lock()
        # A model picked on a page (Load in the ACT card): uploaded here,
        # exported when it is a .pth, kept under checkpoint/ and loaded.
        # The workspace is three levels up from this file's real path.
        workspace = os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))
        self.importer = ModelImport(workspace, self._load_imported,
                                    node.get_logger().info)
        # connection -> the camera streams that page is showing. Empty for a
        # page with its camera off, which then costs nothing.
        self.camera_views = {}
        # connection -> {policy: the last trajectory sequence sent to it}, so
        # each page gets every trajectory once, a page opened later included.
        self._trajectory_sent = {}

    def status(self):
        """The node's status, plus what this dashboard is running."""
        status = self.node.status()
        status["sim_up"] = self.supervisor.sim_running()
        status["motors_expected"] = _expected_motors(status.get("arms"))
        status["connecting"] = self.connecting
        status["modes"] = sorted(MODES)
        status["record_cameras"] = list(self.record_cameras)
        return status

    # -- HTTP ---------------------------------------------------------------

    async def _serve_file(self, connection, request):
        """websockets hands every request here, so one port serves both.

        Returning None lets the handshake go ahead; anything else is sent as
        the reply. The page and the socket share an origin this way, which
        keeps the URL in the browser the only thing to configure.
        """
        if "websocket" in request.headers.get("Upgrade", "").lower():
            return None

        path = request.path.split("?", 1)[0]
        if path in ("", "/"):
            path = "/index.html"
        if path == "/mujoco/visuals.json":
            if self.mujoco_manifest is None:
                return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
            response = connection.respond(
                HTTPStatus.OK, json.dumps(self.mujoco_manifest))
            set_header(response.headers, "Content-Type", "application/json")
            set_header(response.headers, "Cache-Control", "no-cache")
            return response
        # The simulator's OBJ parts, from openarm_mujoco's meshes/visual/.
        mujoco = path.startswith("/mujoco/") and self.mujoco_root is not None
        if mujoco:
            path = path[len("/mujoco"):]
        # normpath, not realpath: it collapses any ".." so the request cannot
        # climb out of web/, while leaving symlinks alone. A colcon
        # --symlink-install tree is all symlinks back to the source, and
        # resolving them would put every file "outside" the root.
        root = os.path.abspath(self.mujoco_root if mujoco else self.web_root)
        target = os.path.normpath(os.path.join(root, path.lstrip("/")))
        inside = target == root or target.startswith(root + os.sep)
        if not inside or not os.path.isfile(target):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

        # Read and gzip off the event loop: a 6 MB mesh read inline would
        # stall every page's state stream while it happened.
        loop = asyncio.get_running_loop()
        try:
            entry = await loop.run_in_executor(None, self.static.get, target)
        except OSError:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

        # The meshes and vendored libraries change only with a git pull, so
        # the browser may keep them a day without asking. The app's own
        # code is revalidated every time -- an edit shows on the next
        # reload -- which with the ETag costs a 304, not the file.
        relative = os.path.relpath(target, root)
        if mujoco or relative.startswith(("assets" + os.sep, "vendor" + os.sep)):
            cache_control = "public, max-age=86400"
        else:
            cache_control = "no-cache"

        if entry.etag in request.headers.get("If-None-Match", ""):
            response = connection.respond(HTTPStatus.NOT_MODIFIED, "")
            response.body = b""
            set_header(response.headers, "Content-Length", "0")
        else:
            accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
            if entry.gzipped is not None and accepts_gzip:
                body = entry.gzipped
            else:
                body = entry.body()
            # respond() only builds a text body, so take the response it
            # returns and put the bytes in; meshes must not be decoded.
            response = connection.respond(HTTPStatus.OK, "")
            response.body = body
            set_header(response.headers, "Content-Length", str(len(body)))
            set_header(response.headers, "Content-Type", entry.content_type)
            if body is entry.gzipped:
                set_header(response.headers, "Content-Encoding", "gzip")
        if entry.gzipped is not None:
            set_header(response.headers, "Vary", "Accept-Encoding")
        set_header(response.headers, "ETag", entry.etag)
        set_header(response.headers, "Last-Modified",
                   formatdate(entry.last_modified, usegmt=True))
        set_header(response.headers, "Cache-Control", cache_control)
        return response

    def _warm_static(self):
        started = time.monotonic()
        assets = os.path.join(self.web_root, "assets")
        if self.mujoco_root is not None:
            # The page draws the simulator's OBJ parts, not the COLLADA.
            count = self.static.warm(assets, suffixes=(".urdf",))
            count += self.static.warm(self.mujoco_root, suffixes=(".obj",))
        else:
            count = self.static.warm(assets)
        files, held = self.static.stats()
        self.node.get_logger().info(
            f"cached {count} meshes in {time.monotonic() - started:.1f} s "
            f"({held / 1e6:.0f} MB in memory, gzipped)")

    # -- WebSocket ----------------------------------------------------------

    async def handle(self, connection):
        self.clients.add(connection)
        try:
            await connection.send(json.dumps({
                "type": "hello",
                "joint_names": JOINT_NAMES,
                "status": self.status(),
                "state": self.node.seed_targets_from_state(),
            }))
            async for raw in connection:
                if isinstance(raw, bytes):
                    # A piece of the model being uploaded, in order. Each is
                    # acknowledged and the page keeps only two in flight:
                    # sent faster than they are written, they would fill
                    # this socket's queue, stop it reading, and the keepalive
                    # pong stuck behind them would close the connection.
                    problem = self.importer.chunk(connection, raw)
                    reply = {"type": "model_upload_ack",
                             "received": self.importer.received} \
                        if problem is None else {
                            "type": "action", "action": "model_import",
                            "ok": False, "message": problem}
                else:
                    reply = await self._dispatch(raw, connection)
                if reply is not None:
                    await connection.send(json.dumps(reply))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.importer.abort(connection, "the page closed")
            self.clients.discard(connection)
            self._trajectory_sent.pop(connection, None)
            self._set_camera_view(connection, [])

    async def _dispatch(self, raw, connection):
        """Run one page request. Service calls block, so they go to a thread."""
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "error", "message": "malformed request"}

        action = message.get("action")
        loop = asyncio.get_running_loop()

        if action == "set_joint":
            problem = self.node.set_target(message.get("name", ""),
                                           message.get("position", 0.0))
            return None if problem is None else {"type": "error",
                                                 "message": problem}

        if action == "set_joints":
            problems = [self.node.set_target(name, position) for name, position
                        in (message.get("positions") or {}).items()]
            problem = next((p for p in problems if p is not None), None)
            return None if problem is None else {"type": "error",
                                                 "message": problem}

        if action == "zero_pose":
            ok, reason = await loop.run_in_executor(None, self.node.go_to_zero)
            return {"type": "action", "action": "zero_pose", "ok": ok,
                    "message": reason}
        if action == "zero_stop":
            ok, reason = await loop.run_in_executor(None, self.node.zero_stop)
            return {"type": "action", "action": "zero_pose", "ok": ok,
                    "message": reason}

        if action == "status":
            return {"type": "status",
                    "status": await loop.run_in_executor(None, self.status)}

        if action == "connect":
            self.node.zero_cancel("reconnecting")
            return await loop.run_in_executor(
                None, self._connect, message.get("mode"))

        if action == "disconnect":
            self.node.zero_cancel("disconnecting")
            ok, reason = await loop.run_in_executor(
                None, self.node.set_remote_parameter,
                self.node.controller_node, "motors_enabled", False)
            return {"type": "action", "action": "disconnect", "ok": ok,
                    "message": "motors disabled, the arm is limp"
                               if ok else reason}

        if action == "set_source":
            source = message.get("source", "none")
            if source == self.node.source_name:
                # The sliders' targets start where the robot is now, not where
                # it was when the page opened.
                self.node.seed_targets_from_state()
            ok, reason = await loop.run_in_executor(
                None, self.node.set_proxy_source, source)
            return {"type": "action", "action": action, "ok": ok,
                    "message": reason, "source": source,
                    "resync": ok and source == self.node.source_name}

        if action == "set_gain":
            joint = message.get("joint")
            field = message.get("field")
            if joint not in JOINT_NAMES or field not in ("kp", "kd", "speed", "effort"):
                return {"type": "error",
                        "message": f"no gain {field!r} on joint {joint!r}"}
            profile = message.get("profile")
            if not profile:
                return {"type": "error", "message": "no gain profile given"}
            ok, reason = await loop.run_in_executor(
                None, self.node.set_remote_parameter,
                self.node.controller_node, f"gains.{profile}.{joint}.{field}",
                float(message.get("value", 0.0)))
            return {"type": "action", "action": action, "ok": ok,
                    "message": f"{profile}/{joint}.{field}: {reason}"}

        if action == "set_interpolator":
            # quintic or filter4, and filter4's w: the controller switches
            # where each joint stands, so nothing jumps.
            replies = []
            for name, value in (("interpolator", message.get("kind")),
                                ("interp_w", message.get("w"))):
                if value is None:
                    continue
                if name == "interp_w":
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        return {"type": "action", "action": action, "ok": False,
                                "message": "w must be a number"}
                ok, reason = await loop.run_in_executor(
                    None, self.node.set_remote_parameter,
                    self.node.controller_node, name, value)
                if not ok:
                    return {"type": "action", "action": action, "ok": False,
                            "message": f"{name}: {reason}"}
                replies.append(f"{name} {value}")
            return {"type": "action", "action": action, "ok": True,
                    "message": ", ".join(replies) or "nothing to change"}

        if action == "set_profile":
            ok, reason = await loop.run_in_executor(
                None, self.node.set_remote_parameter,
                self.node.controller_node, "active_profile",
                message.get("profile", "default"))
            return {"type": "action", "action": action, "ok": ok, "message": reason}

        if action == "camera":
            streams = (message.get("streams") or []) if message.get("on") else []
            self._set_camera_view(connection, streams)
            return {"type": "camera_topics", "topics": self.node.camera_topics}
        if action == "camera_switch":
            return await loop.run_in_executor(None, self._camera_switch)
        if action == "record_cameras":
            return await loop.run_in_executor(
                None, self._set_record_cameras, message.get("cameras"))

        if action.startswith("act_") or action == "plan_trajectory":
            # The ACT and SmolVLA cards send the same actions, each naming
            # its policy; a page from before SmolVLA names none: ACT.
            policy = self.node.policy(message.get("policy") or "act")
            if policy is None and action != "plan_trajectory":
                return {"type": "action", "action": "act", "ok": False,
                        "message": f"no policy {message.get('policy')!r}"}
        if action == "act_execute":
            return await loop.run_in_executor(None, self._act_execute, policy)
        if action == "act_stop":
            return await loop.run_in_executor(None, self._act_stop, policy)
        if action == "act_reset":
            return await loop.run_in_executor(None, self._act_reset, policy)
        if action == "act_start":
            return await loop.run_in_executor(None, self._act_start, policy)
        if action == "act_param":
            return await loop.run_in_executor(
                None, self._act_param, policy, message.get("name"),
                message.get("value"))
        if action == "plan_trajectory":
            return await loop.run_in_executor(
                None, self._plan_trajectory, message.get("rows"),
                message.get("policy"))
        if action == "model_upload_begin":
            problem = self.importer.begin(connection, message.get("name"),
                                          message.get("size"))
            # The page sends the chunks only after this answer.
            return {"type": "model_upload", "ok": problem is None,
                    "message": problem or "send the file"}
        if action == "model_upload_end":
            problem = self.importer.end(connection)
            return {"type": "action", "action": "model_import",
                    "ok": problem is None,
                    "message": problem or "file received"}
        if action == "act_model":
            return await loop.run_in_executor(
                None, self._act_model, message.get("path"))
        if action == "act_generate":
            ok, reason = await loop.run_in_executor(None, policy.generate)
            # Success is only "started": the result comes with the status.
            return {"type": "action", "action": "act", "ok": ok,
                    "message": f"{policy.label}: generating a trajectory…" if ok
                               else f"{policy.label}: cannot generate: {reason}"}

        if action == "record_start":
            return await loop.run_in_executor(
                None, self._record_start, message.get("episode"),
                message.get("cameras"))
        if action == "record_stop":
            return await loop.run_in_executor(None, self._record_stop)

        return {"type": "error", "message": f"unknown action {action!r}"}

    # -- connecting ---------------------------------------------------------

    def _connect(self, mode):
        """Bring up `mode` and arm the motors.

        Real and sim differ in which bus the controller drives, and that is
        fixed at launch, so changing mode restarts the controller. The
        simulator is started or stopped to match, because it is the thing
        answering on the virtual bus.
        """
        def done(ok, message):
            return {"type": "action", "action": "connect", "ok": ok,
                    "mode": mode, "message": message}

        if mode not in MODES:
            return done(False, f"unknown mode {mode!r}; "
                               f"expected one of {', '.join(sorted(MODES))}")

        if not self._connect_lock.acquire(blocking=False):
            return done(False, f"already connecting to {self.connecting}")
        self.connecting = mode
        try:
            return self._switch(mode, done)
        except SupervisorError as error:
            return done(False, str(error))
        finally:
            self.connecting = None
            self._connect_lock.release()

    def _switch(self, mode, done):
        """The mode change itself, with the lock held."""
        before = self.node.status()
        plan = self.supervisor.plan(mode, before.get("mode"))
        label = MODES[mode]["label"]

        # Stop the controller before touching the simulator: a controller
        # left driving a bus whose other end is going away spends the gap
        # reporting motors that stopped answering.
        if plan["restart_controller"]:
            self.supervisor.stop_controller()
        if plan["stop_sim"]:
            self.supervisor.stop_sim()
        if plan["start_sim"]:
            self.supervisor.start_sim()
        if plan["restart_controller"]:
            self.supervisor.start_controller(mode)
            if not self._wait_for_controller():
                return done(False, "the controller started but its parameters "
                                   "never answered")

        after = self.node.status()
        reached = after.get("mode")
        if reached != mode:
            return done(False, f"asked for {mode}, but the controller came up "
                               f"on {', '.join(after.get('interfaces') or ['nothing'])}")

        buses = ", ".join(after.get("interfaces") or [])
        expected = _expected_motors(after.get("arms"))

        # Wait for the motors before arming them, not after. Enabling is
        # written to the bus and nothing acknowledges it, so an enable sent
        # while the other end is still coming up is simply lost -- MuJoCo
        # needs a moment after its process exists before it answers, and the
        # controller starts faster than that.
        answering = self._wait_for_feedback(expected)
        if not answering:
            missing = ("Is the arm powered and wired?" if mode == "real"
                       else "Is MuJoCo running?")
            return done(False, f"{label} is up on {buses}, but no motor "
                               f"answered. {missing}")
        if answering < expected:
            # One arm unpowered looks exactly like this. Reporting it in
            # green and leaving it to be discovered when half the robot does
            # not move would be worse than refusing to arm.
            return done(False, f"{label} is up on {buses}, but only "
                               f"{answering} of {expected} motors answered: "
                               f"{expected - answering} are silent")

        ok, reason = self.node.set_remote_parameter(
            self.node.controller_node, "motors_enabled", True)
        if not ok:
            return done(False, f"{label} is up, but the motors would not arm: "
                               f"{reason}")
        return done(True, f"connected to {label} on {buses} — "
                          f"{answering} motors answering")

    def _wait_for_controller(self):
        """Wait for the new controller's parameter services to answer."""
        deadline = time.monotonic() + CONTROLLER_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self.node.get_remote_parameters(
                    self.node.controller_node, ["left_interface"]):
                return True
            time.sleep(0.25)
        return False

    def _wait_for_feedback(self, expected):
        """Wait for motors to answer on CAN; returns how many do.

        Not the joint_states rate, and not how many joints it carries: the
        controller publishes either way, filling in zeros for a motor that
        has never replied, so a bus with nothing on it still looks alive from
        the outside. `motors_answering` is the controller's own count of
        motors replying, which is the thing that distinguishes a connection
        from a controller talking to itself.
        """
        deadline = time.monotonic() + FEEDBACK_TIMEOUT
        best = 0
        while time.monotonic() < deadline:
            reported = self.node.get_remote_parameters(
                self.node.controller_node, ["motors_answering"]) or {}
            answering = reported.get("motors_answering") or 0
            best = max(best, answering)
            # Replies arrive over a few control ticks, so settle on the full
            # set rather than reporting the first motor that answers.
            if answering >= expected:
                return answering
            time.sleep(0.2)
        return best

    # -- recording ----------------------------------------------------------

    def _arm_recorder(self):
        """Have a recorder armed for the next episode, on the cameras chosen
        for it (any thread). One armed for others is dismissed."""
        from .recorder import Recorder
        with self._record_lock:
            if self.standby is not None and self.standby.alive():
                if self.standby.cameras == self.record_cameras:
                    return
                self.standby.close()
            try:
                self.standby = Recorder(cameras=self.record_cameras)
            except OSError as error:
                self.standby = None
                self.node.get_logger().warning(f"cannot arm a recorder: {error}")

    def _record_cameras_problem(self, cameras):
        """-> (cameras in role order, None) or (None, why not)."""
        if not isinstance(cameras, (list, tuple)) or not cameras:
            return None, "record at least one camera"
        unknown = [name for name in cameras if name not in self.node.cameras]
        if unknown:
            return None, f"no camera {unknown[0]!r}; there are {', '.join(self.node.cameras)}"
        return tuple(name for name in self.node.cameras if name in cameras), None

    def _camera_switch(self):
        """Chest shows the front camera, or the other way round: swap them.
        In cameras.yaml, for the next launch, and at once in the running
        cameras -- each preprocessor reads the other one's driver."""
        def done(ok, message):
            return {"type": "action", "action": "camera_switch", "ok": ok,
                    "message": message}
        if self.recording():
            return done(False, "stop the recording first")
        from openarm_camera import cameras as camera_config
        saved = None
        try:
            if camera_config.config_path().is_file():
                swapped = camera_config.swap()
                saved = ", ".join(f"{role} {entry['serial']}"
                                  for role, entry in swapped.items())
        except (OSError, ValueError) as error:
            return done(False, f"cannot swap {camera_config.config_path()}: {error}")
        nodes = self.node.preprocessor_nodes
        sources = {camera: (self.node.get_remote_parameters(node, ["source"]) or {}).get("source")
                   for camera, node in nodes.items()}
        if len(nodes) == 2 and all(sources.values()):
            (first, a), (second, b) = sources.items()
            ok_a, why_a = self.node.set_remote_parameter(nodes[first], "source", b)
            ok_b, why_b = self.node.set_remote_parameter(nodes[second], "source", a)
            if ok_a and ok_b:
                return done(True, "chest and front switched" +
                            (f"; saved ({saved})" if saved else
                             "; not saved: no cameras.yaml (./setup_camera.sh)"))
            return done(False, f"cannot switch the running cameras: {why_a if not ok_a else why_b}")
        if saved:
            return done(True, f"saved ({saved}); a camera is virtual, so it shows "
                              f"when the cameras restart (./run_main.sh)")
        return done(False, "nothing to switch: both cameras are virtual and there "
                           "is no cameras.yaml (./setup_camera.sh)")

    def _set_record_cameras(self, cameras):
        """The page picked the cameras to record: arm a recorder on them."""
        def done(ok, message):
            return {"type": "action", "action": "record_cameras", "ok": ok,
                    "message": message, "cameras": list(self.record_cameras)}
        chosen, problem = self._record_cameras_problem(cameras)
        if problem:
            return done(False, problem)
        changed = chosen != self.record_cameras
        self.record_cameras = chosen
        if changed and not self.recording():
            self._arm_recorder_soon()
        return done(True, "recording " + " + ".join(chosen))

    def _arm_recorder_soon(self):
        threading.Thread(target=self._arm_recorder, daemon=True).start()

    def _record_start(self, episode, cameras=None):
        """Begin an episode on the armed recorder: tens of milliseconds,
        instead of the two seconds a fresh process takes to start.
        `cameras`: the ones to record, else those picked before."""
        from .recorder import Recorder

        def done(ok, message):
            return {"type": "action", "action": "record_start", "ok": ok,
                    "message": message}

        compressing = self.compressing
        if compressing is not None:
            return done(False, f"still compressing {compressing['episode']}: "
                               f"record when it is done")
        if cameras is not None:
            chosen, problem = self._record_cameras_problem(cameras)
            if problem:
                return done(False, problem)
            self.record_cameras = chosen
        # A camera nobody publishes would leave the recorder waiting for
        # its first frame, writing nothing.
        sources = self.node.camera_sources()
        silent = [name for name in self.record_cameras if not sources.get(name)]
        if silent:
            return done(False, f"no frames from {', '.join(silent)}: is "
                               f"openarm_camera running? (./setup_camera.sh status)")
        with self._record_lock:
            if self.recorder is not None and self.recorder.recording:
                return done(False, "already recording")
            recorder, self.standby = self.standby, None
            if recorder is not None and recorder.cameras != self.record_cameras:
                recorder.close()        # armed for other cameras
                recorder = None
            if recorder is None or not recorder.alive():
                try:
                    # None armed for these cameras: the slow way.
                    recorder = Recorder(cameras=self.record_cameras)
                except OSError as error:
                    return done(False, str(error))
            try:
                name = recorder.start(episode)
            except RuntimeError as error:
                recorder.close()
                self._arm_recorder_soon()
                return done(False, str(error))
            self.recorder = recorder
            self.recording_started = time.monotonic()
            self.episode = name
        virtual = [camera for camera in recorder.cameras if sources.get(camera) == "virtual"]
        self.node.get_logger().info(f"recording {name}: {', '.join(recorder.cameras)}")
        # The next episode's recorder is armed when this one stops, not now:
        # its start-up (imports, then every camera stream) would take CPU
        # from the episode being recorded, and two cameras leave little.
        return done(True, f"{name} ({' + '.join(recorder.cameras)})" +
                    (f" -- {', '.join(virtual)} virtual: black frames" if virtual else ""))

    def _record_stop(self):
        """Save the episode. Waits for its last samples to reach the disk,
        on an executor thread, so the page keeps its state stream."""
        with self._record_lock:
            recorder = self.recorder
            if recorder is None or not recorder.recording:
                return {"type": "action", "action": "record_stop", "ok": False,
                        "message": "not recording"}
            self.recorder = None
            seconds = self.recording_seconds()
            self.recording_started = None
        saved = recorder.stop()
        self.node.get_logger().info(f"recording stopped after {seconds:.1f} s")
        if saved:
            # The recorder has written <episode>.partial.hdf5 and started its
            # compression; the page waits for the final file.
            self.compressing = {
                "episode": recorder.episode, "final": recorder.path,
                "partial": recorder.path[:-len(".hdf5")] + ".partial.hdf5",
                "since": time.monotonic()}
            threading.Thread(target=self._watch_compression,
                             args=(self.compressing,), daemon=True).start()
        # The next episode's, armed while this one compresses: ready by the
        # time it is done.
        self._arm_recorder_soon()
        return {"type": "action", "action": "record_stop", "ok": saved,
                "message": f"{recorder.episode}: {seconds:.1f} s recorded, compressing…"
                           if saved else f"{self.episode}: the recorder did not "
                                         f"finish; its partial file is kept"}

    @staticmethod
    def _compressor_running(partial):
        """Whether an episode_compress process is working on `partial`."""
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    command = f.read().decode(errors="replace")
            except OSError:
                continue
            if "episode_compress" in command and partial in command:
                return True
        return False

    def _watch_compression(self, job):
        """Until the episode's final file is in place (or its compression
        has died): then say how it went -- samples, rate, late samples and
        gaps -- and let the next recording start."""
        final, partial = job["final"], job["partial"]
        relative = os.path.relpath(final, os.path.dirname(os.path.dirname(os.path.dirname(final))))
        while True:
            time.sleep(0.5)
            if os.path.exists(final) and not os.path.exists(partial):
                try:
                    import h5py
                    with h5py.File(final, "r") as f:
                        samples = int(f.attrs.get("samples", 0))
                        duration = float(f.attrs.get("duration_seconds", 0.0))
                        late = int(f.attrs.get("late_samples", 0))
                        gaps = int(f.attrs.get("gaps", 0))
                    rate = (samples - 1) / duration if duration > 0 else 0.0
                    extras = ([f"{late} filled late"] if late else []) + \
                             ([f"{gaps} gaps"] if gaps else [])
                    ok, message = True, (
                        f"saved {relative}: {samples} samples, {rate:.1f} Hz"
                        + (f" ({', '.join(extras)})" if extras else ""))
                except Exception as error:      # noqa: BLE001 -- still saved
                    ok, message = True, f"saved {relative} ({error})"
                break
            if not os.path.exists(partial) and not os.path.exists(final):
                ok, message = False, f"{job['episode']}: no samples were recorded"
                break
            # The recorder starts the compression as it exits; give it a
            # moment to appear before calling its absence a failure.
            if time.monotonic() - job["since"] > 5.0 and \
                    not self._compressor_running(partial):
                ok, message = False, (
                    f"{job['episode']}: compression stopped; the partial file is kept "
                    f"(python3 -m openarm_ui.episode_compress {partial})")
                break
        serial = (self.last_saved or {}).get("id", 0) + 1
        self.last_saved = {"id": serial, "ok": ok, "message": message}
        if self.compressing is job:
            self.compressing = None
        self.node.get_logger().info(message)

    def recorder_ready(self):
        """Whether Start recording would begin at once: a recorder armed on
        the chosen cameras, streams followed, nothing compressing."""
        standby = self.standby
        return (self.compressing is None and not self.recording() and
                standby is not None and standby.alive() and
                standby.ready.is_set() and standby.cameras == self.record_cameras)

    def shutdown_recorders(self):
        """On exit: save an episode in progress, dismiss the armed one."""
        if self.recorder is not None and self.recorder.alive():
            self.recorder.process.send_signal(signal.SIGINT)
        if self.standby is not None:
            self.standby.close()

    def recording(self):
        return self.recorder is not None and self.recorder.recording

    def recording_seconds(self):
        """How long the episode has been running, 0 when not recording."""
        if self.recording_started is None:
            return 0.0
        return time.monotonic() - self.recording_started

    # -- policies (ACT, SmolVLA) --------------------------------------------
    #
    # Each card drives its own node and commands the arm as its own proxy
    # source (act, smolvla); `policy` is the node.PolicyLink.

    @staticmethod
    def _policy_reply(ok, message):
        return {"type": "action", "action": "act", "ok": ok, "message": message}

    def _act_execute(self, policy):
        """Let the previewed policy drive: execute on the node, then the proxy.

        In that order, because the node refuses when it has nothing current
        to execute, and the proxy should not be switched to a source that
        is not going to say anything. Undone if the proxy refuses.
        """
        ok, reason = self.node.set_remote_parameter(
            policy.node_name, "mode", "execute")
        if not ok:
            return self._policy_reply(
                False, f"{policy.label}: cannot run the trajectory: {reason}")
        ok, reason = self.node.set_proxy_source(policy.name)
        if not ok:
            self.node.set_remote_parameter(policy.node_name, "mode", "preview")
            return self._policy_reply(
                False, f"the proxy would not take {policy.label}: {reason}")
        where = {"real": "the real robot", "sim": "the simulator"}.get(
            self.node.status().get("mode"), "the arm")
        return self._policy_reply(True, f"{policy.label}: running on {where}")

    def _plan_trajectory(self, rows, name=None):
        """A trajectory the page planned to its IK goal, for a policy node to
        draw and run: the one named (the card shown), else ACT if it runs,
        else SmolVLA.

        The proxy is switched to its source as well, which commands nothing
        until Run, so the card -- Run, Stop, Reset -- and the drawing show up.
        """
        width = len(JOINT_NAMES)
        if not isinstance(rows, list) or len(rows) < 2 or not all(
                isinstance(row, list) and len(row) == width and
                all(isinstance(v, (int, float)) for v in row) for row in rows):
            return self._policy_reply(
                False, f"a plan is a list of rows of {width} numbers")
        policy = self.node.policy(name) if name else None
        if policy is None or not policy.running():
            policy = next((p for p in self.node.policies.values() if p.running()),
                          self.node.policy("act"))
        ok, reason = policy.plan(rows)
        if not ok:
            return self._policy_reply(False, f"cannot plan: {reason}")
        ok, reason = self.node.set_proxy_source(policy.name)
        if not ok:
            return self._policy_reply(
                False, f"planned, but the proxy would not take {policy.label}: {reason}")
        return self._policy_reply(
            True, f"trajectory sent to {policy.label}: {len(rows)} steps; check it, then Run")

    def _act_reset(self, policy):
        """Ease the arm to the zero pose, grippers open (reset_pose).

        The node first, as for a run: it refuses when there is nowhere to go
        back to. Then the proxy, which still has the policy after a run; a
        switch made only now drops the first commands of the ease, which
        starts where the arm is, so nothing jumps.
        """
        ok, reason = policy.reset()
        if not ok:
            return self._policy_reply(False, f"{policy.label}: cannot reset: {reason}")
        ok, reason = self.node.set_proxy_source(policy.name)
        if not ok:
            self.node.set_remote_parameter(policy.node_name, "mode", "preview")
            return self._policy_reply(
                False, f"the proxy would not take {policy.label}: {reason}")
        return self._policy_reply(True, f"{policy.label}: going to the zero pose")

    # What the page may tune on a policy node while it runs, and how each
    # arrives: the others stay in act.yaml / smolvla.yaml.
    ACT_PARAMS = {"vel_scale": float, "ensemble_k": float, "loop_mode": str,
                  "execute_timeout": float}

    def _act_start(self, policy):
        """Run the policy continuously, previewing: nothing reaches the arm
        until Execute."""
        ok, reason = self.node.set_remote_parameter(
            policy.node_name, "running", True)
        return self._policy_reply(
            ok, f"{policy.label} running: preview" if ok
                else f"cannot start {policy.label}: {reason}")

    def _act_model(self, path):
        """Load another of openarm_act's models. Only one it lists itself --
        the .onnx files under checkpoint/onnx/ -- not any path a page names.
        The load runs on the node; how it went comes with its status."""
        status = self.node.policy("act").snapshot()["status"]
        if status is None:
            return self._policy_reply(False, "ACT is not running")
        names = {model["path"]: model["name"]
                 for model in status.get("models") or []}
        if path not in names:
            return self._policy_reply(
                False, f"not a model under checkpoint/onnx/: {path}")
        ok, reason = self.node.set_remote_parameter(
            self.node.act_node, "checkpoint", path)
        return self._policy_reply(ok, f"loading {names[path]}…" if ok
                                  else f"cannot load: {reason}")

    def _load_imported(self, path):
        """Load an imported .onnx into openarm_act; -> (ok, reason).
        The path is one this dashboard just put under checkpoint/onnx/."""
        if not self.node.policy("act").running():
            return False, "ACT is not running; it lists the model when it is"
        return self.node.set_remote_parameter(
            self.node.act_node, "checkpoint", path)

    def _act_param(self, policy, name, value):
        kind = self.ACT_PARAMS.get(name)
        if kind is None:
            return self._policy_reply(
                False, f"{policy.label}: {name!r} cannot be set from the page")
        try:
            value = kind(value)
        except (TypeError, ValueError):
            return self._policy_reply(
                False, f"{name} must be a {kind.__name__}")
        ok, reason = self.node.set_remote_parameter(policy.node_name, name, value)
        return self._policy_reply(
            ok, f"{policy.label}: {name} = {value}" if ok
                else f"{policy.label}: {name}: {reason}")

    def _act_stop(self, policy):
        """Take the arm back from the policy; it holds its last target.

        The proxy first, so nothing more reaches the arm whatever the node
        does next; then the node stops executing and stops inferring.
        """
        source_ok, source_reason = self.node.set_proxy_source("none")
        mode_ok, _ = self.node.set_remote_parameter(
            policy.node_name, "mode", "preview")
        self.node.set_remote_parameter(policy.node_name, "running", False)
        if source_ok:
            return self._policy_reply(
                True, f"{policy.label} stopped; the arm holds" if mode_ok else
                      f"{policy.label} cut off at the proxy; the arm holds")
        return self._policy_reply(
            False, f"could not stop {policy.label} at the proxy: {source_reason}")

    def policies_payload(self):
        """Every policy's preview for one broadcast, {name: {status, pose}},
        and its trajectory {name: (seq, rows)} where it has one; the
        broadcast adds the rows for the pages still without them."""
        payload, trajectories = {}, {}
        for name, policy in self.node.policies.items():
            snapshot = policy.snapshot()
            payload[name] = {"status": snapshot["status"], "pose": snapshot["pose"]}
            if snapshot["trajectory"] is not None:
                trajectories[name] = snapshot["trajectory"]
        return payload, trajectories

    # -- camera -------------------------------------------------------------

    def _set_camera_view(self, connection, streams):
        """Record what one page shows, and subscribe to what all of them do."""
        streams = {name for name in streams if name in self.node.camera_topics}
        if streams:
            self.camera_views[connection] = streams
        else:
            self.camera_views.pop(connection, None)
        self.node.want_camera(set().union(*self.camera_views.values()))

    async def camera_broadcast(self):
        """Send each watching page its streams, at most camera_rate a second.

        A frame is encoded once however many pages show it, and only when a
        new one has arrived; a page is sent the topic, its age and its rate
        every tick, so it can say "no frames" instead of freezing.
        """
        period = 1.0 / max(0.5, self.node.camera_rate)
        loop = asyncio.get_running_loop()
        encoded = {}        # stream -> (msg it came from, payload)
        rate = {}           # stream -> (frames, monotonic) of the last estimate
        fps = {}
        while True:
            await asyncio.sleep(period)
            if not self.camera_views:
                encoded.clear()
                rate.clear()
                continue

            fresh = {}
            info = {}
            now = time.monotonic()
            for stream, (msg, age, count) in self.node.camera_snapshot().items():
                last_count, last_time = rate.get(stream, (count, now))
                if now - last_time >= 1.0:
                    fps[stream] = (count - last_count) / (now - last_time)
                    rate[stream] = (count, now)
                elif stream not in rate:
                    rate[stream] = (count, now)
                info[stream] = {"topic": self.node.camera_topics[stream],
                                "age": age, "fps": fps.get(stream, 0.0),
                                # openarm_camera's stand-in: black frames.
                                "virtual": bool(msg is not None and
                                                msg.header.frame_id.endswith("_virtual"))}
                if msg is None or encoded.get(stream, (None,))[0] is msg:
                    continue
                try:
                    jpeg, width, height = await loop.run_in_executor(
                        None, camera.encode_message, stream, msg,
                        self.node.camera_jpeg_quality,
                        *self.node.depth_range_mm)
                except Exception as error:      # a bad frame, not a dead view
                    info[stream]["error"] = str(error)
                    continue
                payload = {"jpeg": base64.b64encode(jpeg).decode("ascii"),
                           "width": width, "height": height}
                encoded[stream] = (msg, payload)
                fresh[stream] = payload

            for connection, streams in list(self.camera_views.items()):
                message = json.dumps({
                    "type": "camera",
                    "frames": {name: fresh[name] for name in streams
                               if name in fresh},
                    "info": {name: info[name] for name in streams
                             if name in info},
                })
                try:
                    await connection.send(message)
                except websockets.ConnectionClosed:
                    self._set_camera_view(connection, [])

    # -- broadcast ----------------------------------------------------------

    async def broadcast(self):
        """Push state to every page at telemetry_rate."""
        period = 1.0 / max(1.0, self.node.telemetry_rate)
        status_every = max(1, int(self.node.telemetry_rate))  # about once a second
        tick = 0
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(period)
            if not self.clients:
                tick = 0
                continue
            recording = self.recording()
            # The recorder can die on its own; the page should stop counting.
            if not recording and self.recording_started is not None:
                self.recording_started = None
            payload = {"type": "state", **self.node.state_snapshot(),
                       "recording": recording,
                       "recording_seconds": self.recording_seconds(),
                       "episode": self.episode if recording else None,
                       "recording_cameras": (list(self.recorder.cameras)
                                             if recording else None),
                       "recorder_ready": self.recorder_ready(),
                       "compressing": ({"episode": self.compressing["episode"],
                                        "seconds": time.monotonic() - self.compressing["since"]}
                                       if self.compressing else None),
                       "last_saved": self.last_saved}
            policies, trajectories = self.policies_payload()
            payload["policies"] = policies
            payload["zero"] = self.node.zero_snapshot()
            payload["model_import"] = self.importer.snapshot()
            # The proxy's selection as it changes, not once a second.
            payload["active_source"] = self.node.proxy_source
            tick += 1
            if tick % status_every == 0:
                payload["status"] = await loop.run_in_executor(None, self.status)
            message = json.dumps(payload)
            # Built only for the set of trajectories some page still needs.
            with_trajectories = {}
            for connection in list(self.clients):
                sent = self._trajectory_sent.setdefault(connection, {})
                missing = tuple(name for name, trajectory in trajectories.items()
                                if sent.get(name) != trajectory[0])
                text = message
                if missing:
                    if missing not in with_trajectories:
                        with_trajectories[missing] = json.dumps({**payload, "policies": {
                            name: ({**entry, "trajectory": trajectories[name][1]}
                                   if name in missing else entry)
                            for name, entry in policies.items()}})
                    text = with_trajectories[missing]
                    for name in missing:
                        sent[name] = trajectories[name][0]
                try:
                    await connection.send(text)
                except websockets.ConnectionClosed:
                    self.clients.discard(connection)
                    self._trajectory_sent.pop(connection, None)

    async def run(self):
        """Serve until asked to stop.

        `ros2 launch` ends a node with SIGINT and then SIGTERM. Without
        handlers asyncio ignores both, the process outlives its launch and
        keeps the port, so the next run cannot bind it.
        """
        loop = asyncio.get_running_loop()
        stop = loop.create_future()

        def request_stop():
            if not stop.done():
                stop.set_result(None)

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signal_number, request_stop)
            except NotImplementedError:      # not POSIX
                pass

        # Compress the meshes while nobody is waiting for them, so even the
        # first page after a start gets them from memory.
        threading.Thread(target=self._warm_static, daemon=True).start()
        # Arm the first recorder now: its two seconds of start-up are spent
        # before anyone presses Start recording.
        self._arm_recorder_soon()

        port = self.node.http_port
        serving = False
        try:
            # Frames up to 16 MiB: a model is uploaded in 4 MiB pieces. No
            # permessage-deflate: inflating hundreds of MB of weights, which
            # do not compress, cost more than the upload itself; the state
            # messages are small without it.
            async with serve(self.handle, "0.0.0.0", port,
                             process_request=self._serve_file,
                             max_size=16 * 1024 * 1024, compression=None):
                serving = True
                await self._serve_until(stop)
            # Whatever this dashboard started goes down with it, so closing
            # the page's launch does not leave a controller on the bus.
            self.supervisor.shutdown()
        except OSError as error:
            if serving:
                raise
            # Failing to bind is almost always a dashboard that is still
            # running. Say that, rather than ending on a traceback from deep
            # inside asyncio.
            raise RuntimeError(
                f"cannot serve on port {port}: {error}. Something else holds "
                f"it, most likely another dashboard. Find it with "
                f"`ss -lptn 'sport = :{port}'` and stop it, or start this one "
                f"with http_port:=<another port>.") from None

    async def _serve_until(self, stop):
        """Push state to the pages until the stop signal arrives."""
        tasks = [asyncio.create_task(self.broadcast()),
                 asyncio.create_task(self.camera_broadcast())]
        try:
            await stop
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
