// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Wiring: which message feeds which panel, and which button does what.

import { state } from "./state.js";
import { connect, on, send } from "./transport.js";
import {
  applyCurrent, applyTheme, initViewport, setPlannedVisible, setView,
} from "./viewport.js";
import { initTheme, onThemeChange } from "./theme.js";
import {
  buildSliders, execute, setAllLocks, stageHomePose, syncToCurrent,
  updatePlanHint,
} from "./panel-joints.js";
import {
  initInterpolator, renderGainsTable, renderInterpolator, renderProfiles,
  showGainsPane,
} from "./panel-gains.js";
import { renderTelemetry } from "./panel-telemetry.js";
import { initCamera, onCamera, requestCamera } from "./panel-camera.js";
import { initPolicies, renderPolicies } from "./panel-act.js";
import { initGoalPanel } from "./panel-goal.js";
import {
  markSource, pickMode, renderStatus, setLink, setRecording, tickRate,
} from "./panel-status.js";
import { renderWorkspace } from "./panel-workspace.js";
import { renderJointState } from "./panel-jointstate.js";
import { initZero, renderZero, zeroBusy } from "./panel-zero.js";
import { initModelImport, renderImport } from "./panel-model.js";
import {
  initRecordCameras, recordCameras, renderCameraSources, renderRecordCameras,
  sendRecordCameras,
} from "./panel-record.js";
import { toast } from "./toast.js";

function onHello(message) {
  state.jointNames = message.joint_names;
  message.state.joints.forEach((joint) => {
    state.latest[joint.name] = joint;
    state.planned[joint.name] = joint.position;
  });
  if (message.status) {
    renderProfiles(message.status);
    renderInterpolator(message.status);
    if (message.status.gains) state.gains = message.status.gains;
  }
  buildSliders();
  renderStatus(message.status);
  renderCameraSources(message.status);
  if (message.status) renderWorkspace(message.status.active_source);
  updatePlanHint();
}

function onState(message) {
  message.joints.forEach((joint) => { state.latest[joint.name] = joint; });
  renderTelemetry(message.joints);
  applyCurrent();
  if (message.status) {
    renderProfiles(message.status);
    renderInterpolator(message.status);
    if (message.status.gains) state.gains = message.status.gains;
    renderStatus(message.status);
    renderCameraSources(message.status);
    renderGainsTable();
  }
  // The stream's own source is newer than the status, which comes once a
  // second; the workspace follows it.
  if (message.active_source) markSource(message.active_source);
  if (renderZero(message.zero)) {
    // The arm is at zero now, and so are the server's slider targets.
    syncToCurrent(true);
  }
  if (state.status) renderWorkspace(state.status.active_source, zeroBusy());
  // Recording is for demonstrations: teleop drives them. An episode
  // already running keeps its card, so it can be stopped.
  const recordable = Boolean(state.status && state.status.active_source === "teleop");
  document.getElementById("recording-card").hidden =
    !(recordable || message.recording);
  renderJointState(message.joints, state.jointNames);
  setRecording(message.recording, message.recording_seconds || 0,
               message.episode, message.recording_cameras,
               { compressing: message.compressing, ready: message.recorder_ready });
  announceSaved(message.last_saved);
  renderRecordCameras(message.recording, message.recording_cameras);
  renderImport(message.model_import);
  if (message.policies) renderPolicies(message.policies);
  updatePlanHint();
  tickRate();
}

// The last episode's end, said once: its samples and rate once compressed.
let savedId = null;
function announceSaved(saved) {
  if (!saved) {
    if (savedId === null) savedId = 0;
    return;
  }
  // The first state after loading the page only tells of an old one.
  if (savedId !== null && saved.id !== savedId) toast(saved.message, saved.ok ? "ok" : "bad");
  savedId = saved.id;
}

function onAction(message) {
  // connect and disconnect already report a whole sentence, so prefixing
  // them with the action name would only repeat it.
  // The camera choice is sent on every connect: only a refusal is news.
  if (message.action === "record_cameras" && message.ok) return;
  const spelt = ["connect", "disconnect", "act", "zero_pose"].includes(message.action);
  const text = spelt ? message.message : `${message.action}: ${message.message}`;
  toast(text, message.ok ? "ok" : "bad");
  // Selecting ui: the server restarted its targets from the robot's pose,
  // so the plan, the sliders and the ghost start there too.
  if (message.action === "set_source" && message.resync) syncToCurrent(true);
  if (message.action === "record_start" && message.ok) setRecording(true);
  if (message.action === "record_stop" && message.ok) {
    setRecording(false, 0, null, null, { compressing: { episode: "", seconds: 0 } });
  }
}

function wire() {
  const click = (id, handler) =>
    document.getElementById(id).addEventListener("click", handler);

  document.querySelectorAll("#mode-buttons button[data-mode]")
    .forEach((button) => {
      button.addEventListener("click", () => pickMode(button.dataset.mode));
    });
  click("btn-connect", (event) => {
    if (event.currentTarget.dataset.action === "disconnect") {
      return send({ action: "disconnect" });
    }
    if (!state.mode) return toast("Pick real or sim first.", "bad");
    // Starting MuJoCo or restarting the controller takes seconds, and the
    // reply only comes when it is done, so say that something is happening.
    toast(`Connecting to ${state.mode}…`);
    return send({ action: "connect", mode: state.mode });
  });
  click("btn-record", () => send(state.recording
    ? { action: "record_stop" }
    : { action: "record_start", cameras: recordCameras() }));

  click("btn-unlock-all", () => setAllLocks(false));
  click("btn-lock-all", () => setAllLocks(true));
  click("btn-home", stageHomePose);
  click("btn-sync", () => syncToCurrent());
  click("btn-execute", execute);

  document.querySelectorAll(".tab[data-arm]").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab[data-arm]")
        .forEach((other) => other.classList.remove("active"));
      tab.classList.add("active");
      state.arm = tab.dataset.arm;
      buildSliders();
    });
  });

  document.querySelectorAll("#viewport-tools button[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  document.getElementById("show-planned").addEventListener("change", (event) => {
    setPlannedVisible(event.target.checked);
  });

  document.querySelectorAll(".tab.pane").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab.pane")
        .forEach((other) => other.classList.remove("active"));
      tab.classList.add("active");
      showGainsPane(tab.dataset.pane === "gains");
    });
  });
  document.getElementById("gain-profile").addEventListener("change", (event) => {
    send({ action: "set_profile", profile: event.target.value });
  });
}

on("hello", onHello);
on("state", onState);
on("status", (message) => {
  renderProfiles(message.status);
  renderInterpolator(message.status);
  if (message.status.gains) state.gains = message.status.gains;
  renderStatus(message.status);
  renderCameraSources(message.status);
  renderGainsTable();
  // The workspace waits for the next state message, whose source is live.
});
on("action", onAction);
on("camera", onCamera);
on("error", (message) => toast(message.message, "bad"));

document.getElementById("profile-pick").hidden = true;
initTheme();
// Without WebGL (no GPU, a broken driver) the 3D view cannot start; the rest
// of the page -- connect, sources, sliders, camera -- must still work.
try {
  initViewport((error) => toast(`3D model failed to load: ${error}`, "bad"));
} catch (error) {
  toast(`3D view unavailable: ${error.message}`, "bad");
}
onThemeChange(applyTheme);
wire();
initCamera();
initRecordCameras();
initPolicies();
initZero();
initModelImport();
initGoalPanel();
initInterpolator();
connect({
  onOpen: () => {
    setLink("connected", true);
    requestCamera();
    sendRecordCameras();
  },
  onClose: () => setLink("disconnected", false),
});
