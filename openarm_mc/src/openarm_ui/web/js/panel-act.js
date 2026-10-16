// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The policy cards: ACT (openarm_act) and SmolVLA (openarm_smolvla), which
// run the same way, each as its own command source.
//
// Continuously, as act_pipeline's infer_robot.py: Start runs the model over
// and over while nothing moves -- the arrows follow its newest chunk -- and
// Execute lets it drive, each step the ensembled first action of the
// chunks, smoothed; it ends by itself back home (or on a timeout) and eases
// the arm to zero. One chunk: Generate infers once and Run trajectory plays
// those 50 steps once; Reset to zero eases the arm to zero, grippers open.
// Stop takes the arm back from either and holds it.
//
// ACT's model list swaps the model for another .onnx under checkpoint/onnx/
// while nothing runs; Load… brings a new one from this computer
// (panel-model.js). SmolVLA's model is whatever its server serves, trained
// on one task, which its card shows: there is no prompt to give it.
//
// Each card's elements are named after its id prefix: `${prefix}-state`,
// `btn-${prefix}-start` and so on.

import { state } from "./state.js";
import { send } from "./transport.js";
import { toast } from "./toast.js";
import { setTrajectory, setTrajectoryVisible } from "./viewport.js";
import { currentWorkspace } from "./panel-workspace.js";
import { importBusy } from "./panel-model.js";

const byId = (id) => document.getElementById(id);
// File names go into the picker's markup.
const escape = (text) => String(text).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// A card renders with every state message; writing a text or a class that
// has not changed would still cost the page a style pass.
const setText = (id, text) => {
  const node = byId(id);
  if (node.textContent !== text) node.textContent = text;
  return node;
};
const setClass = (id, name) => {
  const node = byId(id);
  if (node.className !== name) node.className = name;
};

function describe(status) {
  if (!status) return ["not running", "bad"];
  if (status.model_loading) return ["loading model…", "warn"];
  if (status.mode === "execute") {
    if (status.going_home) return ["going home", "live"];
    if (status.returning) return ["going to zero", "live"];
    if (status.warming_up) return ["easing in", "live"];
    if (status.running) return [`policy: step ${status.steps_executed}`, "live"];
    const run = status.playback;
    return [run ? `running ${run.step}/${run.steps}` : "executing", "live"];
  }
  if (status.generating) return ["generating…", "warn"];
  if (status.running) {
    if (status.predicting) return ["policy: preview", "ok"];
    return [status.waiting ? `waiting: ${status.waiting}` : "waiting", "warn"];
  }
  if (status.plan) return [`ready (${status.plan.age_s.toFixed(0)} s ago)`, "ok"];
  return ["idle", "warn"];
}

function confirmOnRobot(what) {
  const robot = state.status && state.status.mode;
  if (robot === "sim") return true;
  const where = robot === "real" ? "the REAL robot" : "the arm";
  return window.confirm(`${what} on ${where}?\n\n` +
    "It eases in from where the arm is. Press Stop at any time to take " +
    "the arm back.");
}

/**
 * One card. `key` is the policy, which is also its proxy source and its
 * workspace; `prefix` names its elements. `models`: the ACT model picker;
 * `server`: SmolVLA's server and the task its model was trained on.
 */
function createPolicyPanel({ key, prefix, label, models = false, server = false }) {
  const id = (name) => `${prefix}-${name}`;
  const button = (name) => byId(`btn-${prefix}-${name}`);
  const panel = {
    status: null,       // the node's status, null when it is not running
    shown: true,        // the trajectory arrows are drawn
    visible: false,     // they are, now
    generateId: null,   // last_generate.id already announced
    modelLoadId: null,  // last_model_load.id already announced
    modelsKey: "",      // the model list the picker was built from
    picked: null,       // the operator's choice in the list, until loaded
    pickedAt: 0,        // when, so a refused pick goes back to the loaded one
    wasExecuting: false,
  };
  const act = (action, extra = {}) => send({ action, policy: key, ...extra });

  /** Say when a generation, or a model load, has finished; once each. */
  function announce(status) {
    const result = status && status.last_generate;
    if (result && result.id !== panel.generateId) {
      // The first status after loading the page only tells what is already
      // drawn; it is not news.
      if (panel.generateId !== null) {
        toast(`${label}: ${result.message}`, result.ok ? "ok" : "bad");
      }
      panel.generateId = result.id;
    }
    const load = status && status.last_model_load;
    if (load && load.id !== panel.modelLoadId) {
      if (panel.modelLoadId !== null) toast(load.message, load.ok ? "ok" : "bad");
      panel.modelLoadId = load.id;
      panel.picked = null;        // the picker shows what is loaded again
    } else if (status && !load && panel.modelLoadId === null) {
      panel.modelLoadId = 0;
    }
  }

  /** The model picker: every .onnx the node found, the loaded one chosen. */
  function renderModels(status, busy) {
    const select = byId(id("model-select"));
    const found = (status && status.models) || [];
    const listKey = found.map((model) => model.path).join("\n");
    if (listKey !== panel.modelsKey) {
      panel.modelsKey = listKey;
      select.innerHTML = found.length
        ? found.map((model) => {
          const date = new Date(model.mtime * 1000).toLocaleString([], {
            month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
          });
          return `<option value="${escape(model.path)}">${escape(model.name)} · ${date}</option>`;
        }).join("")
        : '<option value="">no .onnx under checkpoint/onnx/</option>';
    }
    const loaded = status ? status.checkpoint : "";
    if (panel.picked && !(status && status.model_loading) &&
        performance.now() - panel.pickedAt > 3000) {
      panel.picked = null;          // refused, or long done
    }
    const wanted = panel.picked ?? (status && status.model_loading) ?? loaded;
    if (document.activeElement !== select && select.value !== wanted &&
        found.some((model) => model.path === wanted)) {
      select.value = wanted;
    }
    select.disabled = !status || !found.length || busy || importBusy() ||
      Boolean(status.model_loading);
    select.title = busy ? "Stop the policy first: a model loads only while nothing runs"
      : "The models under checkpoint/onnx/: pick one to load it";
  }

  /** SmolVLA: the task its model was trained on, and the server. */
  function renderServer(status) {
    const info = status && status.smolvla;
    setText(id("task"), (info && info.task) || "—").title = (info && info.task) || "";
    const address = info
      ? `${info.server}${info.connected ? "" : " · reconnecting"}` +
        (info.step != null ? ` · step ${info.step}` : "")
      : "—";
    setText(id("server"), address).title = info && info.checkpoint ? info.checkpoint : "";
  }

  /** The parameter inputs show the node's values, unless being edited. */
  function syncParams(status) {
    if (!status) return;
    const show = (name, value) => {
      const input = byId(id(name));
      if (document.activeElement === input || value === undefined || value === null) return;
      if (input.value !== String(value)) input.value = value;
    };
    show("vel-scale", status.vel_scale);
    show("ensemble-m", status.ensemble_m);
    show("execute-timeout", status.execute_timeout);
    show("loop-mode", status.loop_mode);
    byId(id("loop-mode")).disabled = status.mode === "execute";
  }

  /** Whether the trajectory is worth drawing: it can still be run, it
   * runs, or continuous inference keeps it current. */
  const current = (status) => Boolean(status && (status.plan || status.playback ||
                                                 (status.running && status.predicting)));

  function showTrajectory() {
    // Drawn only while this card's workspace is up, as the card is.
    panel.visible = currentWorkspace() === key && panel.shown && current(panel.status);
    setTrajectoryVisible(key, panel.visible);
  }

  function render(payload) {
    const status = payload ? payload.status : null;
    panel.status = status;
    announce(status);

    const [text, kind] = describe(status);
    setText(id("state"), text);
    setClass(id("state"), `act-state ${kind}`);

    // While the policy runs, how many chunks a second the model delivers:
    // about 50 on a GPU, a few on a CPU, two for SmolVLA (one every 25 steps).
    const rate = status && status.running && status.infer_hz
      ? ` · ${status.infer_hz.toFixed(1)} chunks/s` : "";
    const modelMs = status && status.smolvla && status.smolvla.server_ms != null
      ? ` (model ${status.smolvla.server_ms.toFixed(0)} ms)` : "";
    setText(id("latency"),
      status && status.latency_ms != null
        ? `${status.latency_ms.toFixed(0)} ms${modelMs}${rate}` : "—");
    // A camera the policy does not really see: a virtual one (black, no
    // RealSense for it) or none at all (infer_robot.py's mock frame).
    const blind = status && status.cameras
      ? Object.entries(status.cameras).filter(([, seen]) => seen && seen !== "live")
        .map(([camera, seen]) => `${camera} ${seen === "virtual" ? "virtual (black)" : "missing: mock frame"}`)
      : [];
    const model = status
      ? status.policy + (blind.length ? ` · not seeing: ${blind.join(", ")}` : "")
      : "—";
    setText(id("model"), model).title = status ? model : "";

    const executing = Boolean(status && status.mode === "execute");
    if (panel.wasExecuting && !executing && status && status.stop_reason) {
      // It stopped by itself (the trajectory is done, or a timeout): say why.
      toast(`${label} stopped -- ${status.stop_reason}. The arm holds.`, "ok");
    }
    panel.wasExecuting = executing;
    const steps = executing && status.running ? ` · ${status.steps_executed} steps` : "";
    setText(id("run"), executing && status.executing_s != null
      ? `${status.executing_s.toFixed(1)} s${steps}`
      : (status && status.stop_reason ? `last: ${status.stop_reason}` : "—"));

    const generating = Boolean(status && status.generating);
    const running = Boolean(status && status.running);
    const loading = Boolean(status && status.model_loading);
    if (models) {
      renderModels(status, running || executing || generating ||
                           Boolean(status && status.playback));
    }
    if (server) renderServer(status);
    // The continuous policy.
    button("start").disabled =
      !status || running || executing || generating || loading;
    button("run-policy").disabled = !(running && status.predicting) || executing;
    syncParams(status);
    // One chunk.
    const generate = button("generate");
    generate.disabled = !status || running || executing || generating || loading;
    setText(`btn-${prefix}-generate`, generating ? "Generating…" : "Generate trajectory");
    button("execute").disabled =
      !(status && status.plan) || running || executing || generating;
    button("reset").disabled =
      !(status && status.can_reset) || executing || generating || running;
    // Stop ends either, and stops a preview too.
    button("stop").disabled = !(executing || running);
    byId(id("card")).classList.toggle("executing", executing);

    if (payload && payload.trajectory) {
      const names = state.jointNames;
      setTrajectory(key, payload.trajectory.map((row) =>
        Object.fromEntries(names.map((name, index) => [name, row[index]]))));
    }
    // Stale is hidden, not dropped: the trajectory can arrive a moment
    // before the status that makes it current, and is sent to a page once.
    showTrajectory();
  }

  /** Play the generated chunk once. */
  function execute() {
    const status = panel.status;
    if (!status || !status.plan) {
      toast("No trajectory to run yet: generate one first.", "bad");
      return;
    }
    if (!confirmOnRobot(`Run the ${label} trajectory (${status.plan.steps} steps)`)) return;
    toast(`Running the ${label} trajectory…`);
    act("act_execute");
  }

  /** Let the continuously running policy drive the arm. */
  function executePolicy() {
    const status = panel.status;
    if (!status || !status.running || !status.predicting) {
      toast("Start the policy first, and let it predict.", "bad");
      return;
    }
    const limit = status.execute_timeout > 0
      ? `up to ${status.execute_timeout} s` : "no time limit";
    const task = status.smolvla && status.smolvla.task ? `, "${status.smolvla.task}"` : "";
    if (!confirmOnRobot(`Run ${label} continuously (${status.loop_mode}, ` +
                        `speed ${status.vel_scale}, ${limit}${task})`)) return;
    toast(`Executing ${label}…`);
    act("act_execute");
  }

  function reset() {
    const robot = state.status && state.status.mode;
    if (robot !== "sim") {
      const where = robot === "real" ? "the REAL robot" : "the arm";
      const ok = window.confirm(
        `Move ${where} to the zero pose, grippers open?\n\n` +
        "It eases there slowly. Press Stop at any time.");
      if (!ok) return;
    }
    toast("Going to the zero pose…");
    act("act_reset");
  }

  function init() {
    button("start").addEventListener("click", () => act("act_start"));
    button("run-policy").addEventListener("click", executePolicy);
    [["vel-scale", "vel_scale"], ["ensemble-m", "ensemble_k"],
     ["execute-timeout", "execute_timeout"],
     ["loop-mode", "loop_mode"]].forEach(([name, parameter]) => {
      byId(id(name)).addEventListener("change", (event) =>
        act("act_param", { name: parameter, value: event.target.value }));
    });
    button("execute").addEventListener("click", execute);
    button("stop").addEventListener("click", () => act("act_stop"));
    button("reset").addEventListener("click", reset);
    button("generate").addEventListener("click", () => act("act_generate"));
    byId(id("show")).addEventListener("change", (event) => {
      panel.shown = event.target.checked;
      showTrajectory();
      renderLegend();
    });
    if (models) {
      // Picking a model in the list loads it.
      const select = byId(id("model-select"));
      select.addEventListener("change", () => {
        const path = select.value;
        if (!path || !panel.status || path === panel.status.checkpoint) return;
        panel.picked = path;
        panel.pickedAt = performance.now();
        const name = select.selectedOptions[0] ? select.selectedOptions[0].textContent : path;
        toast(`Loading ${name}…`);
        act("act_model", { path });
        select.blur();
      });
    }
  }

  return { key, render, init, isVisible: () => panel.visible };
}

const panels = [
  createPolicyPanel({ key: "act", prefix: "act", label: "ACT", models: true }),
  createPolicyPanel({ key: "smolvla", prefix: "smolvla", label: "SmolVLA", server: true }),
];

function renderLegend() {
  byId("legend-traj").hidden = !panels.some((panel) => panel.isVisible());
}

/** `policies` is {act: {status, pose, trajectory?}, smolvla: {...}}. */
export function renderPolicies(policies) {
  panels.forEach((panel) => panel.render(policies[panel.key] || null));
  renderLegend();
}

export function initPolicies() {
  panels.forEach((panel) => panel.init());
}
