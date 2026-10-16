// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The header strip and the right-hand cards: what the robot is attached to,
// which source is driving it, the gains, and the recording state.

import { state } from "./state.js";
import { send } from "./transport.js";

export function renderStatus(status) {
  if (!status) return;
  state.status = status;

  const mode = document.getElementById("chip-mode");
  if (status.connecting) {
    mode.textContent = `switching to ${status.connecting}…`;
    mode.className = "chip warn";
  } else {
    mode.textContent = `mode ${status.mode}`;
    // Real is the one where a mistake moves a physical arm, so it is the
    // one the header calls out.
    mode.className = `chip ${status.mode === "real" ? "warn" : "ok"}`;
  }

  document.getElementById("chip-bus").textContent =
    status.interfaces.length ? `bus ${status.interfaces.join(", ")}` : "bus —";

  // joint_states is published whether or not the motors answer, so the
  // count of those replying is the part that says the robot is really there.
  const motors = document.getElementById("fact-motors");
  const answering = status.motors_answering;
  if (status.motors_enabled === null || status.motors_enabled === undefined) {
    motors.textContent = "controller not running";
  } else if (!status.motors_enabled) {
    motors.textContent = "disabled";
  } else {
    motors.textContent = answering
      ? `enabled, ${answering} answering`
      : "enabled, none answering";
  }

  document.getElementById("fact-bus").textContent =
    status.interfaces.join(", ") || "—";
  document.getElementById("fact-state").textContent =
    status.state_age === null ? "no messages"
      : `${status.state_age.toFixed(2)} s ago`;

  renderLink(status);
  renderMode(status);
  renderSources(status);
}

/**
 * The banner at the top of the Robot card: which robot, and whether it is
 * really there. Connected means the motors are armed and every one of them
 * answers -- a running controller on an empty bus is not a connection.
 */
function renderLink(status) {
  const target = status.connecting || status.mode;
  let kind, label, detail = "";
  const expected = status.motors_expected || 16;
  const answering = status.motors_answering || 0;
  if (status.connecting) {
    kind = "warn";
    label = "Connecting…";
  } else if (status.mode === "unknown") {
    kind = "idle";
    label = "Not connected";
    detail = "no controller";
  } else if (!status.motors_enabled) {
    kind = "idle";
    label = "Not connected";
    detail = "motors off";
  } else if (answering >= expected) {
    kind = "ok";
    label = "Connected";
    detail = `${answering}/${expected} motors`;
  } else if (answering > 0) {
    kind = "warn";
    label = "Partly connected";
    detail = `${answering}/${expected} motors`;
  } else {
    kind = "bad";
    label = "No response";
    detail = `0/${expected} motors`;
  }

  document.getElementById("link-banner").className = `link-banner ${kind}`;
  document.getElementById("link-target").textContent =
    target === "sim" ? "SIM" : target === "real" ? "REAL" : "—";
  document.getElementById("link-state").textContent = label;
  document.getElementById("link-detail").textContent = detail;
}

/** The real/sim picker, and what it is allowed to do right now. */
function renderMode(status) {
  // Adopt whatever is running, so the picker opens on the truth rather than
  // on a guess. An explicit click wins from then on.
  if (!state.modePicked && (status.mode === "real" || status.mode === "sim")) {
    state.mode = status.mode;
  }

  const busy = Boolean(status.connecting);
  document.querySelectorAll("#mode-buttons button[data-mode]")
    .forEach((button) => {
      const picked = button.dataset.mode === state.mode;
      button.setAttribute("aria-pressed", String(picked));
      button.disabled = busy;
    });

  // One button: green Connect until the picked robot is armed, then red
  // Disconnect. Picking the other mode turns it back into Connect, since
  // the click would then switch robots rather than stop this one.
  const armed = (status.mode === "real" || status.mode === "sim")
    && Boolean(status.motors_enabled);
  const disconnect = armed && state.mode === status.mode;
  const connect = document.getElementById("btn-connect");
  connect.dataset.action = disconnect ? "disconnect" : "connect";
  connect.className = disconnect ? "off" : "on";
  connect.disabled = busy || (!disconnect && !state.mode);
  connect.textContent = busy
    ? "Connecting…"
    : disconnect
      ? "Disconnect"
      : (state.mode && state.mode !== status.mode
          ? `Connect (${state.mode})`
          : "Connect");

  const running = status.mode === "unknown" ? "no controller" : status.mode;
  document.getElementById("fact-mode").textContent =
    status.sim_up ? `${running}, MuJoCo running` : running;
}

export function pickMode(mode) {
  state.mode = mode;
  state.modePicked = true;
  if (state.status) renderMode(state.status);
}

function renderSources(status) {
  const host = document.getElementById("source-buttons");
  const wanted = ["none", ...(status.sources || [])];
  if (host.dataset.sources !== wanted.join(",")) {
    host.dataset.sources = wanted.join(",");
    host.innerHTML = wanted
      .map((name) => `<button data-source="${name}">${name}</button>`)
      .join("");
    host.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () =>
        send({ action: "set_source", source: button.dataset.source }));
    });
  }
  host.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active",
      button.dataset.source === status.active_source);
  });
}

/** The proxy's selection as the state stream reports it, between the
 * once-a-second statuses. */
export function markSource(source) {
  if (!state.status || state.status.active_source === source) return;
  state.status.active_source = source;
  document.querySelectorAll("#source-buttons button").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
}

export function setLink(text, ok) {
  const chip = document.getElementById("chip-link");
  chip.textContent = text;
  chip.className = `chip ${ok ? "ok" : "bad"}`;
}

// An episode is normally 15-25 s, per the dataset contract; the clock turns
// amber past that so a long take is obvious before it is saved.
const EPISODE_TARGET_S = 25;

function clock(seconds) {
  const whole = Math.floor(seconds);
  const minutes = String(Math.floor(whole / 60)).padStart(2, "0");
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * The recording card. Between episodes the button waits for two things: the
 * last episode's compression (it would take CPU from the next recording) and
 * a recorder armed on the chosen cameras (so Start begins at once).
 * `phase`: {compressing: {episode, seconds} | null, ready: bool}.
 */
export function setRecording(active, seconds = 0, episode = null, cameras = null,
                             phase = {}) {
  const elapsed = document.getElementById("record-elapsed");
  const shown = clock(seconds);
  if (elapsed.textContent !== shown) elapsed.textContent = shown;
  elapsed.classList.toggle("running", active);
  elapsed.classList.toggle("over", active && seconds > EPISODE_TARGET_S);

  const { compressing = null, ready = true } = phase;
  let label;
  let text;
  let busy = false;
  if (active) {
    // The episode name arrives with the first state message after the
    // start action, so the line is rewritten whenever either changes.
    const which = cameras && cameras.length ? ` (${cameras.join(" + ")})` : "";
    label = `Recording ${episode || "an episode"}${which}…`;
    text = "Stop recording";
  } else if (compressing) {
    label = `Compressing ${compressing.episode || "the episode"}… record again when it is done.`;
    text = `Compressing… ${Math.floor(compressing.seconds || 0)} s`;
    busy = true;
  } else if (!ready) {
    label = "Preparing the recorder for the cameras ticked…";
    text = "Preparing…";
    busy = true;
  } else {
    label = "Ready: recording starts at once, at 50 Hz.";
    text = "Start recording";
  }
  const status = document.getElementById("record-status");
  if (status.textContent !== label) status.textContent = label;

  state.recording = active;
  const button = document.getElementById("btn-record");
  if (button.textContent !== text) button.textContent = text;
  button.disabled = busy;
  button.classList.toggle("active", active);
  button.classList.toggle("busy", busy);
}

let frames = 0;
let lastTick = performance.now();

export function tickRate() {
  frames += 1;
  const now = performance.now();
  if (now - lastTick < 1000) return;
  document.getElementById("chip-rate").textContent =
    `${Math.round((frames * 1000) / (now - lastTick))} Hz`;
  frames = 0;
  lastTick = now;
}
