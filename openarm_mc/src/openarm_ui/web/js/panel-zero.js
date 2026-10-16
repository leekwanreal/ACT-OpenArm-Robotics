// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Zero pose: every arm joint to 0 from wherever the arm is, whoever was
// driving it. The server does the work (ros_node.py, zero_pose.py): it cuts
// the old source off at the proxy, stops a policy that was executing,
// streams the motion as ui, and afterwards hands the arm to no one (or back
// to the sliders, when they were driving). While it runs the sliders are
// refused and the button stops it where it is.

import { state } from "./state.js";
import { send } from "./transport.js";
import { toast } from "./toast.js";

const zero = {
  phase: "idle",
  resultId: null,       // last.id already announced
};

const byId = (id) => document.getElementById(id);

export const zeroBusy = () => zero.phase !== "idle";

function describe(snapshot) {
  switch (snapshot.phase) {
    case "starting": return "Taking the arm…";
    case "moving":
      return `Going to zero · ${snapshot.remaining_s.toFixed(1)} s left`;
    case "finishing": return "Handing the arm back…";
    default:
      return snapshot.last ? snapshot.last.message
        : "Every arm joint to 0, grippers kept.";
  }
}

/** The zero part of a state message; -> true when a motion just ended. */
export function renderZero(snapshot) {
  if (!snapshot) return false;
  const was = zero.phase;
  zero.phase = snapshot.phase;
  const busy = zeroBusy();

  const last = snapshot.last;
  if (last && last.id !== zero.resultId) {
    // The first message after loading only tells what happened before.
    if (zero.resultId !== null) toast(last.message, last.ok ? "ok" : "bad");
    zero.resultId = last.id;
  } else if (!last && zero.resultId === null) {
    zero.resultId = 0;
  }

  const button = byId("btn-zero");
  const label = busy ? "Stop" : "Zero pose";
  if (button.textContent !== label) button.textContent = label;
  button.classList.toggle("off", busy);
  const status = state.status;
  const armed = Boolean(status && status.motors_enabled);
  button.disabled = busy ? snapshot.phase === "finishing" : !armed;
  button.title = busy ? "Stop where the arm is; it holds there"
    : armed ? "Ease every arm joint to 0 from wherever the arm is; the grippers stay as they are"
      : "Connect first: the motors are off";

  const text = describe(snapshot);
  const line = byId("zero-text");
  if (line.textContent !== text) line.textContent = text;
  byId("zero-pose").classList.toggle("busy", busy);
  byId("zero-bar").style.width = `${(snapshot.progress * 100).toFixed(1)}%`;

  // The sliders' Execute would be refused: say so before it is pressed.
  byId("btn-execute").disabled = busy;
  document.querySelector(".joint-control-card").classList.toggle("zero-busy", busy);
  return was !== "idle" && !busy;
}

function start() {
  const robot = state.status && state.status.mode;
  if (robot !== "sim") {
    const where = robot === "real" ? "the REAL robot" : "the arm";
    const source = state.status && state.status.active_source;
    const from = source && source !== "none"
      ? `\n\n${source} is cut off first; afterwards no source drives the arm` +
        (source === "ui" ? " but the sliders." : ".")
      : "";
    const ok = window.confirm(
      `Move ${where} to the zero pose (every arm joint at 0)?` +
      ` The grippers stay as they are.${from}\n\nPress Stop at any time.`);
    if (!ok) return;
  }
  toast("Going to the zero pose…");
  send({ action: "zero_pose" });
}

export function initZero() {
  byId("btn-zero").addEventListener("click", () => {
    if (zeroBusy()) send({ action: "zero_stop" });
    else start();
  });
}
