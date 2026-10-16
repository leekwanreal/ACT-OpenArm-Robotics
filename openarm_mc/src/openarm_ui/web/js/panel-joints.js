// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The joint panel: per-joint safety lock, sliders that stage a plan, and the
// Execute that runs it.

import {
  ARM_STEP, GRIPPER_STEP, HOME_POSE, JOINTS,
  clamp, formatValue, isGripper,
} from "./joints.js";
import { isUnlocked, planError, state, unlockedJoints } from "./state.js";
import { send } from "./transport.js";
import { applyPlanned } from "./viewport.js";
import { toast } from "./toast.js";

export function buildSliders() {
  const host = document.getElementById("sliders");
  host.innerHTML = "";
  state.jointNames
    .filter((name) => name.startsWith(state.arm))
    .forEach((name, index) => {
      const spec = JOINTS[name];
      if (!spec) return;
      host.appendChild(jointCard(name, spec, index));
    });
  refreshSliderValues();
}

function jointCard(name, spec, index) {
  const locked = !isUnlocked(name);
  const step = isGripper(name) ? GRIPPER_STEP : ARM_STEP;
  const bounds = isGripper(name)
    ? ["0.0 mm", "43.0 mm"]
    : [spec.min.toFixed(2), `+${spec.max.toFixed(2)}`];

  const card = document.createElement("div");
  card.className = `joint-card${locked ? " locked" : ""}`;
  card.innerHTML = `
    <div class="joint-head">
      <button class="lock"
              title="${locked ? "Locked. Click to unlock." : "Unlocked. Click to lock."}">
        ${locked ? "OFF" : "ON"}
      </button>
      <span class="badge">J${index + 1}</span>
      <span class="joint-label">${spec.label}</span>
      <span class="motor">${spec.motor}</span>
      <span class="joint-value" id="o-${name}">—</span>
      <button class="zero" ${locked ? "disabled" : ""} title="Plan this joint at 0">0</button>
    </div>
    <div class="joint-track">
      <span class="bound">${bounds[0]}</span>
      <input type="range" id="s-${name}" min="${spec.min}" max="${spec.max}"
             step="${step}" value="${state.planned[name] ?? 0}"
             ${locked ? "disabled" : ""} />
      <span class="bound">${bounds[1]}</span>
    </div>`;

  const input = card.querySelector("input");
  input.addEventListener("input", () => planJoint(name, parseFloat(input.value)));
  // While a slider is held, do not fight the user with robot feedback.
  input.addEventListener("pointerdown", () => { state.sliderHeld = name; });
  input.addEventListener("pointerup", () => { state.sliderHeld = null; });

  card.querySelector(".lock").addEventListener("click", () => {
    state.locked[name] = isUnlocked(name);
    buildSliders();
    updatePlanHint();
  });
  card.querySelector(".zero").addEventListener("click", () => planJoint(name, 0));
  return card;
}



/** Stage a value for one joint. The robot does not move; the ghost does. */
export function planJoint(name, value) {
  // A locked joint is the safety catch: its plan cannot be edited either.
  if (!isUnlocked(name)) return;
  const clamped = clamp(name, value);
  state.planned[name] = clamped;
  const input = document.getElementById(`s-${name}`);
  if (input) input.value = clamped;
  const output = document.getElementById(`o-${name}`);
  if (output) output.textContent = formatValue(name, clamped);
  applyPlanned();
  updatePlanHint();
}

export function refreshSliderValues() {
  state.jointNames.forEach((name) => {
    const input = document.getElementById(`s-${name}`);
    if (!input || state.sliderHeld === name) return;
    const value = state.planned[name];
    if (value === undefined) return;
    input.value = value;
    document.getElementById(`o-${name}`).textContent =
      formatValue(name, Number(value));
  });
}

/** Move the plan onto the robot's own pose. Commands nothing. */
export function syncToCurrent(quiet = false) {
  Object.values(state.latest).forEach((joint) => {
    state.planned[joint.name] = joint.position;
  });
  refreshSliderValues();
  applyPlanned();
  updatePlanHint();
  if (!quiet) toast("plan reset to where the robot is");
}

export function execute() {
  const unlocked = unlockedJoints();
  if (!unlocked.length) {
    toast("unlock the joints you want to move first", "bad");
    return;
  }
  // Only the unlocked joints: the server keeps every other one at the
  // setpoint the controller holds it at. Sending the measured position
  // instead would let a locked joint drop by its sag under gravity.
  const positions = {};
  unlocked.forEach((name) => { positions[name] = state.planned[name]; });
  send({ action: "set_joints", positions });
  toast(`executing ${unlocked.length} joint${unlocked.length > 1 ? "s" : ""}`);
}

export function setAllLocks(locked) {
  state.jointNames.forEach((name) => { state.locked[name] = Boolean(locked); });
  buildSliders();
  updatePlanHint();
}

export function stageHomePose() {
  const unlocked = unlockedJoints();
  if (!unlocked.length) {
    toast("unlock a joint first", "bad");
    return;
  }
  unlocked.forEach((name) => {
    const key = isGripper(name) ? "gripper" : name.split("_")[1];
    planJoint(name, HOME_POSE[key] ?? 0);
  });
  toast("home pose staged, press Execute to run it");
}

export function updatePlanHint() {
  const hint = document.getElementById("plan-hint");
  const error = planError();
  const pending = error >= 1e-3;
  const text = pending
    ? `Plan differs from the robot by ${error.toFixed(3)} on its widest joint.`
      + " Press Execute to run it."
    : "Plan matches the robot. Move a slider to stage a new pose, then Execute.";
  // Called with every state message: touch the page only on a change.
  if (hint.textContent !== text) hint.textContent = text;
  hint.classList.toggle("pending", pending);
}
