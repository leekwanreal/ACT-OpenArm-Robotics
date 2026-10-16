// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The PD tuning panel: every joint's gains for one profile, side by side so
// a shoulder and a wrist can be compared rather than scrolled between.
//
// A profile is a set of gains for one way of driving the arm -- teleoperation
// wants a compliant arm the operator can push against, a replayed policy
// wants a stiff one that tracks. Editing a profile that is not running
// changes what it will use next time, not what the motors are doing now.

import { GAIN_FIELDS, JOINTS, gainsFor } from "./joints.js";
import { state } from "./state.js";
import { send } from "./transport.js";
import { toast } from "./toast.js";

let built = false;

/** The controller's interpolator and filter4's w, unless being edited. */
export function renderInterpolator(status) {
  const kind = document.getElementById("interp-kind");
  const w = document.getElementById("interp-w");
  if (status.interpolator && document.activeElement !== kind &&
      kind.value !== status.interpolator) {
    kind.value = status.interpolator;
  }
  if (status.interp_w != null && document.activeElement !== w &&
      w.value !== String(status.interp_w)) {
    w.value = status.interp_w;
  }
  document.getElementById("interp-w-label").hidden = kind.value !== "filter4";
}

export function initInterpolator() {
  const kind = document.getElementById("interp-kind");
  kind.addEventListener("change", () => {
    document.getElementById("interp-w-label").hidden = kind.value !== "filter4";
    send({ action: "set_interpolator", kind: kind.value });
  });
  document.getElementById("interp-w").addEventListener("change", (event) =>
    send({ action: "set_interpolator", w: event.target.value }));
}

export function renderProfiles(status) {
  const select = document.getElementById("gain-profile");
  const names = status.profiles || [];
  if (select.dataset.names !== names.join(",")) {
    select.dataset.names = names.join(",");
    select.innerHTML = names
      .map((name) => `<option value="${name}">${name}</option>`).join("");
  }
  state.profiles = names;
  const changed = state.activeProfile !== status.active_profile;
  state.activeProfile = status.active_profile;
  if (status.active_profile && document.activeElement !== select) {
    select.value = status.active_profile;
  }
  if (changed) renderGainsTable(true);
}

/** Build the table once, then only refresh values that are not being edited. */
export function renderGainsTable(force = false) {
  const body = document.querySelector("#gains-table tbody");
  if (!built || force) {
    body.innerHTML = state.jointNames.map(gainRow).join("");
    body.querySelectorAll("input[data-joint]").forEach(wire);
    built = true;
  }
  updateHint();

  Object.entries(state.gains).forEach(([joint, fields]) => {
    Object.entries(fields).forEach(([field, value]) => {
      const input = body.querySelector(
        `input[data-joint="${joint}"][data-field="${field}"]`);
      if (!input || document.activeElement === input) return;
      if (value !== null && value !== undefined && input.value !== String(value)) {
        input.value = value;
      }
    });
  });
}

function gainRow(joint) {
  const spec = JOINTS[joint] || {};
  const cells = gainsFor(joint).map((field) => {
    const gain = GAIN_FIELDS[field];
    return `<td class="gain-cell">
      <label title="${gain.hint}">${gain.label}</label>
      <input type="number" data-joint="${joint}" data-field="${field}"
             min="${gain.min}" max="${gain.max}" step="${gain.step}" />
    </td>`;
  }).join("");
  return `<tr><td class="gain-joint">${joint}</td>
    <td class="gain-motor">${spec.motor || ""}</td>${cells}</tr>`;
}

function wire(input) {
  input.addEventListener("change", () => {
    const { joint, field } = input.dataset;
    const gain = GAIN_FIELDS[field];
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < gain.min || value > gain.max) {
      toast(`${joint} ${field} must be between ${gain.min} and ${gain.max}`, "bad");
      return;
    }
    send({ action: "set_gain", profile: state.activeProfile, joint, field, value });
  });
}

function updateHint() {
  const hint = document.getElementById("gains-hint");
  if (!state.activeProfile) {
    hint.textContent = "Controller not running.";
    return;
  }
  hint.innerHTML =
    `Editing <code>${state.activeProfile}</code>, which is the profile the `
    + "motors are running. Arm joints take kp/kd in their MIT frames; the "
    + "grippers run position-force, so theirs is a speed and a current cap. "
    + "Saved in <code>openarm_controller/config/gains.yaml</code>.";
}

export function showGainsPane(show) {
  document.getElementById("gains-pane").hidden = !show;
  document.getElementById("telemetry").hidden = show;
  document.getElementById("profile-pick").hidden = !show;
  if (show) renderGainsTable();
}
