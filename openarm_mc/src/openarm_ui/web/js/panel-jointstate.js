// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Teleop's view of the arm: every joint of both arms against its range,
// read-only. The telemetry table has the numbers; this shows at a glance
// how far each joint is from its limits while the leader arm drives.
//
// Redrawn at most TABLE_HZ times a second, and only while it is shown.

import { JOINTS, formatValue, isGripper } from "./joints.js";

/** Short enough to leave the track room: rad and degrees, or mm. */
function compact(name, value) {
  if (isGripper(name)) return `${(value * 1000).toFixed(1)} mm`;
  return `${value.toFixed(3)} · ${(value * 180 / Math.PI).toFixed(0)}°`;
}

const TABLE_HZ = 10;
// Closer than this to a limit, as a fraction of the range, is called out.
const NEAR_LIMIT = 0.05;

let rows = null;        // name -> {fill, marker, value, row, text}
let lastDrawn = 0;

function build(host, names) {
  const arms = ["left", "right"].map((arm) => {
    const joints = names.filter((name) => name.startsWith(arm) && JOINTS[name]);
    const items = joints.map((name, index) => {
      const spec = JOINTS[name];
      const label = isGripper(name) ? "Grip" : `J${index + 1}`;
      return `<div class="js-row" data-name="${name}" title="${spec.label} (${spec.motor})">
          <span class="js-label">${label}</span>
          <span class="js-track"><i class="js-zero"></i><i class="js-marker"></i></span>
          <span class="js-value"></span>
        </div>`;
    }).join("");
    return `<div class="js-arm"><h3>${arm === "left" ? "Left arm" : "Right arm"}</h3>${items}</div>`;
  }).join("");
  host.innerHTML = arms;

  rows = {};
  host.querySelectorAll(".js-row").forEach((row) => {
    const name = row.dataset.name;
    const spec = JOINTS[name];
    // Where 0 sits on the track, for the arm joints whose range spans it.
    const zero = (0 - spec.min) / (spec.max - spec.min);
    const tick = row.querySelector(".js-zero");
    if (isGripper(name) || zero <= 0 || zero >= 1) tick.hidden = true;
    else tick.style.left = `${(zero * 100).toFixed(2)}%`;
    rows[name] = {
      row, marker: row.querySelector(".js-marker"),
      value: row.querySelector(".js-value"), text: "", left: "",
    };
  });
}

/** joints: [{name, position, ...}] from a state message. */
export function renderJointState(joints, names) {
  const host = document.getElementById("jointstate");
  if (host.offsetParent === null) return;           // not shown
  const now = performance.now();
  const fresh = !rows || !host.firstChild;
  if (!fresh && now - lastDrawn < 1000 / TABLE_HZ) return;
  lastDrawn = now;
  if (fresh) build(host, names);

  joints.forEach((joint) => {
    const cells = rows[joint.name];
    const spec = JOINTS[joint.name];
    if (!cells || !spec) return;
    const span = spec.max - spec.min;
    const fraction = Math.min(1, Math.max(0, (joint.position - spec.min) / span));
    const left = `${(fraction * 100).toFixed(1)}%`;
    if (cells.left !== left) {
      cells.left = left;
      cells.marker.style.left = left;
    }
    const text = compact(joint.name, joint.position);
    if (cells.text !== text) {
      cells.text = text;
      cells.value.textContent = text;
      cells.row.title = `${spec.label} (${spec.motor}): ${formatValue(joint.name, joint.position)}`;
    }
    const near = !isGripper(joint.name) &&
      (fraction < NEAR_LIMIT || fraction > 1 - NEAR_LIMIT);
    cells.row.classList.toggle("near-limit", near);
  });
}
