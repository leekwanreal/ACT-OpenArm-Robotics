// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Position, velocity and effort for every joint, straight off joint_states.
//
// State arrives at the telemetry rate (30 Hz); a table of 48 numbers that
// fast is unreadable and costs a slow machine a layout per message, so it is
// redrawn at most TABLE_HZ times a second, and only the cells that changed.

const TABLE_HZ = 10;
let cells = null;               // name -> {p, v, e, text: [p, v, e]}
let lastDrawn = 0;

function build(body, joints) {
  body.innerHTML = joints
    .map((joint) => `<tr data-name="${joint.name}"><td>${joint.name}</td>
      <td class="num p"></td><td class="num v"></td><td class="num e"></td></tr>`)
    .join("");
  cells = {};
  body.querySelectorAll("tr").forEach((row) => {
    cells[row.dataset.name] = {
      p: row.querySelector(".p"), v: row.querySelector(".v"),
      e: row.querySelector(".e"), text: ["", "", ""],
    };
  });
}

export function renderTelemetry(joints) {
  const now = performance.now();
  const body = document.querySelector("#telemetry tbody");
  const fresh = !cells || body.children.length !== joints.length;
  if (!fresh && now - lastDrawn < 1000 / TABLE_HZ) return;
  lastDrawn = now;
  if (fresh) build(body, joints);
  joints.forEach((joint) => {
    const row = cells[joint.name];
    if (!row) return;
    const text = [joint.position.toFixed(4), joint.velocity.toFixed(3),
                  joint.effort.toFixed(2)];
    ["p", "v", "e"].forEach((key, index) => {
      if (row.text[index] === text[index]) return;
      row.text[index] = text[index];
      row[key].textContent = text[index];
    });
  });
}
