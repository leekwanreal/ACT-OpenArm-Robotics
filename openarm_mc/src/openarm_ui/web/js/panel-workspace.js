// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The panel under the 3D view follows whoever drives the arm.
//
//   ui      the sliders: this page plans and executes poses
//   teleop  the camera, recording and the joint state -- the sliders would
//           only fight the leader arm
//   act     the camera and the ACT policy's controls
//   smolvla the camera and the SmolVLA policy's controls
//
// `none` keeps whatever was shown, so stopping a policy leaves its controls
// up to generate the next run. A zero pose borrows `ui` for a few seconds;
// that does not count as a switch either.
//
// The camera card is one element, moved between the sidebar and the
// workspace rather than copied: its stream and listeners come along.

import { ensureCameraOn } from "./panel-camera.js";

const LAYOUTS = ["ui", "teleop", "act", "smolvla"];
const WITH_CAMERA = ["teleop", "act", "smolvla"];

let current = "ui";

export const currentWorkspace = () => current;

/** Show the workspace for `source`; `borrowed` while a zero pose runs. */
export function renderWorkspace(source, borrowed = false) {
  if (borrowed || !LAYOUTS.includes(source) || source === current) return;
  current = source;
  document.getElementById("workspace").dataset.workspace = source;
  document.getElementById("joints-pane").dataset.workspace = source;

  const camera = document.getElementById("camera-card");
  const host = WITH_CAMERA.includes(source)
    ? document.getElementById("workspace-camera")
    : document.getElementById("camera-home");
  if (camera.parentElement !== host) host.appendChild(camera);
  // Away in the workspace, the sidebar's empty slot would still take a gap.
  document.getElementById("camera-home").hidden = WITH_CAMERA.includes(source);
  // Watching the robot is the point of these two; the sidebar's camera
  // keeps the choice the operator made.
  if (WITH_CAMERA.includes(source)) ensureCameraOn();
}
