// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// A goal for a gripper tip, as MoveIt's interactive marker sets one: drag the
// gizmo at the tip (arrows to move, rings to turn) and inverse kinematics
// poses the planned ghost -- and the sliders -- to reach it. Plan then makes
// a joint-space trajectory from where the robot is to the planned pose and
// hands it to the policy whose card is up -- openarm_act, or openarm_smolvla
// -- which draws it with its frames and runs it on Run, like a trajectory
// the model generated.

import { state } from "./state.js";
import { send } from "./transport.js";
import { toast } from "./toast.js";
import { currentWorkspace } from "./panel-workspace.js";
import { solveIK } from "./ik.js";
import { refreshSliderValues, updatePlanHint } from "./panel-joints.js";
import {
  applyPlanned, goalRobot, initGoal, setGoalArm, setGoalMode,
} from "./viewport.js";

const CONTROL_HZ = 50;          // openarm_act runs a plan one row per step
const PLAN_SPEED = 0.5;         // rad/s its farthest joint may average
const MIN_PLAN_S = 1.0;
// While dragging, a few iterations per move keep up with the mouse; each
// starts from the last answer, so it settles as the gizmo slows down.
const DRAG_ITERATIONS = 25;
const RELEASE_ITERATIONS = 150;
// Out of reach from where the drag left it, the release tries other starts.
const RELEASE_RESTARTS = 12;

const byId = (id) => document.getElementById(id);
let lastResult = null;

function report(result) {
  const text = !result ? "" : result.reached
    ? "reached"
    : `out of reach: ${(result.position * 1000).toFixed(0)} mm, ` +
      `${(result.rotation * 180 / Math.PI).toFixed(0)}° off`;
  const node = byId("goal-status");
  node.textContent = text;
  node.className = `goal-status ${result && !result.reached ? "bad" : "ok"}`;
}

function reach(arm, pose, iterations, restarts = 0) {
  const robot = goalRobot();
  if (!robot) return;
  const result = solveIK(robot, arm, pose, state.planned, iterations, restarts);
  Object.assign(state.planned, result.values);
  applyPlanned();
  refreshSliderValues();
  updatePlanHint();
  lastResult = result;
  report(result);
}

let lastPose = null;

function onMove(arm, pose) {
  lastPose = pose;
  reach(arm, pose, DRAG_ITERATIONS);
}

function onRelease(arm) {
  if (lastPose) reach(arm, lastPose, RELEASE_ITERATIONS, RELEASE_RESTARTS);
  if (lastResult && !lastResult.reached) {
    toast("The goal is out of reach; the plan stops at the closest pose.", "bad");
  }
}

/** Smoothstep of degree five: zero speed and acceleration at both ends. */
const ease = (t) => t * t * t * (10 + t * (-15 + 6 * t));

/** Rows of 16 values, one per control step, from the robot to the plan. */
function planRows() {
  const names = state.jointNames;
  const from = names.map((name) => state.latest[name]?.position);
  if (from.some((value) => value === undefined)) return null;
  const to = names.map((name, i) => state.planned[name] ?? from[i]);
  const farthest = Math.max(...names.map((name, i) =>
    name.endsWith("gripper") ? 0 : Math.abs(to[i] - from[i])));
  if (farthest < 1e-3) return [];
  const seconds = Math.max(MIN_PLAN_S, farthest / PLAN_SPEED);
  const steps = Math.round(seconds * CONTROL_HZ);
  const rows = [];
  for (let k = 0; k <= steps; k++) {
    const s = ease(k / steps);
    rows.push(from.map((value, i) => value + (to[i] - value) * s));
  }
  return rows;
}

function plan() {
  const rows = planRows();
  if (rows === null) {
    toast("No joint state yet: connect first.", "bad");
    return;
  }
  if (!rows.length) {
    toast("The goal is where the robot already is: move the gizmo first.", "bad");
    return;
  }
  // To the policy whose card is up; otherwise the server picks one that runs.
  const policy = ["act", "smolvla"].includes(currentWorkspace())
    ? currentWorkspace() : undefined;
  send({ action: "plan_trajectory", rows, policy });
}

function pick(selector, button) {
  document.querySelectorAll(selector).forEach((other) =>
    other.classList.toggle("active", other === button));
}

export function initGoalPanel() {
  if (!initGoal(onMove, onRelease)) {
    byId("goal-tools").hidden = true;   // no TransformControls: no gizmo
    return;
  }
  document.querySelectorAll("#goal-tools button[data-goal-arm]").forEach((button) => {
    button.addEventListener("click", () => {
      const arm = button.dataset.goalArm || null;
      pick("#goal-tools button[data-goal-arm]", button);
      setGoalArm(arm);
      lastResult = null;
      report(null);
    });
  });
  document.querySelectorAll("#goal-tools button[data-goal-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      pick("#goal-tools button[data-goal-mode]", button);
      setGoalMode(button.dataset.goalMode);
    });
  });
  byId("goal-plan").addEventListener("click", plan);
}

