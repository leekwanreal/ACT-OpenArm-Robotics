// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Inverse kinematics for one arm: the seven joint values that put its gripper
// tip (TIP_OFFSET along openarm_<arm>_hand_tcp) at a goal pose.
//
// Damped least squares on the URDF the page already loads, with a numeric
// Jacobian: each iteration poses an undrawn copy of the robot eight times.
// Joint limits are the panel's (joints.js). Close enough is 1 mm and half a
// degree; out of reach, it returns the closest it got and says how far off.

import { JOINTS } from "./joints.js";

const STEP = 1e-4;              // rad, for the numeric Jacobian
const DAMPING = 0.05;           // lambda: keeps it steady near singularities
const MAX_DELTA = 0.2;          // rad a joint may move in one iteration
const ROT_WEIGHT = 0.3;         // m per rad: 0.3 rad off counts like 9 cm
const POS_TOL = 0.001;          // m
const ROT_TOL = 0.5 * Math.PI / 180;

export const armJoints = (arm) =>
  [1, 2, 3, 4, 5, 6, 7].map((index) => `${arm}_j${index}`);
export const tcpLink = (arm) => `openarm_${arm}_hand_tcp`;

// The URDF puts openarm_<arm>_hand_tcp at the wrist (joint 7's origin); the
// tip is further out along its z, between the finger pads: where the
// MuJoCo model puts its tcp, 6.5 cm past the fingers' joint (0.1025 m), a
// centimetre and a half short of the finger ends (0.183 m).
export const TIP_OFFSET = new THREE.Vector3(0, 0, 0.1675);

/** The gripper tip's world position and orientation, from the tcp link. */
export function tipFrame(link, position = new THREE.Vector3(),
                         orientation = new THREE.Quaternion()) {
  link.localToWorld(position.copy(TIP_OFFSET));
  link.getWorldQuaternion(orientation);
  return { p: position, q: orientation };
}

function tcpPose(robot, arm) {
  robot.updateMatrixWorld(true);
  return tipFrame(robot.links[tcpLink(arm)]);
}

/** [dx, dy, dz, rx, ry, rz]: from `pose` to the goal, rotation as axis*angle. */
function error(pose, goal) {
  const dp = goal.p.clone().sub(pose.p);
  const dq = goal.q.clone().multiply(pose.q.clone().invert());
  if (dq.w < 0) dq.set(-dq.x, -dq.y, -dq.z, -dq.w);   // the short way round
  const sin = Math.hypot(dq.x, dq.y, dq.z);
  const angle = 2 * Math.atan2(sin, dq.w);
  const k = sin > 1e-9 ? angle / sin : 2;
  return [dp.x, dp.y, dp.z, dq.x * k, dq.y * k, dq.z * k];
}

/** Solve A x = b for a small dense A (Gaussian elimination, pivoting). */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(M[r][c]) > Math.abs(M[pivot][c])) pivot = r;
    }
    [M[c], M[pivot]] = [M[pivot], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = M[r][n];
    for (let k = r + 1; k < n; k++) sum -= M[r][k] * x[k];
    x[r] = sum / M[r][r];
  }
  return x;
}

const weigh = (e) => e.map((v, i) => (i < 3 ? v : v * ROT_WEIGHT));
const clampJoint = (name, value) =>
  Math.min(JOINTS[name].max, Math.max(JOINTS[name].min, value));

/**
 * Joint values for `arm` that bring its tip to `goal` ({p: Vector3,
 * q: Quaternion}, world frame), starting from `seed` ({name: value}).
 * `robot` is an undrawn URDFRobot; it is left at the answer.
 *
 * From one start the solver can settle in a local minimum, a joint pinned
 * at its limit; with `restarts` it tries that many more starts inside the
 * joint ranges (the same ones each time) and keeps the closest answer.
 * -> {values: {name: value}, position: m off, rotation: rad off, reached}
 */
export function solveIK(robot, arm, goal, seed, iterations = 60, restarts = 0) {
  const names = armJoints(arm);
  let best = descend(robot, arm, goal, names.map((name) => seed[name] ?? 0), iterations);
  let state = 12345;                    // a fixed sequence: same goal, same answer
  const random = () => {
    state = (state * 16807) % 2147483647;
    return state / 2147483647;
  };
  for (let attempt = 0; attempt < restarts && !best.reached; attempt++) {
    const start = names.map((name) => {
      const { min, max } = JOINTS[name];
      return min + (max - min) * (0.15 + 0.7 * random());
    });
    const tried = descend(robot, arm, goal, start, iterations);
    if (tried.score < best.score) best = tried;
  }
  names.forEach((name, i) => robot.setJointValue(name, best.q[i]));
  robot.updateMatrixWorld(true);
  return {
    values: Object.fromEntries(names.map((name, i) => [name, best.q[i]])),
    position: best.posErr,
    rotation: best.rotErr,
    reached: best.reached,
  };
}

/** One damped-least-squares descent from `start`; -> the closest it got. */
function descend(robot, arm, goal, start, iterations) {
  const names = armJoints(arm);
  let q = names.map((name, i) => clampJoint(name, start[i]));
  const set = (values) => names.forEach((name, i) => robot.setJointValue(name, values[i]));

  let best = null;
  for (let iteration = 0; iteration <= iterations; iteration++) {
    set(q);
    const e = error(tcpPose(robot, arm), goal);
    const posErr = Math.hypot(e[0], e[1], e[2]);
    const rotErr = Math.hypot(e[3], e[4], e[5]);
    const score = posErr + ROT_WEIGHT * rotErr;
    if (!best || score < best.score) {
      best = { q: [...q], posErr, rotErr, score };
    }
    if ((posErr < POS_TOL && rotErr < ROT_TOL) || iteration === iterations) break;

    // Numeric Jacobian, 6 x 7, weighted like the error.
    const base = weigh(e);
    const J = [[], [], [], [], [], []];
    names.forEach((name, j) => {
      const nudged = [...q];
      nudged[j] += STEP;
      set(nudged);
      const ej = weigh(error(tcpPose(robot, arm), goal));
      // error = goal - pose, so d(pose)/dq = -(d error)/dq
      for (let r = 0; r < 6; r++) J[r][j] = (base[r] - ej[r]) / STEP;
    });

    // dq = J^T (J J^T + lambda^2 I)^-1 e
    const JJt = J.map((row, r) => J.map((other, c) =>
      row.reduce((sum, v, k) => sum + v * other[k], 0) + (r === c ? DAMPING ** 2 : 0)));
    const y = solve(JJt, base);
    let dq = names.map((_, j) => J.reduce((sum, row, r) => sum + row[j] * y[r], 0));
    const largest = Math.max(...dq.map(Math.abs));
    if (largest > MAX_DELTA) dq = dq.map((v) => v * MAX_DELTA / largest);
    q = q.map((v, j) => clampJoint(names[j], v + dq[j]));
  }

  best.reached = best.posErr < POS_TOL * 5 && best.rotErr < ROT_TOL * 4;
  return best;
}

/** The tip's pose in the world frame for the robot as it stands. */
export function tipPose(robot, arm) {
  return tcpPose(robot, arm);
}
