// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// What every panel reads and writes.
//
// Two poses, two robots on screen: `latest` is what joint_states reports and
// drives the solid arm; `planned` is what the sliders ask for and drives the
// ghost. Nothing crosses from planned to the robot until Execute, the way a
// motion planner stages a goal before running it.

export const state = {
  jointNames: [],
  latest: {},    // name -> {name, position, velocity, effort}
  planned: {},   // name -> number
  // Every joint starts locked, as openarm_can's panel does: a slider cannot
  // stage anything until it is deliberately switched on.
  locked: {},
  gains: {},     // name -> {kp, kd} or {speed, effort}, for the active profile
  profiles: [],
  activeProfile: null,
  showGains: false,
  recording: false,
  // Which of real and sim the next connect is for. Null until a status
  // message says what is running or the operator picks one: preselecting
  // would make Connect move a robot nobody chose.
  mode: null,
  modePicked: false,
  arm: "left",
  sliderHeld: null,
  status: null,
};

export const isUnlocked = (name) => state.locked[name] === false;

export const unlockedJoints = () => state.jointNames.filter(isUnlocked);

/** How far the plan is from the robot, over the joints that would move. */
export function planError() {
  let worst = 0;
  for (const name of state.jointNames) {
    if (!isUnlocked(name)) continue;
    const current = state.latest[name];
    if (!current || state.planned[name] === undefined) continue;
    worst = Math.max(worst, Math.abs(state.planned[name] - current.position));
  }
  return worst;
}
