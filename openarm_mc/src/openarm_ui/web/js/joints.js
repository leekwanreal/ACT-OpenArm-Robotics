// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The joint table, as openarm_can's panel defines it. Values are in the unit
// the topics carry: radians for the arm, metres of finger stroke for the
// grippers. The two arms are mirrored, so J1 and J2 have opposite ranges --
// one table for both would drive the right arm into limits it does not have.

export const JOINTS = {
  left_j1: { label: "Shoulder Pitch", motor: "DM8009", min: -3.4907, max: 1.3963 },
  left_j2: { label: "Shoulder Roll", motor: "DM8009", min: -3.3161, max: 0.1745 },
  left_j3: { label: "Arm Twist", motor: "DM4340", min: -1.5708, max: 1.5708 },
  left_j4: { label: "Elbow Pitch", motor: "DM4340", min: 0.0, max: 2.4435 },
  left_j5: { label: "Forearm Twist", motor: "DM4310", min: -1.5708, max: 1.5708 },
  left_j6: { label: "Wrist Pitch", motor: "DM4310", min: -0.7854, max: 0.7854 },
  left_j7: { label: "Wrist Roll", motor: "DM4310", min: -1.5708, max: 1.5708 },
  left_gripper: { label: "Gripper", motor: "DM4310", min: 0.0, max: 0.043 },

  right_j1: { label: "Shoulder Pitch", motor: "DM8009", min: -1.3963, max: 3.4907 },
  right_j2: { label: "Shoulder Roll", motor: "DM8009", min: -0.1745, max: 3.3161 },
  right_j3: { label: "Arm Twist", motor: "DM4340", min: -1.5708, max: 1.5708 },
  right_j4: { label: "Elbow Pitch", motor: "DM4340", min: 0.0, max: 2.4435 },
  right_j5: { label: "Forearm Twist", motor: "DM4310", min: -1.5708, max: 1.5708 },
  right_j6: { label: "Wrist Pitch", motor: "DM4310", min: -0.7854, max: 0.7854 },
  right_j7: { label: "Wrist Roll", motor: "DM4310", min: -1.5708, max: 1.5708 },
  right_gripper: { label: "Gripper", motor: "DM4310", min: 0.0, max: 0.043 },
};

// Elbows bent, grippers closed: the same home pose as openarm_can.
export const HOME_POSE = { j4: 0.75 };

export const ARM_STEP = 0.005;        // rad
export const GRIPPER_STEP = 0.0005;   // m, half a millimetre

export const GRIPPER_MAX_STROKE_M = 0.044;

export const isGripper = (name) => name.endsWith("gripper");

export function clamp(name, value) {
  const spec = JOINTS[name];
  if (!spec) return value;
  return Math.min(spec.max, Math.max(spec.min, value));
}

/** The value a URDF joint takes for this joint's topic value. */
export function urdfValue(name, value) {
  return isGripper(name)
    ? Math.min(GRIPPER_MAX_STROKE_M, Math.max(0, Math.abs(value)))
    : value;
}

/** Both units at once, the way the panel shows them. */
export function formatValue(name, value) {
  if (isGripper(name)) {
    const mm = value * 1000;
    const word = value < 0.002 ? "closed" : (value >= 0.040 ? "open" : "");
    return `${mm.toFixed(1)} mm${word ? ` (${word})` : ""}`;
  }
  return `${value.toFixed(3)} rad (${(value * 180 / Math.PI).toFixed(1)}°)`;
}

// The gains each joint carries, and the range the CAN frame can hold for
// each. Arm joints run MIT impedance control; the grippers run position-force,
// so their pair is a speed and a current cap instead.
export const GAIN_FIELDS = {
  kp: { label: "kp", min: 0, max: 500, step: 1, hint: "position gain" },
  kd: { label: "kd", min: 0, max: 5, step: 0.05, hint: "velocity gain" },
  speed: { label: "speed", min: 0, max: 65, step: 0.5, hint: "rad/s" },
  effort: { label: "effort", min: 0, max: 1, step: 0.01,
            hint: "current cap: low holds, high crushes" },
};

export const gainsFor = (name) =>
  isGripper(name) ? ["speed", "effort"] : ["kp", "kd"];
