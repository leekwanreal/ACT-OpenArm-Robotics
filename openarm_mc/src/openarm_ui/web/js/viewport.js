// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The 3D view: the robot as it is, the ghost of the pose the sliders plan
// (its own colour, --ghost-planned, so it cannot be taken for the robot),
// drawn only while the plan differs from the robot,
// and each policy's trajectory (ACT, SmolVLA): per gripper, its path over
// the 50 steps of the last chunk with an xyz frame every few steps. three.js and URDFLoader come from classic <script> tags, so
// they are globals here rather than imports.
//
// The robot is drawn as the simulator draws it: each COLLADA mesh of the
// URDF is swapped for openarm_mujoco's OBJ parts with their MuJoCo
// materials (matte black, silver), which the server lists at
// mujoco/visuals.json (mujoco_visuals.py). Without it, the COLLADA stays.

import { urdfValue } from "./joints.js";
import { tipFrame } from "./ik.js";
import { state } from "./state.js";

const URDF_PATH =
  "assets/openarm_v1/openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v1_codebase.urdf";
const TCP_LINKS = ["openarm_left_hand_tcp", "openarm_right_hand_tcp"];
// The trajectory path's colour per TCP link, and the frames' x, y, z, as
// style.css tokens.
const TRAJ_COLORS = [["--traj-left", "#0891b2"], ["--traj-right", "#c026d3"]];
const AXIS_COLORS = [["--axis-x", "#ff1f1f"], ["--axis-y", "#00c22a"],
                     ["--axis-z", "#1f5bff"]];

const VIEWS = {
  front: [0, 0.6, 1.6], left: [-1.6, 0.6, 0], right: [1.6, 0.6, 0],
  top: [0, 2.0, 0.01], reset: [1.1, 0.8, 1.1],
};

let scene, camera, renderer, controls;
let hemisphere = null;
let grid = null;
let robot = null;               // current
// name -> { robot, token, fallback, visible }
const ghosts = {
  planned: { robot: null, token: "--ghost-planned", fallback: "#10b981", visible: true },
};
// A copy of the robot that is never drawn: posed at each trajectory step to
// read where the grippers are.
let kinematics = null;
// Another, for inverse kinematics: posed many times per goal move.
let ikRobot = null;
// The IK goal: an empty object at the chosen gripper tip, moved with a
// MoveIt-style gizmo (arrows to move, rings to turn).
const goal = { arm: null, target: null, gizmo: null, dragging: false, onMove: null };
// policy -> { steps: the last trajectory, drawn once kinematics loads,
//            arrows: a THREE.Group per gripper (path and frames), visible }
const trajectories = {};
const trajectoryOf = (key) =>
  (trajectories[key] ??= { steps: [], arrows: [], visible: false });

// Each robot is about two million triangles of COLLADA, so the view is drawn
// only when something in it changed, never on a timer, and at no more than
// MAX_PIXEL_RATIO device pixels per CSS pixel.
const MAX_PIXEL_RATIO = 1.5;
let renderPending = false;
// The ghost shows only while the plan is this far from the robot on some
// joint (rad, m on a gripper): closer, it would only tint the robot.
const GHOST_MIN_GAP = 0.01;

// The trajectory, in metres. Points closer than minGap are one point, so an
// arm that holds still does not give the tube a zero-length segment. A frame
// every frameEvery steps and at the last; frames grow from firstScale to
// full size and fade in along the way, so the last is the boldest.
const TRAJ = { radius: 0.004, minGap: 0.0005, frameEvery: 5,
               axisLength: 0.05, axisRadius: 0.003, firstScale: 0.7,
               firstOpacity: 0.6 };

const token = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim()
    || fallback;

export function initViewport(onError) {
  const host = document.getElementById("viewport");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef1f5);   // repainted by applyTheme

  camera = new THREE.PerspectiveCamera(
    45, host.clientWidth / host.clientHeight, 0.01, 50);
  camera.position.set(1.1, 0.8, 1.1);

  const pixelRatio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
  // A dense screen smooths edges by itself; multisampling it as well costs
  // a slow GPU most of its frame.
  renderer = new THREE.WebGLRenderer({ antialias: pixelRatio <= 1 });
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(host.clientWidth, host.clientHeight);
  host.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.35, 0);
  controls.addEventListener("change", requestRender);
  controls.update();

  hemisphere = new THREE.HemisphereLight(0xffffff, 0xb8c2cc, 1.0);
  scene.add(hemisphere);
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(2, 3, 2);
  scene.add(key);
  applyTheme();

  // The simulator's look first, if the server has it; the COLLADA if not.
  loadMujocoLook().then(() => loadRobots(onError));

  window.addEventListener("resize", () => {
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(host.clientWidth, host.clientHeight);
    requestRender();
  });
  requestRender();
}

/** Draw the view once, on the next animation frame. Cheap to call often. */
export function requestRender() {
  if (renderPending || !renderer) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    renderer.render(scene, camera);
  });
}

function loadRobots(onError) {
  // The same model three times: independent URDFRobots are simpler than
  // cloning one, whose joint machinery does not survive a deep copy.
  loadOne(onError, (loaded) => {
    scene.add(loaded);
    robot = loaded;
    applyCurrent();
    requestRender();
  }, requestRender);
  Object.entries(ghosts).forEach(([name, ghost]) => {
    loadOne(onError, (loaded) => {
      loaded.visible = false;     // until the plan moves off the robot
      scene.add(loaded);
      ghost.robot = loaded;
      if (name === "planned") applyPlanned();
    }, (mesh) => {
      tint(mesh, ghost);
      requestRender();
    });
  });
  // Only its frames are read, so it gets no meshes at all.
  loadOne(onError, (loaded) => {
    kinematics = loaded;
    // Those that arrived while the meshes loaded.
    Object.keys(trajectories).forEach(drawTrajectory);
  }, null, false);
  loadOne(onError, (loaded) => { ikRobot = loaded; }, null, false);
}

/**
 * The goal gizmo. `onMove(arm, {p, q})` runs while it is dragged, with the
 * goal in the world frame; `onRelease(arm)` once it is let go.
 */
export function initGoal(onMove, onRelease) {
  if (!THREE.TransformControls || !scene) return false;
  goal.target = new THREE.Object3D();
  scene.add(goal.target);
  goal.gizmo = new THREE.TransformControls(camera, renderer.domElement);
  goal.gizmo.setSize(0.9);
  goal.gizmo.setSpace("local");         // as MoveIt's marker: along the tip
  goal.gizmo.addEventListener("change", requestRender);
  goal.gizmo.addEventListener("dragging-changed", (event) => {
    goal.dragging = event.value;
    controls.enabled = !event.value;    // the view must not orbit meanwhile
    if (!event.value && goal.arm) {
      onRelease(goal.arm);
      placeGoal();                      // onto what IK could reach
    }
  });
  goal.gizmo.addEventListener("objectChange", () => {
    if (!goal.arm) return;
    goal.target.updateMatrixWorld(true);
    onMove(goal.arm, {
      p: goal.target.getWorldPosition(new THREE.Vector3()),
      q: goal.target.getWorldQuaternion(new THREE.Quaternion()),
    });
  });
  scene.add(goal.gizmo);
  return true;
}

/** The undrawn robot IK poses, or null until it has loaded. */
export const goalRobot = () => ikRobot;

/** Put the goal on `arm`'s tip as planned (null hides the gizmo). */
export function setGoalArm(arm) {
  if (!goal.gizmo) return;
  goal.arm = arm;
  if (arm) {
    goal.gizmo.attach(goal.target);
    placeGoal();
  } else {
    goal.gizmo.detach();
  }
  requestRender();
}

export function setGoalMode(mode) {
  if (!goal.gizmo) return;
  goal.gizmo.setMode(mode);             // "translate" or "rotate"
  requestRender();
}

/** Move the goal onto the planned pose's tip, unless it is being dragged. */
function placeGoal() {
  if (!goal.arm || goal.dragging || !ikRobot) return;
  applyPose(ikRobot, state.planned, (value) => value);
  ikRobot.updateMatrixWorld(true);
  const link = ikRobot.links[`openarm_${goal.arm}_hand_tcp`];
  if (!link) return;
  tipFrame(link, goal.target.position, goal.target.quaternion);
  goal.target.updateMatrixWorld(true);
  requestRender();
}

/**
 * Each mesh file, parsed once for all three robots. The visual meshes are
 * about 70 MB of COLLADA; loading and parsing them per robot tripled both
 * the download and the time the tab spent in the parser. A robot gets a
 * clone, which shares the geometry and so the GPU memory as well.
 */
const meshCache = new Map();    // path -> Promise<Object3D | null>

// {materials: {name: {rgba, shininess, reflectance}}, meshes: {"link4.dae":
// [{file, material}]}} from the server, or null to keep the COLLADA.
let mujoco = null;
const mujocoMaterials = new Map();   // name -> THREE.Material, shared

async function loadMujocoLook() {
  try {
    const reply = await fetch("mujoco/visuals.json");
    if (reply.ok && typeof THREE.OBJLoader === "function") mujoco = await reply.json();
  } catch (error) { /* keep the COLLADA */ }
}

/** MuJoCo's material as Phong: its shininess and reflectance as a
 * highlight, so the silver catches the light and the black stays matte. */
function mujocoMaterial(name) {
  if (!mujocoMaterials.has(name)) {
    const spec = (mujoco.materials[name]) ||
      { rgba: [0.7, 0.7, 0.7, 1], shininess: 0.5, reflectance: 0 };
    const [r, g, b, a] = spec.rgba;
    const highlight = 0.1 + 0.5 * spec.reflectance;
    mujocoMaterials.set(name, new THREE.MeshPhongMaterial({
      color: new THREE.Color(r, g, b),
      specular: new THREE.Color(highlight, highlight, highlight),
      shininess: 20 + 80 * spec.shininess,
      transparent: a < 1, opacity: a,
    }));
  }
  return mujocoMaterials.get(name);
}

/** One .dae as the simulator's parts, parsed once per file and shared. */
function mujocoMesh(parts, done) {
  const loader = new THREE.OBJLoader();
  Promise.all(parts.map((part) => {
    const path = `mujoco/${part.file}`;
    if (!meshCache.has(path)) {
      meshCache.set(path, new Promise((resolve) => {
        loader.load(path, resolve, undefined, () => resolve(null));
      }));
    }
    return meshCache.get(path).then((loaded) => {
      if (!loaded) return null;
      const copy = loaded.clone();
      const material = mujocoMaterial(part.material);
      copy.traverse((item) => { if (item.isMesh) item.material = material; });
      return copy;
    });
  })).then((copies) => {
    if (copies.some((copy) => !copy)) return done(null);
    // A Group, not a Mesh: URDFLoader keeps a Group's own materials.
    const group = new THREE.Group();
    copies.forEach((copy) => group.add(copy));
    done(group);
  });
}

function sharedMesh(loader, path, manager, done) {
  if (!meshCache.has(path)) {
    meshCache.set(path, new Promise((resolve) => {
      loader.defaultMeshLoader(path, manager, (mesh) => resolve(mesh || null));
    }));
  }
  meshCache.get(path).then((mesh) => done(mesh ? mesh.clone() : null));
}

/**
 * `onLoad` runs as soon as the URDF is parsed; the meshes arrive after it,
 * one by one, and `onMesh` sees each as it lands -- after URDFLoader has
 * given it the URDF's material, so a restyle made there sticks. (Waiting for
 * the LoadingManager does not work: three.js folds requests for the same
 * file into one and only the first loader's manager hears of it, so a ghost
 * would be "complete" with no meshes, and look like the robot.)
 */
function loadOne(onError, onLoad, onMesh, meshes = true) {
  const loader = new URDFLoader(new THREE.LoadingManager());
  loader.packages = {
    openarm_description: "assets/openarm_v1/openarm_description",
  };
  loader.loadMeshCb = (path, manager, done) => {
    if (!meshes) {
      done(null);
      return;
    }
    const parts = mujoco && mujoco.meshes[path.split("/").pop()];
    const deliver = (mesh) => {
      done(mesh);
      if (mesh && onMesh) onMesh(mesh);
    };
    if (parts) mujocoMesh(parts, deliver);
    else sharedMesh(loader, path, manager, deliver);
  };
  loader.load(URDF_PATH, (result) => {
    // URDF is z-up, three.js is y-up.
    result.rotation.x = -Math.PI / 2;
    Object.values(result.joints).forEach((joint) => { joint.ignoreLimits = true; });
    onLoad(result);
  }, undefined, onError);
}

/** Give every mesh of a ghost its translucent colour. Safe to repeat. */
function tint(target, ghost) {
  if (!target) return;
  const color = new THREE.Color(token(ghost.token, ghost.fallback));
  target.traverse((item) => {
    if (!item.isMesh) return;
    const sources = Array.isArray(item.material) ? item.material : [item.material];
    const materials = sources.map((source) => {
      const material = source.userData.ghost ? source : source.clone();
      material.userData.ghost = true;
      material.transparent = true;
      material.opacity = 0.38;
      material.depthWrite = false;      // so the solid arm reads through it
      material.color = color.clone();
      if (material.emissive) material.emissive = color.clone().multiplyScalar(0.25);
      return material;
    });
    item.material = materials.length === 1 ? materials[0] : materials;
    item.renderOrder = 1;
  });
}

// A joint that moved less than this (rad, m on a gripper) is left where it
// is drawn: the arm holding still jitters by about that much, and redrawing
// for it would cost a frame per state message for nothing anyone can see.
const MIN_VISIBLE_STEP = 1e-4;

/** Pose `target`; -> whether any joint moved visibly. */
function applyPose(target, pose, read) {
  if (!target) return false;
  let moved = false;
  for (const [name, entry] of Object.entries(pose)) {
    const joint = target.joints[name];
    if (!joint) continue;
    const value = urdfValue(name, read(entry));
    const now = joint.jointValue ? joint.jointValue[0] : undefined;
    if (now !== undefined && Math.abs(now - value) < MIN_VISIBLE_STEP) continue;
    moved = target.setJointValue(name, value) || moved;
  }
  return moved;
}

/** Whether the plan is far enough from the robot to be worth a ghost. */
function planDiffers() {
  return Object.entries(state.planned).some(([name, value]) => {
    const joint = state.latest[name];
    return joint && Math.abs(joint.position - value) > GHOST_MIN_GAP;
  });
}

function refreshGhost() {
  const ghost = ghosts.planned;
  if (!ghost.robot) return;
  const visible = ghost.visible && planDiffers();
  if (ghost.robot.visible === visible) return;
  ghost.robot.visible = visible;
  requestRender();
}

export function applyCurrent() {
  if (applyPose(robot, state.latest, (joint) => joint.position)) requestRender();
  refreshGhost();
}

export function applyPlanned() {
  if (applyPose(ghosts.planned.robot, state.planned, (value) => value)) {
    requestRender();
  }
  refreshGhost();
  placeGoal();                  // a slider moved the plan: the goal follows
}

function disposeArrow(group) {
  scene.remove(group);
  group.traverse((part) => {
    if (part.geometry) part.geometry.dispose();
    if (part.material) part.material.dispose();
  });
}

/** A material for the trajectory, drawn over the robot. */
function overlay(color, opacity) {
  return new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthTest: false });
}

/**
 * The xyz frame of one step: a thin rod along each axis of `quaternion`,
 * from `position`. Each rod knows its axis, so a theme change can recolour it.
 */
function buildFrame(position, quaternion, scale, opacity) {
  const frame = new THREE.Group();
  const length = TRAJ.axisLength * scale;
  const geometry = new THREE.CylinderGeometry(
    TRAJ.axisRadius, TRAJ.axisRadius, length, 6);
  // CylinderGeometry runs along +y, centred: turn it onto each axis.
  const turns = [new THREE.Euler(0, 0, -Math.PI / 2), new THREE.Euler(0, 0, 0),
                 new THREE.Euler(Math.PI / 2, 0, 0)];
  const units = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0),
                 new THREE.Vector3(0, 0, 1)];
  turns.forEach((turn, axis) => {
    const rod = new THREE.Mesh(
      axis ? geometry.clone() : geometry,
      overlay(new THREE.Color(token(...AXIS_COLORS[axis])), opacity));
    rod.rotation.copy(turn);
    rod.position.copy(units[axis]).multiplyScalar(length / 2);
    rod.userData.axis = axis;
    frame.add(rod);
  });
  frame.position.copy(position);
  frame.quaternion.copy(quaternion);
  return frame;
}

/**
 * One gripper's trajectory through `poses` ({position, quaternion} of its
 * TCP, one per step): a tube along the path, in the gripper's colour, and
 * an xyz frame every TRAJ.frameEvery steps and at the last step.
 */
function buildTrajectory(poses, color) {
  const group = new THREE.Group();
  const points = poses.map((pose) => pose.position);
  const distinct = [points[0]];
  points.forEach((point) => {
    if (point.distanceTo(distinct[distinct.length - 1]) >= TRAJ.minGap) {
      distinct.push(point);
    }
  });
  if (distinct.length >= 2) {
    const path = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3(distinct),
                             Math.max(8, distinct.length * 2), TRAJ.radius, 6, false),
      overlay(color, 0.9));
    path.userData.path = true;
    group.add(path);
  }

  const last = poses.length - 1;
  poses.forEach((pose, step) => {
    if (step % TRAJ.frameEvery && step !== last) return;
    const along = last ? step / last : 1;
    group.add(buildFrame(
      pose.position, pose.quaternion,
      TRAJ.firstScale + (1 - TRAJ.firstScale) * along,
      TRAJ.firstOpacity + (1 - TRAJ.firstOpacity) * along));
  });

  group.renderOrder = 2;
  group.traverse((part) => { part.renderOrder = 2; });
  return group;
}

/**
 * Draw where the grippers go over the last predicted chunk.
 * `steps` is a list of {name: value} poses, one per control step (50 for
 * the ACT model). Each gripper's points are found by posing the undrawn
 * kinematics copy at every step and reading its TCP link; until that copy
 * has loaded the steps are kept and drawn when it has.
 */
export function setTrajectory(key, steps) {
  trajectoryOf(key).steps = steps || [];
  drawTrajectory(key);
}

function drawTrajectory(key) {
  const trajectory = trajectoryOf(key);
  if (trajectory.arrows.length) requestRender();
  trajectory.arrows.forEach(disposeArrow);
  trajectory.arrows = [];
  if (!kinematics || !trajectory.steps.length) return;
  requestRender();

  const poses = TCP_LINKS.map(() => []);
  trajectory.steps.forEach((pose) => {
    applyPose(kinematics, pose, (value) => value);
    kinematics.updateMatrixWorld(true);
    TCP_LINKS.forEach((link, index) => {
      const frame = kinematics.links[link];
      if (frame) {
        const tip = tipFrame(frame);
        poses[index].push({ position: tip.p, quaternion: tip.q });
      }
    });
  });

  poses.forEach((path, index) => {
    if (!path.length) return;
    const arrow = buildTrajectory(path, new THREE.Color(token(...TRAJ_COLORS[index])));
    arrow.userData.colorIndex = index;
    arrow.visible = trajectory.visible;
    scene.add(arrow);
    trajectory.arrows.push(arrow);
  });
}

export function setTrajectoryVisible(key, visible) {
  // Called with every state message; only a change needs a frame.
  const trajectory = trajectoryOf(key);
  if (visible === trajectory.visible) return;
  trajectory.visible = visible;
  trajectory.arrows.forEach((arrow) => { arrow.visible = visible; });
  requestRender();
}

/** Repaint the scene from the theme's tokens. */
export function applyTheme() {
  if (!scene || !hemisphere) return;
  scene.background = new THREE.Color(token("--scene-bg", "#eef1f5"));
  hemisphere.groundColor = new THREE.Color(token("--scene-ground", "#b8c2cc"));
  // A GridHelper's colours are baked into its geometry, so swap the grid.
  if (grid) {
    scene.remove(grid);
    grid.geometry.dispose();
    grid.material.dispose();
  }
  grid = new THREE.GridHelper(4, 20, new THREE.Color(token("--scene-grid", "#aab4be")),
                              new THREE.Color(token("--scene-grid-minor", "#d4dae0")));
  scene.add(grid);
  Object.values(ghosts).forEach((ghost) => {
    if (ghost.robot) tint(ghost.robot, ghost);
  });
  Object.values(trajectories).flatMap((t) => t.arrows).forEach((arrow) => {
    const color = new THREE.Color(token(...TRAJ_COLORS[arrow.userData.colorIndex]));
    arrow.traverse((part) => {
      if (!part.material) return;
      if (part.userData.path) part.material.color = color;
      else if (part.userData.axis !== undefined) {
        part.material.color = new THREE.Color(token(...AXIS_COLORS[part.userData.axis]));
      }
    });
  });
  requestRender();
}

export function setPlannedVisible(visible) {
  ghosts.planned.visible = visible;
  refreshGhost();
}

export function setView(name) {
  const position = VIEWS[name];
  if (!position) return;
  camera.position.set(...position);
  controls.target.set(0, 0.35, 0);
  controls.update();
  requestRender();
}
