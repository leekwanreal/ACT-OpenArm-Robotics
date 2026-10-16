// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// The cameras' RGB-D pairs, as the policies and the recorder see them: chest
// (on the robot) and front (across the table), each a colour and a depth
// stream, shown as a grid -- a row per camera, a column per stream.
//
// Off by default: frames cost bandwidth and the server only subscribes to a
// stream while some page is showing it. The choice is remembered per
// browser, and sent again whenever the socket reconnects, since a restarted
// server starts with every view off. A camera that is not plugged in is a
// virtual one (openarm_camera): black frames, and the caption says so.

import { send } from "./transport.js";

const STORE_KEY = "openarm.camera";
const CAMERAS = ["chest", "front"];
const KINDS = { rgb: ["rgb"], depth: ["depth"], both: ["rgb", "depth"] };
const PICKS = { chest: ["chest"], front: ["front"], both: CAMERAS };
const LABELS = { chest: "Chest", front: "Front", rgb: "RGB", depth: "Depth" };
// A frame older than this is shown as stale rather than as live.
const STALE_S = 1.0;

const view = { on: false, kinds: "rgb", cams: "chest" };

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    if (typeof saved.on === "boolean") view.on = saved.on;
    // "pick" is what the one-camera page called the streams.
    const kinds = saved.kinds ?? saved.pick;
    if (KINDS[kinds]) view.kinds = kinds;
    if (PICKS[saved.cams]) view.cams = saved.cams;
  } catch (error) { /* private window or blocked storage: keep defaults */ }
}

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(view));
  } catch (error) { /* nothing to do */ }
}

/** The streams shown: "chest/rgb", "front/depth", ... */
const shownStreams = () =>
  PICKS[view.cams].flatMap((cam) => KINDS[view.kinds].map((kind) => `${cam}/${kind}`));

const figure = (stream) =>
  document.querySelector(`.camera-view[data-stream="${stream}"]`);

function setEmpty(stream, text) {
  const host = figure(stream);
  host.querySelector(".camera-empty").textContent = text;
  host.classList.toggle("empty", Boolean(text));
}

/** One figure per camera and stream, built once; render() shows some. */
function build() {
  const host = document.getElementById("camera-views");
  host.innerHTML = CAMERAS.flatMap((cam) => ["rgb", "depth"].map((kind) => `
    <figure class="camera-view empty" data-stream="${cam}/${kind}" hidden>
      <img alt="${LABELS[cam]} ${kind === "rgb" ? "colour" : "depth, near red and far blue"}" />
      <span class="camera-tag">${LABELS[cam]} · ${LABELS[kind]}</span>
      <div class="camera-empty">Camera off</div>
      <figcaption></figcaption>
    </figure>`)).join("");
}

function render() {
  const button = document.getElementById("btn-camera");
  button.textContent = view.on ? "Turn off" : "Turn on";
  button.setAttribute("aria-pressed", String(view.on));
  button.classList.toggle("active", view.on);
  button.classList.toggle("on", !view.on);
  button.classList.toggle("off", view.on);

  document.querySelectorAll("#camera-streams button").forEach((choice) => {
    choice.classList.toggle("active", choice.dataset.streams === view.kinds);
  });
  document.querySelectorAll("#camera-cams button").forEach((choice) => {
    choice.classList.toggle("active", choice.dataset.cams === view.cams);
  });

  const shown = shownStreams();
  CAMERAS.forEach((cam) => ["rgb", "depth"].forEach((kind) => {
    const stream = `${cam}/${kind}`;
    const host = figure(stream);
    host.hidden = !shown.includes(stream);
    if (!view.on) {
      host.querySelector("img").removeAttribute("src");
      host.querySelector("figcaption").textContent = "";
      setEmpty(stream, "Camera off");
    } else if (!host.querySelector("img").getAttribute("src")) {
      setEmpty(stream, "Waiting for frames…");
    }
  }));
  const views = document.getElementById("camera-views");
  views.classList.toggle("off", !view.on);
  // The grid: a column per stream, a row per camera.
  views.style.setProperty("--camera-cols", String(KINDS[view.kinds].length));
  views.style.setProperty("--camera-rows", String(PICKS[view.cams].length));
}

/** Turn the view on for a workspace that is about the camera. Not saved:
 * the sidebar keeps the operator's own choice on the next load. */
export function ensureCameraOn() {
  if (view.on) return;
  view.on = true;
  render();
  requestCamera();
}

/** Tell the server what this page wants; call again after a reconnect. */
export function requestCamera() {
  send({ action: "camera", on: view.on, streams: shownStreams() });
}

export function onCamera(message) {
  if (!view.on) return;
  Object.entries(message.frames || {}).forEach(([stream, frame]) => {
    const host = figure(stream);
    if (!host) return;
    host.querySelector("img").src = `data:image/jpeg;base64,${frame.jpeg}`;
    host.dataset.size = `${frame.width}×${frame.height}`;
  });
  Object.entries(message.info || {}).forEach(([stream, info]) => {
    const host = figure(stream);
    if (!host) return;
    const caption = host.querySelector("figcaption");
    if (info.age === null || info.age === undefined) {
      caption.textContent = info.topic;
      setEmpty(stream, `No frames on ${info.topic}`);
      return;
    }
    const size = host.dataset.size ? `${host.dataset.size} · ` : "";
    caption.textContent = `${size}${info.fps.toFixed(0)} fps${info.virtual ? " · virtual" : ""}`;
    host.classList.toggle("virtual", Boolean(info.virtual));
    if (info.error) {
      setEmpty(stream, `Cannot show this frame: ${info.error}`);
    } else if (info.age > STALE_S) {
      setEmpty(stream, `Last frame ${info.age.toFixed(1)} s ago`);
    } else if (info.virtual) {
      setEmpty(stream, "Virtual camera: no RealSense plugged in for it");
    } else {
      setEmpty(stream, "");
    }
  });
}

export function initCamera() {
  load();
  build();
  // Chest and front the wrong way round: the server swaps them, at once.
  document.getElementById("btn-camera-switch").addEventListener("click", () => {
    send({ action: "camera_switch" });
  });
  document.getElementById("btn-camera").addEventListener("click", () => {
    view.on = !view.on;
    save();
    render();
    requestCamera();
  });
  const pick = (selector, key, attr) =>
    document.querySelectorAll(selector).forEach((choice) => {
      choice.addEventListener("click", () => {
        view[key] = choice.dataset[attr];
        save();
        render();
        if (view.on) requestCamera();
      });
    });
  pick("#camera-streams button", "kinds", "streams");
  pick("#camera-cams button", "cams", "cams");
  render();
}
