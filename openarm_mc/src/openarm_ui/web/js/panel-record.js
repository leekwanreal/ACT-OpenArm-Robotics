// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Which cameras an episode records: chest, front, or both. The choice is
// remembered per browser and told to the server, which arms the next
// episode's recorder on those cameras ahead of time, so Start recording
// still answers at once. Each camera says where its frames come from: a
// RealSense, the virtual (black) camera standing in for one, or nothing --
// and a camera with no frames cannot be recorded.

import { send } from "./transport.js";

const STORE_KEY = "openarm.record.cameras";
const CAMERAS = ["chest", "front"];
const SOURCE_TEXT = { realsense: "RealSense", virtual: "virtual · black", null: "no frames" };

let chosen = ["chest"];
let sources = {};           // camera -> realsense | virtual | null, from the status

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (Array.isArray(saved) && saved.length && saved.every((c) => CAMERAS.includes(c))) {
      chosen = saved;
    }
  } catch (error) { /* keep the default */ }
}

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(chosen));
  } catch (error) { /* nothing to do */ }
}

const box = (camera) =>
  document.querySelector(`#record-cameras input[value="${camera}"]`);

/** The cameras to record, in role order. */
export const recordCameras = () => CAMERAS.filter((camera) => chosen.includes(camera));

/** Tell the server; again after a reconnect, which forgets it. */
export function sendRecordCameras() {
  send({ action: "record_cameras", cameras: recordCameras() });
}

/** Each camera's source, from the once-a-second status. */
export function renderCameraSources(status) {
  if (!status || !status.cameras) return;
  sources = status.cameras;
  CAMERAS.forEach((camera) => {
    const source = sources[camera] ?? null;
    const tag = document.querySelector(`#record-cameras [data-source="${camera}"]`);
    const text = SOURCE_TEXT[source];
    if (tag.textContent !== text) tag.textContent = text;
    tag.className = `record-source ${source || "none"}`;
  });
}

/** While an episode runs, the boxes show what it records and stay put. */
export function renderRecordCameras(recording, recordingCameras) {
  CAMERAS.forEach((camera) => {
    const input = box(camera);
    input.disabled = recording;
    const on = recording ? (recordingCameras || []).includes(camera)
                         : chosen.includes(camera);
    if (input.checked !== on) input.checked = on;
  });
}

export function initRecordCameras() {
  load();
  CAMERAS.forEach((camera) => {
    const input = box(camera);
    input.checked = chosen.includes(camera);
    input.addEventListener("change", () => {
      const next = CAMERAS.filter((c) => box(c).checked);
      if (!next.length) {          // one camera at least
        input.checked = true;
        return;
      }
      chosen = next;
      save();
      sendRecordCameras();
    });
  });
}
