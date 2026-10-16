// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Load… in the ACT card: pick an .onnx or a .pth on this computer. The page
// sends it to the server in pieces over its socket; the server keeps it
// under checkpoint/, exports a .pth to .onnx (utils/export_onnx.py, about
// half a minute), and loads the .onnx into openarm_act (model_import.py).
// Its progress comes back with every state message.

import { isOpen, on, send, sendBytes } from "./transport.js";
import { toast } from "./toast.js";

const CHUNK = 4 * 1024 * 1024;       // under the server's 16 MiB frame limit
// Pieces sent and not yet acknowledged: the server writes each before it
// reads the next, and a deeper queue would stall its keepalive.
const IN_FLIGHT = 2;

const upload = {
  file: null,             // the file this page is sending
  sent: 0,
  acked: 0,               // bytes the server has written
  onAck: null,
  waiting: null,          // resolves with the server's answer to begin
  resultId: null,         // last.id already announced
  phase: "idle",
};

const byId = (id) => document.getElementById(id);
/** The next acknowledgement, or a second without one (then check again). */
const acked = () => new Promise((resolve, reject) => {
  if (!isOpen()) return reject(new Error("the connection closed"));
  const timer = setTimeout(resolve, 1000);
  upload.onAck = () => { clearTimeout(timer); resolve(); };
});
const mb = (bytes) => `${(bytes / 1e6).toFixed(0)} MB`;

export const importBusy = () => upload.phase !== "idle" || upload.file !== null;

async function send_file(file) {
  upload.file = file;
  upload.sent = 0;
  upload.acked = 0;
  render(null);
  const answer = new Promise((resolve) => { upload.waiting = resolve; });
  if (!send({ action: "model_upload_begin", name: file.name, size: file.size })) {
    upload.file = null;
    return toast("Not connected to the dashboard.", "bad");
  }
  const reply = await answer;
  if (!reply.ok) {
    upload.file = null;
    render(null);
    return toast(`Cannot load ${file.name}: ${reply.message}`, "bad");
  }
  try {
    for (let offset = 0; offset < file.size; offset += CHUNK) {
      while (offset - upload.acked >= IN_FLIGHT * CHUNK) await acked();
      if (!isOpen()) throw new Error("the connection closed");
      const piece = await file.slice(offset, offset + CHUNK).arrayBuffer();
      sendBytes(piece);
      upload.sent = Math.min(file.size, offset + CHUNK);
    }
    while (upload.acked < file.size) await acked();
    send({ action: "model_upload_end" });
  } catch (error) {
    toast(`Upload of ${file.name} stopped: ${error.message}`, "bad");
  } finally {
    upload.file = null;
  }
}

function describe(snapshot) {
  if (upload.file) {
    const share = upload.file.size ? upload.acked / upload.file.size : 0;
    return [`Uploading ${upload.file.name} · ${(share * 100).toFixed(0)}% of ` +
            `${mb(upload.file.size)}`, share];
  }
  if (!snapshot) return ["", 0];
  const seconds = snapshot.elapsed_s ? ` · ${snapshot.elapsed_s.toFixed(0)} s` : "";
  switch (snapshot.phase) {
    case "receiving":
      return [`Receiving ${snapshot.name} · ${mb(snapshot.received)}`,
              snapshot.size ? snapshot.received / snapshot.size : 0];
    case "saving": return [`Saving ${snapshot.name}…`, 1];
    case "exporting": {
      const last = (snapshot.log || []).slice(-1)[0] || "";
      return [`Exporting ${snapshot.name} to .onnx${seconds} · ${last}`, null];
    }
    case "loading": return [`Loading ${snapshot.name}…`, 1];
    default: return ["", 0];
  }
}

/** The import part of a state message. */
export function renderImport(snapshot) {
  if (snapshot) {
    upload.phase = snapshot.phase;
    const last = snapshot.last;
    if (last && last.id !== upload.resultId) {
      if (upload.resultId !== null) toast(last.message, last.ok ? "ok" : "bad");
      upload.resultId = last.id;
    } else if (!last && upload.resultId === null) {
      upload.resultId = 0;
    }
  }
  render(snapshot);
}

function render(snapshot) {
  const [text, share] = describe(snapshot);
  const box = byId("act-import");
  box.hidden = !text;
  const line = byId("act-import-text");
  if (line.textContent !== text) line.textContent = text;
  line.title = snapshot && snapshot.log ? snapshot.log.join("\n") : "";
  const bar = byId("act-import-bar");
  bar.classList.toggle("indeterminate", share === null);
  bar.firstElementChild.style.width = `${((share ?? 0) * 100).toFixed(1)}%`;
  const button = byId("btn-act-load");
  button.disabled = importBusy();
  const label = importBusy() ? "Loading…" : "Load…";
  if (button.textContent !== label) button.textContent = label;
}

export function initModelImport() {
  on("model_upload_ack", (message) => {
    upload.acked = message.received;
    render(null);
    if (upload.onAck) upload.onAck();
    upload.onAck = null;
  });
  on("model_upload", (message) => {
    if (upload.waiting) upload.waiting(message);
    upload.waiting = null;
  });
  const input = byId("act-model-file");
  byId("btn-act-load").addEventListener("click", () => {
    if (importBusy()) return;
    input.value = "";
    input.click();
  });
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const kind = file.name.toLowerCase().endsWith(".pth")
      ? "it is exported to .onnx first (about half a minute), then loaded"
      : "it is kept under checkpoint/onnx/ and loaded";
    toast(`${file.name}: ${kind}.`);
    send_file(file);
  });
}
