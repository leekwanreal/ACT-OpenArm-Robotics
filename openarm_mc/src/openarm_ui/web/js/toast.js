// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// One line of feedback at a time, bottom right.
//
// A connect that worked is worth saying out loud, not just leaving for the
// facts list to show, so success is its own kind rather than the plain
// styling an informational message gets.

let timer = null;

// Connecting reports which bus and how many motors answered, which is longer
// than a rejection and worth leaving up for longer.
const DWELL_MS = { info: 4000, ok: 6000, bad: 7000 };

/** `kind` is "info", "ok" or "bad". */
export function toast(text, kind = "info") {
  const host = document.getElementById("toast");
  host.textContent = text;
  host.className = `show ${kind}`;
  clearTimeout(timer);
  timer = setTimeout(() => { host.className = ""; }, DWELL_MS[kind] || 4000);
}
