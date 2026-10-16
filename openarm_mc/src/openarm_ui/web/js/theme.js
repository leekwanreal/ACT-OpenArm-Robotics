// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// Light and dark. The page follows the system until the switch in the header
// is pressed; from then on the choice is remembered per browser. The colours
// themselves are CSS tokens (style.css); this only sets data-theme on <html>
// and tells whoever paints outside CSS -- the 3D view -- to repaint.

const STORE_KEY = "openarm.theme";
const listeners = [];
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function saved() {
  try {
    const value = localStorage.getItem(STORE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch (error) {
    return null;
  }
}

/** "light" or "dark", whichever the page is showing now. */
export function currentTheme() {
  return saved() || (systemDark.matches ? "dark" : "light");
}

/** A CSS token's value, e.g. cssToken("--scene-bg"). */
export function cssToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Call `listener(theme)` now and on every change. */
export function onThemeChange(listener) {
  listeners.push(listener);
  listener(currentTheme());
}

function apply(theme) {
  document.documentElement.dataset.theme = theme;
  const button = document.getElementById("btn-theme");
  if (button) {
    // The button names the theme it switches to.
    const next = theme === "dark" ? "Light" : "Dark";
    button.textContent = next;
    button.title = `Switch to ${next.toLowerCase()} mode`;
  }
  listeners.forEach((listener) => listener(theme));
}

export function initTheme() {
  apply(currentTheme());
  document.getElementById("btn-theme").addEventListener("click", () => {
    const theme = currentTheme() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(STORE_KEY, theme);
    } catch (error) { /* still switch for this visit */ }
    apply(theme);
  });
  // Without a saved choice the page keeps following the system.
  systemDark.addEventListener("change", () => {
    if (!saved()) apply(currentTheme());
  });
}
