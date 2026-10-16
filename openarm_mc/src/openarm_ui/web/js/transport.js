// Copyright 2026 Enactic, Inc.
// Licensed under the Apache License, Version 2.0.
//
// One WebSocket to the ROS node. Panels subscribe to message types rather
// than share a socket, so adding a panel does not mean touching this file.

let socket = null;
const handlers = new Map();

export function on(type, handler) {
  handlers.set(type, handler);
}

export function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
    return true;
  }
  return false;
}

/** Raw bytes (a piece of an uploaded model); -> false when closed. */
export function sendBytes(data) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(data);
    return true;
  }
  return false;
}

export const isOpen = () => Boolean(socket && socket.readyState === WebSocket.OPEN);

export function connect({ onOpen, onClose, retryMs = 2000 } = {}) {
  socket = new WebSocket(`ws://${location.host}/`);

  socket.addEventListener("open", () => onOpen && onOpen());
  socket.addEventListener("close", () => {
    if (onClose) onClose();
    setTimeout(() => connect({ onOpen, onClose, retryMs }), retryMs);
  });
  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      return;
    }
    const handler = handlers.get(message.type);
    if (handler) handler(message);
  });
}
