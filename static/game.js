// Medieval RPG v3 — Bevy/WASM entry + Web Audio synthesis
// Bevy owns the game loop. JS handles audio (no Rust audio crate needed).

import init from './pkg/game.js';

const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const canvas = document.getElementById('canvas');

// ── Web Audio ────────────────────────────────────────────────────────────────

let audioCtx = null;

function getAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function masterGain(ctx, vol = 0.3) {
  const g = ctx.createGain();
  g.gain.value = vol;
  g.connect(ctx.destination);
  return g;
}

window.__playHitSound = function () {
  try {
    const ctx = getAudio();
    const osc = ctx.createOscillator();
    const g = masterGain(ctx);
    osc.type = 'square';
    osc.frequency.setValueAtTime(200, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(50, ctx.currentTime + 0.08);
    osc.connect(g);
    osc.start();
    osc.stop(ctx.currentTime + 0.08);
  } catch (_) {}
};

window.__playDeathSound = function () {
  try {
    const ctx = getAudio();
    const osc = ctx.createOscillator();
    const g = masterGain(ctx);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(120, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(40, ctx.currentTime + 0.3);
    g.gain.setValueAtTime(0.3, ctx.currentTime);
    g.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.3);
    osc.connect(g);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (_) {}
};

window.__playEquipSound = function () {
  try {
    const ctx = getAudio();
    const notes = [261.63, 329.63, 392.0]; // C4, E4, G4
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      g.gain.value = 0.15;
      g.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.value = freq;
      osc.connect(g);
      const t = ctx.currentTime + i * 0.05;
      osc.start(t);
      osc.stop(t + 0.07);
    });
  } catch (_) {}
};

window.__playTransitionSound = function () {
  try {
    const ctx = getAudio();
    const chord = [130.81, 164.81, 196.0]; // C3, E3, G3
    chord.forEach(freq => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, ctx.currentTime);
      g.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.1);
      g.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.27);
      g.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.value = freq;
      osc.connect(g);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.27);
    });
  } catch (_) {}
};

// ── Boot ─────────────────────────────────────────────────────────────────────

async function start() {
  try {
    // init() calls #[wasm_bindgen(start)] which runs Bevy's App::run().
    // Bevy sets up its own requestAnimationFrame loop internally.
    await init();
    loading.style.display = 'none';
    canvas.focus();
  } catch (err) {
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'Failed to load WASM:\n' + err;
  }
}

// Fullscreen toggle
fullscreenBtn.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    canvas.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
});

document.addEventListener('fullscreenchange', () => {
  fullscreenBtn.textContent = document.fullscreenElement ? '\u2715' : '\u26F6';
});

start();
