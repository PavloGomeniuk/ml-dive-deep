import init, { Game } from './pkg/game.js';

const canvas = document.getElementById('canvas');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const fullscreenBtn = document.getElementById('fullscreen-btn');

// ── Web Audio ────────────────────────────────────────────────────────────────

let audioCtx = null;

function getAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function masterGain(ctx) {
  const g = ctx.createGain();
  g.gain.value = 0.3;
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

// ── Game loop ────────────────────────────────────────────────────────────────

async function start() {
  try {
    await init();
    const game = new Game();

    loading.style.display = 'none';

    let last = performance.now();
    function frame(ts) {
      const dt = Math.min((ts - last) / 1000, 0.05);
      last = ts;
      game.tick(dt);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    // Click handler with coordinate scaling
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const scaleX = canvas.width / r.width;
      const scaleY = canvas.height / r.height;
      game.on_click((e.clientX - r.left) * scaleX, (e.clientY - r.top) * scaleY);
    });

    // Keyboard handler
    document.addEventListener('keydown', (e) => {
      // Prevent space from scrolling the page
      if (e.key === ' ') e.preventDefault();
      game.on_key(e.key);
    });

    // Fullscreen button
    fullscreenBtn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        canvas.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    // Update fullscreen button icon based on state
    document.addEventListener('fullscreenchange', () => {
      fullscreenBtn.textContent = document.fullscreenElement ? '\u2715' : '\u26F6';
    });

  } catch (err) {
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'Failed to load WASM:\n' + err;
  }
}

start();
