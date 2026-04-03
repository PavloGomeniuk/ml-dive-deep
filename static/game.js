import init, { Game } from './pkg/game.js';

const canvas = document.getElementById('canvas');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');

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

    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const scaleX = canvas.width / r.width;
      const scaleY = canvas.height / r.height;
      game.on_click((e.clientX - r.left) * scaleX, (e.clientY - r.top) * scaleY);
    });

  } catch (err) {
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'Failed to load WASM:\n' + err;
  }
}

start();
