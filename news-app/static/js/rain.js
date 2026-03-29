// Matrix digital rain — 30fps capped canvas animation
(function () {
  const CHARS = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF';
  const FONT_SIZE = 14;
  const FPS_CAP = 30;
  const FRAME_MS = 1000 / FPS_CAP;

  let canvas, ctx, columns, drops;
  let rafId = null;
  let lastTime = 0;

  function init(c) {
    canvas = c;
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
  }

  function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    columns = Math.floor(canvas.width / FONT_SIZE);
    drops = Array.from({ length: columns }, () => Math.random() * -50);
  }

  function draw(timestamp) {
    if (!canvas) return;
    rafId = requestAnimationFrame(draw);

    if (timestamp - lastTime < FRAME_MS) return;
    lastTime = timestamp;

    // Fade trail
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#00ff41';
    ctx.font = FONT_SIZE + 'px Courier New';

    for (let i = 0; i < drops.length; i++) {
      const char = CHARS[Math.floor(Math.random() * CHARS.length)];
      const x = i * FONT_SIZE;
      const y = drops[i] * FONT_SIZE;

      // Lead character is brighter
      ctx.fillStyle = drops[i] > 0 ? '#afffbf' : '#00ff41';
      ctx.fillText(char, x, y);

      // Reset drop after it passes the screen (random chance)
      if (y > canvas.height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i] += 1;
    }
  }

  function start(canvasEl) {
    init(canvasEl);
    rafId = requestAnimationFrame(draw);
  }

  function stop() {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    window.removeEventListener('resize', resize);
    canvas = null;
    ctx = null;
  }

  window.MatrixRain = { start, stop };
})();
