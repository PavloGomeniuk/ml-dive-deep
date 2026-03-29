// Opening crawl engine
// Usage: CrawlEngine.start({ headlines, theme, onComplete })
(function () {
  const CRAWL_DURATION_MS = 12000; // 12 seconds for full crawl
  const LOGO_LABELS = {
    matrix:   'THE MATRIX NEWS FEED',
    starwars: 'STAR WARS — THE DAILY DISPATCH',
    lotr:     'THE RED BOOK OF WESTMARCH',
  };

  let rafId = null;
  let skipHandlerKey = null;
  let skipHandlerClick = null;
  let startTime = null;
  let onCompleteCallback = null;
  let isRunning = false;

  const crawlScreen  = document.getElementById('crawl-screen');
  const crawlWrapper = document.getElementById('crawl-wrapper');
  const crawlText    = document.getElementById('crawl-text');
  const rainCanvas   = document.getElementById('matrix-rain');

  function buildCrawlHTML(headlines, theme) {
    const logo = LOGO_LABELS[theme] || 'CINEMATIC NEWS';
    let html = `<span class="crawl-logo">${logo}</span>`;
    const items = headlines.slice(0, 5);
    items.forEach((h, i) => {
      html += `
        <div class="crawl-item">
          <span class="crawl-index">&#8212; ${i + 1} &#8212;</span>
          ${h.title}
        </div>`;
    });
    return html;
  }

  function getScrollDistance() {
    // Total distance the text needs to travel: off-bottom to off-top
    return crawlText.scrollHeight + window.innerHeight;
  }

  function animate(timestamp) {
    if (!isRunning) return;

    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;
    const progress = Math.min(elapsed / CRAWL_DURATION_MS, 1);

    const totalDist = getScrollDistance();
    // Translate from 100vh (below screen) to -scrollHeight (above screen)
    const translateY = window.innerHeight - progress * totalDist;
    crawlText.style.transform = `translateY(${translateY}px)`;

    if (progress < 1) {
      rafId = requestAnimationFrame(animate);
    } else {
      finish();
    }
  }

  function finish() {
    isRunning = false;
    removeSkipListeners();
    if (onCompleteCallback) onCompleteCallback();
  }

  function skip() {
    if (!isRunning) return;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    finish();
  }

  function addSkipListeners() {
    skipHandlerKey = function (e) {
      // Ignore modifier-only keypresses
      if (['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) return;
      skip();
    };
    skipHandlerClick = function () { skip(); };
    document.addEventListener('keydown', skipHandlerKey);
    crawlScreen.addEventListener('click', skipHandlerClick);
  }

  function removeSkipListeners() {
    if (skipHandlerKey) document.removeEventListener('keydown', skipHandlerKey);
    if (skipHandlerClick) crawlScreen.removeEventListener('click', skipHandlerClick);
    skipHandlerKey = null;
    skipHandlerClick = null;
  }

  function start({ headlines, theme, onComplete }) {
    // Stop any previous run
    if (isRunning) stop();

    isRunning = true;
    startTime = null;
    onCompleteCallback = onComplete || null;

    // Build content
    crawlText.innerHTML = buildCrawlHTML(headlines, theme);
    // Reset position
    crawlText.style.transform = `translateY(${window.innerHeight}px)`;

    // Start Matrix rain if needed
    if (theme === 'matrix') {
      MatrixRain.start(rainCanvas);
    }

    addSkipListeners();
    rafId = requestAnimationFrame(animate);
  }

  function stop() {
    isRunning = false;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    removeSkipListeners();
    MatrixRain.stop();
    startTime = null;
  }

  window.CrawlEngine = { start, stop };
})();
