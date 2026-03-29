// Main application controller
(function () {

  // ── State ────────────────────────────────────────────────────────
  let currentTheme = null;
  let isLightMode = false;
  let newsData = null;
  const audio = document.getElementById('theme-audio');

  // ── DOM refs ─────────────────────────────────────────────────────
  const pickerScreen  = document.getElementById('theme-picker');
  const crawlScreen   = document.getElementById('crawl-screen');
  const newsScreen    = document.getElementById('news-screen');
  const siteTitle     = document.getElementById('site-title');
  const toggleBtn     = document.getElementById('toggle-btn');
  const backBtn       = document.getElementById('back-btn');
  const fetchedAt     = document.getElementById('fetched-at');
  const worldGrid     = document.getElementById('world-grid');
  const securityGrid  = document.getElementById('security-grid');

  const THEME_META = {
    matrix: {
      title:       'MATRIX NEWS FEED',
      css:         '/static/themes/matrix.css',
      audio:       '/static/audio/matrix.mp3',
      attribution: 'Music: royalty-free synthwave',
    },
    starwars: {
      title:       'GALACTIC DISPATCH',
      css:         '/static/themes/starwars.css',
      audio:       '/static/audio/starwars.mp3',
      attribution: 'Music: royalty-free fanfare',
    },
    lotr: {
      title:       'The Red Book of Westmarch',
      css:         '/static/themes/lotr.css',
      audio:       '/static/audio/lotr.mp3',
      attribution: 'Music: royalty-free orchestral',
    },
  };

  // ── Theme CSS loader ─────────────────────────────────────────────
  function loadThemeCSS(theme) {
    const link = document.getElementById('theme-css');
    link.href = THEME_META[theme].css;
  }

  function applyThemeClass(theme) {
    document.body.classList.remove('theme-matrix', 'theme-starwars', 'theme-lotr');
    if (theme) document.body.classList.add('theme-' + theme);
  }

  // ── Audio ────────────────────────────────────────────────────────
  function startAudio(theme) {
    const src = THEME_META[theme].audio;
    audio.src = src;
    audio.loop = true;
    audio.volume = 1.0;
    audio.play().catch(() => {
      // Autoplay blocked (shouldn't happen — triggered by user click)
      console.warn('Audio autoplay blocked');
    });
  }

  function fadeToLoopVolume() {
    // After crawl ends: drop to 30% volume
    const target = 0.3;
    const step = 0.02;
    const interval = setInterval(() => {
      if (audio.volume > target + step) {
        audio.volume = Math.max(target, audio.volume - step);
      } else {
        audio.volume = target;
        clearInterval(interval);
      }
    }, 50);
  }

  function stopAudio() {
    audio.pause();
    audio.src = '';
    audio.volume = 1.0;
  }

  // ── News fetch ───────────────────────────────────────────────────
  async function fetchNews() {
    try {
      const resp = await fetch('/api/news');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      newsData = await resp.json();
      return newsData;
    } catch (e) {
      console.error('Failed to fetch news:', e);
      return { world_news: [], web_security: [], fetched_at: null };
    }
  }

  // ── Render ───────────────────────────────────────────────────────
  function renderCard(article) {
    const card = document.createElement('div');
    card.className = 'news-card';
    const href = article.link && article.link !== '#' ? article.link : null;
    const titleEl = href
      ? `<a href="${href}" target="_blank" rel="noopener"><span class="news-card-title">${escHtml(article.title)}</span></a>`
      : `<span class="news-card-title">${escHtml(article.title)}</span>`;
    card.innerHTML = `
      ${titleEl}
      ${article.summary ? `<p class="news-card-summary">${escHtml(article.summary)}</p>` : ''}
      <p class="news-card-meta">${escHtml(article.source || '')}${article.published ? ' &middot; ' + fmtDate(article.published) : ''}</p>
    `;
    return card;
  }

  function renderNews(data) {
    worldGrid.innerHTML = '';
    securityGrid.innerHTML = '';

    const worldItems = (data.world_news || []);
    const secItems   = (data.web_security || []);

    if (worldItems.length === 0) {
      worldGrid.innerHTML = '<p class="news-empty">No world news available.</p>';
    } else {
      worldItems.forEach(a => worldGrid.appendChild(renderCard(a)));
    }

    if (secItems.length === 0) {
      securityGrid.innerHTML = '<p class="news-empty">No security news available.</p>';
    } else {
      secItems.forEach(a => securityGrid.appendChild(renderCard(a)));
    }

    if (data.fetched_at) {
      fetchedAt.textContent = 'Updated ' + fmtDate(data.fetched_at);
    }

    const attr = document.getElementById('audio-attribution');
    if (currentTheme && attr) {
      attr.textContent = THEME_META[currentTheme].attribution;
    }
  }

  // ── Theme selection ──────────────────────────────────────────────
  async function selectTheme(theme) {
    if (currentTheme && currentTheme !== theme) {
      stopAudio();
    }
    currentTheme = theme;

    loadThemeCSS(theme);
    applyThemeClass(theme);

    // Show crawl, hide others
    show(crawlScreen);
    hide(pickerScreen);
    hide(newsScreen);

    // Fetch news (parallel with crawl start is fine)
    const data = await fetchNews();

    // Update title
    siteTitle.textContent = THEME_META[theme].title;

    // Start audio (user gesture already happened — theme button click)
    startAudio(theme);

    // Build headline list for crawl
    const allHeadlines = [
      ...(data.world_news || []),
      ...(data.web_security || []),
    ].slice(0, 5);

    CrawlEngine.start({
      headlines: allHeadlines.length > 0
        ? allHeadlines
        : [{ title: 'No headlines available — check your network connection.' }],
      theme,
      onComplete: () => {
        fadeToLoopVolume();
        MatrixRain.stop();
        hide(crawlScreen);
        show(newsScreen);
        renderNews(data);
      },
    });
  }

  // ── Back button ──────────────────────────────────────────────────
  function goToPicker() {
    CrawlEngine.stop();
    MatrixRain.stop();
    stopAudio();
    hide(crawlScreen);
    hide(newsScreen);
    applyThemeClass(null);
    document.getElementById('theme-css').removeAttribute('href');
    if (isLightMode) {
      document.body.classList.remove('light-mode');
      isLightMode = false;
      updateToggleBtn();
    }
    currentTheme = null;
    show(pickerScreen);
  }

  // ── Light/dark toggle ────────────────────────────────────────────
  function toggleLightMode() {
    isLightMode = !isLightMode;
    document.body.classList.toggle('light-mode', isLightMode);
    updateToggleBtn();
  }

  function updateToggleBtn() {
    toggleBtn.textContent = isLightMode ? '☽ Dark' : '☀ Light';
  }

  // ── Utility ──────────────────────────────────────────────────────
  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fmtDate(str) {
    try {
      const d = new Date(str);
      if (isNaN(d)) return str;
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch { return str; }
  }

  // ── Event listeners ──────────────────────────────────────────────
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => selectTheme(btn.dataset.theme));
  });

  backBtn.addEventListener('click', goToPicker);
  toggleBtn.addEventListener('click', toggleLightMode);

})();
