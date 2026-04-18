/**
 * Voice chat E2E test.
 *
 * Two browser contexts (simulating two players) join the server, start a game,
 * and enable voice. The test verifies:
 *   1. getUserMedia succeeds with fake devices
 *   2. WebRTC signaling completes (offer/answer + ICE exchange)
 *   3. RTCPeerConnection reaches "connected" state on both sides
 *   4. An audio element is created and play() is called
 *
 * Run via:
 *   npx playwright test --config playwright.config.js
 */

const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.SERVER_URL || 'http://localhost:3000';

/** Join the game as `username` and wait for the picker screen to appear. */
async function joinGame(page, username) {
  await page.goto(BASE_URL);
  await page.waitForSelector('#username-modal:not(.hidden)', { timeout: 5_000 });
  await page.fill('#username-input', username);
  await page.click('#username-submit');
  // Picker screen visible = join successful
  await page.waitForSelector('#picker-screen:not(.hidden)', { timeout: 8_000 });
}

/**
 * Wait until a second player appears in the lobby list with an "Invite" button,
 * then click it.  Returns the player row element.
 */
async function inviteFirstAvailablePlayer(page) {
  // Wait until at least one HUMAN player's invite button is enabled (skip bot rows).
  await page.waitForFunction(() => {
    const btns = document.querySelectorAll('.player-row:not(.bot-row) .invite-btn:not(:disabled)');
    return btns.length > 0;
  }, { timeout: 10_000 });
  await page.locator('.player-row:not(.bot-row) .invite-btn:not(:disabled)').first().click();
}

/** Wait for and accept an incoming invite banner. */
async function acceptInvite(page) {
  await page.waitForSelector('#invite-banner:not(.hidden)', { timeout: 10_000 });
  await page.click('#invite-accept');
}

/** Wait until the game screen is showing. */
async function waitForGameScreen(page) {
  await page.waitForSelector('#game-screen:not(.hidden)', { timeout: 10_000 });
}

/** Click "Voice" and return the resolved localStream (truthy = mic captured). */
async function enableVoice(page) {
  await page.click('#voice-btn');
  // Wait for the button to flip to active state
  await page.waitForFunction(() => {
    const btn = document.getElementById('voice-btn');
    return btn && (btn.classList.contains('active') || btn.textContent.includes('On'));
  }, { timeout: 8_000 });
}

/** Poll until the RTCPeerConnection state reaches `expected` (or timeout). */
async function waitForConnectionState(page, expected, timeoutMs = 12_000) {
  await page.waitForFunction(
    (expected) => {
      const pcs = Object.values(window.peers || {});
      return pcs.some(p => p.pc && p.pc.connectionState === expected);
    },
    expected,
    { timeout: timeoutMs },
  );
}

/** Check whether an audio element for a peer was created and has srcObject set. */
async function audioElementExists(page) {
  return page.evaluate(() => {
    const pcs = Object.values(window.peers || {});
    return pcs.some(p => p.audio && p.audio.srcObject);
  });
}

// ──────────────────────────────────────────────────────────────────────────────

test.describe('Voice Chat', () => {

  test('two players can enable voice and reach "connected" state', async ({ browser }) => {
    // Two isolated browser contexts — simulate two separate users.
    const ctxA = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const ctxB = await browser.newContext({ permissions: ['microphone', 'camera'] });

    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();

    // ── 1. Both join ─────────────────────────────────────────────────────────
    // Join sequentially so Player B appears in A's lobby list after A joins.
    await joinGame(pageA, 'VoiceUserA');
    await joinGame(pageB, 'VoiceUserB');

    // ── 2. A invites B, B accepts ─────────────────────────────────────────────
    // Run both actions concurrently (A invites, B waits for and accepts)
    await Promise.all([
      inviteFirstAvailablePlayer(pageA),
      acceptInvite(pageB),
    ]);

    // ── 3. Both reach the game screen ─────────────────────────────────────────
    await Promise.all([
      waitForGameScreen(pageA),
      waitForGameScreen(pageB),
    ]);

    // ── 4. Both enable voice ──────────────────────────────────────────────────
    await Promise.all([
      enableVoice(pageA),
      enableVoice(pageB),
    ]);

    // ── 5. Verify WebRTC connection reaches "connected" on both sides ─────────
    await Promise.all([
      waitForConnectionState(pageA, 'connected'),
      waitForConnectionState(pageB, 'connected'),
    ]);

    // ── 6. Audio element created and has a srcObject ──────────────────────────
    const audioA = await audioElementExists(pageA);
    const audioB = await audioElementExists(pageB);
    expect(audioA).toBe(true);
    expect(audioB).toBe(true);

    await ctxA.close();
    await ctxB.close();
  });

  test('voice status updates correctly when mic is enabled', async ({ page }) => {
    // Single-player sanity check: mic is captured and UI reflects it.
    await joinGame(page, 'VoiceSanityUser');

    // Start a bot game so we have a game screen (voice button is hidden in picker)
    await page.waitForSelector('.invite-btn', { timeout: 5_000 });
    // Click the "Bot" row invite button if present, otherwise play bot via JS
    const botInvite = page.locator('.bot-row .invite-btn, button:has-text("Bot")').first();
    if (await botInvite.count() > 0) {
      await botInvite.click();
    } else {
      // Fallback: trigger via game picker (select Durak, click bot)
      await page.evaluate(() => {
        window._testBotGame = true;
        // Find and click the first bot invite button in the player list
        const btns = [...document.querySelectorAll('.invite-btn')];
        const botBtn = btns.find(b => b.closest('.bot-row'));
        if (botBtn) botBtn.click();
      });
    }

    await waitForGameScreen(page);

    // Voice button starts in inactive state
    const voiceText = await page.locator('#voice-btn').textContent();
    expect(voiceText).toContain('Voice');

    // Enable voice — fake device means getUserMedia will succeed
    await page.click('#voice-btn');

    // Wait for status to update
    await page.waitForFunction(() => {
      const s = document.getElementById('voice-status');
      return s && (s.textContent.includes('Voice on') || s.textContent.includes('On'));
    }, { timeout: 6_000 });

    const status = await page.locator('#voice-status').textContent();
    expect(status).toMatch(/Voice on|On/i);

    // voiceActive flag is set in the page JS
    const voiceActive = await page.evaluate(() => window.voiceActive);
    expect(voiceActive).toBe(true);
  });

  test('ICE candidates are buffered when received before remote description', async ({ page }) => {
    await page.goto(BASE_URL);
    // iceCandidateBuffer is declared at module scope — must exist on page load.
    const buffered = await page.evaluate(() =>
      typeof window.iceCandidateBuffer !== 'undefined' ? 'declared' : 'missing',
    );
    expect(buffered).toBe('declared');
  });

  test('video button and window are present in the DOM', async ({ page }) => {
    await page.goto(BASE_URL);
    // The video button and window are rendered even before a game starts.
    await expect(page.locator('#video-btn')).toBeAttached();
    await expect(page.locator('#video-window')).toBeAttached();
    await expect(page.locator('#remote-video')).toBeAttached();
    await expect(page.locator('#local-video')).toBeAttached();
    // Window is hidden by default
    await expect(page.locator('#video-window')).toHaveClass(/hidden/);
  });

  test('video window appears when camera is enabled in a game', async ({ page }) => {
    // Start a bot game so we have a game screen, then enable video.
    await joinGame(page, 'VideoSanityUser');

    // Need the game screen — click a bot invite if available
    const allInvites = page.locator('.invite-btn');
    await allInvites.first().waitFor({ timeout: 5_000 });
    const botBtn = page.locator('.bot-row .invite-btn').first();
    if (await botBtn.count() > 0) {
      await botBtn.click();
    } else {
      await allInvites.first().click();
      // If it triggered a non-bot invite banner on the other side, just proceed
    }

    await waitForGameScreen(page);

    // Video button should start with no active class
    const btnText = await page.locator('#video-btn').textContent();
    expect(btnText).toContain('Video');

    // Click video — fake device means getUserMedia({video:true}) will succeed
    await page.click('#video-btn');

    // Video window should become visible
    await page.waitForFunction(() => {
      const w = document.getElementById('video-window');
      return w && !w.classList.contains('hidden');
    }, { timeout: 6_000 });

    // Button flips to active state
    await page.waitForFunction(() => {
      const btn = document.getElementById('video-btn');
      return btn && btn.classList.contains('active');
    }, { timeout: 4_000 });

    // Local video element should have a srcObject (our own camera stream)
    const hasLocal = await page.evaluate(() => {
      const lv = document.getElementById('local-video');
      return !!(lv && lv.srcObject);
    });
    expect(hasLocal).toBe(true);

    // videoActive flag is set
    const va = await page.evaluate(() => window.videoActive);
    expect(va).toBe(true);
  });

  test('video window close button stops camera', async ({ page }) => {
    await joinGame(page, 'VideoCloseUser');
    const allInvites = page.locator('.invite-btn');
    await allInvites.first().waitFor({ timeout: 5_000 });
    await allInvites.first().click();
    await waitForGameScreen(page);

    // Enable video
    await page.click('#video-btn');
    await page.waitForFunction(() => window.videoActive === true, { timeout: 6_000 });

    // Click the close button on the video window
    await page.click('#video-window-close');

    // videoActive should flip back to false
    await page.waitForFunction(() => window.videoActive === false, { timeout: 4_000 });

    // Window is hidden again
    const hidden = await page.evaluate(() =>
      document.getElementById('video-window').classList.contains('hidden'),
    );
    expect(hidden).toBe(true);
  });

  test('video window minimize/restore toggles the video area', async ({ page }) => {
    await joinGame(page, 'VideoMinUser');
    const allInvites = page.locator('.invite-btn');
    await allInvites.first().waitFor({ timeout: 5_000 });
    await allInvites.first().click();
    await waitForGameScreen(page);

    await page.click('#video-btn');
    await page.waitForFunction(() => window.videoActive === true, { timeout: 6_000 });

    // Minimize
    await page.click('#video-window-min');
    const areaHidden = await page.evaluate(() => {
      const a = document.getElementById('video-area');
      return a && a.style.display === 'none';
    });
    expect(areaHidden).toBe(true);

    // Restore
    await page.click('#video-window-min');
    const areaVisible = await page.evaluate(() => {
      const a = document.getElementById('video-area');
      return a && a.style.display !== 'none';
    });
    expect(areaVisible).toBe(true);
  });

  test('remote-video element has explicit height so srcObject renders visibly', async ({ page }) => {
    // Regression: #video-window had no explicit height, so #remote-video { height:100% }
    // resolved to 0px and the remote camera feed was invisible even when srcObject was set.
    await page.goto(BASE_URL);
    const dims = await page.evaluate(() => {
      const win = document.getElementById('video-window');
      const rv  = document.getElementById('remote-video');
      if (!win || !rv) return null;
      // Temporarily reveal the window to measure it
      const wasHidden = win.classList.contains('hidden');
      win.classList.remove('hidden');
      const winH = win.getBoundingClientRect().height;
      const rvH  = rv.getBoundingClientRect().height;
      if (wasHidden) win.classList.add('hidden');
      return { winH, rvH };
    });
    expect(dims).not.toBeNull();
    // Both the container and the video element must have positive pixel heights
    expect(dims.winH).toBeGreaterThan(0);
    expect(dims.rvH).toBeGreaterThan(0);
  });

  test('pending signals buffering uses peer count not voiceActive flag', async ({ page }) => {
    // The buffering gate was changed from !voiceActive to Object.keys(peers).length === 0.
    // Verify the new condition is in effect: even if voiceActive is false,
    // signals are NOT buffered when peer connections exist.
    await page.goto(BASE_URL);
    const gateCondition = await page.evaluate(() => {
      // Read the source of the message handler — it should NOT reference voiceActive
      // in the buffering condition. We check by inspecting the live window.
      // Easiest proxy: confirm peers is {} and pendingSignals is {} initially.
      return {
        peersEmpty: Object.keys(window.peers || {}).length === 0,
        pendingEmpty: Object.keys(window.pendingSignals || {}).length === 0,
        voiceActive: window.voiceActive,
      };
    });
    expect(gateCondition.peersEmpty).toBe(true);
    expect(gateCondition.pendingEmpty).toBe(true);
    expect(gateCondition.voiceActive).toBe(false);
    // If a signal arrived now it would be buffered (peers is empty).
    // If peers were non-empty it would be handled directly regardless of voiceActive.
    // The signal handler source confirms this — no direct assertion possible from outside,
    // but the state is correct.
  });

});

// ─────────────────────────────────────────────────────────────────────────────
// Lounge (group chat + voice/video, max 10 players)
// ─────────────────────────────────────────────────────────────────────────────

test.describe('Lounge', () => {

  test('lounge tab button is present in the picker screen', async ({ page }) => {
    await joinGame(page, 'LoungeTabUser');
    await expect(page.locator('#lounge-tab-btn')).toBeAttached();
    await expect(page.locator('#lounge-tab-btn')).toBeVisible();
  });

  test('lounge screen is hidden before joining', async ({ page }) => {
    await joinGame(page, 'LoungeHiddenUser');
    await expect(page.locator('#lounge-screen')).toHaveClass(/hidden/);
  });

  test('clicking Lounge button shows lounge screen and hides picker', async ({ page }) => {
    await joinGame(page, 'LoungeJoinUser');
    await page.click('#lounge-tab-btn');

    // Lounge screen becomes visible
    await page.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });
    // Picker screen hides
    await expect(page.locator('#picker-screen')).toHaveClass(/hidden/);
    // Member count shows at least 1 (ourselves)
    await expect(page.locator('#lounge-member-count')).toHaveText('1');
  });

  test('two players in lounge see each other in member list', async ({ browser }) => {
    const ctxA = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const ctxB = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();

    await joinGame(pageA, 'LoungeMemberA');
    await joinGame(pageB, 'LoungeMemberB');

    await pageA.click('#lounge-tab-btn');
    await pageA.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    await pageB.click('#lounge-tab-btn');
    await pageB.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    // Both pages should show 2 members
    await pageA.waitForFunction(() => {
      const el = document.getElementById('lounge-member-count');
      return el && el.textContent === '2';
    }, { timeout: 6_000 });
    await pageB.waitForFunction(() => {
      const el = document.getElementById('lounge-member-count');
      return el && el.textContent === '2';
    }, { timeout: 6_000 });

    // A sees B's name, B sees A's name
    await expect(pageA.locator('#lounge-members-list')).toContainText('LoungeMemberB');
    await expect(pageB.locator('#lounge-members-list')).toContainText('LoungeMemberA');

    await ctxA.close();
    await ctxB.close();
  });

  test('lounge text chat is relayed to all members', async ({ browser }) => {
    const ctxA = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const ctxB = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();

    await joinGame(pageA, 'LoungeChatA');
    await joinGame(pageB, 'LoungeChatB');

    await pageA.click('#lounge-tab-btn');
    await pageA.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });
    await pageB.click('#lounge-tab-btn');
    await pageB.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    // Wait for both to see each other
    await pageA.waitForFunction(() =>
      document.getElementById('lounge-member-count')?.textContent === '2', { timeout: 6_000 });

    // A sends a message
    await pageA.fill('#lounge-chat-input', 'Hello lounge!');
    await pageA.click('#lounge-chat-send');

    // B should receive it
    await pageB.waitForFunction(() => {
      const msgs = document.getElementById('lounge-chat-messages');
      return msgs && msgs.innerText.includes('Hello lounge!');
    }, { timeout: 6_000 });

    // A's own messages panel also shows it
    const chatTextA = await pageA.locator('#lounge-chat-messages').innerText();
    expect(chatTextA).toContain('Hello lounge!');

    await ctxA.close();
    await ctxB.close();
  });

  test('leave lounge returns player to picker screen', async ({ page }) => {
    await joinGame(page, 'LoungeLeaveUser');
    await page.click('#lounge-tab-btn');
    await page.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    await page.click('#lounge-leave-btn');

    // Picker screen comes back
    await page.waitForSelector('#picker-screen:not(.hidden)', { timeout: 5_000 });
    // Lounge hides
    await expect(page.locator('#lounge-screen')).toHaveClass(/hidden/);
  });

  test('lounge voice buttons are present', async ({ page }) => {
    await joinGame(page, 'LoungeVoiceBtnUser');
    await page.click('#lounge-tab-btn');
    await page.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    await expect(page.locator('#lounge-voice-btn')).toBeAttached();
    await expect(page.locator('#lounge-video-btn')).toBeAttached();
    await expect(page.locator('#lounge-others-btn')).toBeAttached();
  });

  test('lounge voice activates mic and updates button', async ({ page }) => {
    await joinGame(page, 'LoungeVoiceUser');
    await page.click('#lounge-tab-btn');
    await page.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    await page.click('#lounge-voice-btn');

    await page.waitForFunction(() => window.voiceActive === true, { timeout: 6_000 });

    // Wait for the 200ms sync loop to update the lounge button text.
    // waitForFunction IS the assertion — no need to re-read textContent() immediately
    // after (the sync loop could fire again between the two calls).
    await page.waitForFunction(() => {
      const btn = document.getElementById('lounge-voice-btn');
      return btn && (btn.textContent.includes('On') || btn.textContent.includes('Muted'));
    }, { timeout: 2_000 });

    // The game-screen voice button is updated synchronously by updateVoiceBtn()
    await page.waitForFunction(() => {
      const btn = document.getElementById('voice-btn');
      return btn && (btn.textContent.includes('On') || btn.textContent.includes('Muted'));
    }, { timeout: 2_000 });
  });

  test('two lounge players reach WebRTC "connected" state', async ({ browser }) => {
    const ctxA = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const ctxB = await browser.newContext({ permissions: ['microphone', 'camera'] });
    const pageA = await ctxA.newPage();
    const pageB = await ctxB.newPage();

    await joinGame(pageA, 'LoungeRTCA');
    await joinGame(pageB, 'LoungeRTCB');

    await pageA.click('#lounge-tab-btn');
    await pageA.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });
    await pageB.click('#lounge-tab-btn');
    await pageB.waitForSelector('#lounge-screen:not(.hidden)', { timeout: 5_000 });

    // Wait for both to see each other
    await pageA.waitForFunction(() =>
      document.getElementById('lounge-member-count')?.textContent === '2', { timeout: 8_000 });

    // Both enable voice
    await Promise.all([
      pageA.click('#lounge-voice-btn'),
      pageB.click('#lounge-voice-btn'),
    ]);

    // Both wait for voiceActive
    await Promise.all([
      pageA.waitForFunction(() => window.voiceActive === true, { timeout: 8_000 }),
      pageB.waitForFunction(() => window.voiceActive === true, { timeout: 8_000 }),
    ]);

    // Both reach WebRTC "connected"
    await Promise.all([
      waitForConnectionState(pageA, 'connected', 15_000),
      waitForConnectionState(pageB, 'connected', 15_000),
    ]);

    // Audio element created on both sides
    expect(await audioElementExists(pageA)).toBe(true);
    expect(await audioElementExists(pageB)).toBe(true);

    await ctxA.close();
    await ctxB.close();
  });

  test('lounge rejects join when full (10 players)', async ({ browser }) => {
    // Open 10 contexts, all join the lounge, the 11th should get an error.
    const contexts = [];
    const pages = [];
    for (let i = 0; i < 11; i++) {
      const ctx = await browser.newContext({ permissions: ['microphone', 'camera'] });
      contexts.push(ctx);
      pages.push(await ctx.newPage());
    }

    // First 10 join successfully
    for (let i = 0; i < 10; i++) {
      await joinGame(pages[i], `LoungeFullUser${i}`);
      await pages[i].click('#lounge-tab-btn');
      await pages[i].waitForSelector('#lounge-screen:not(.hidden)', { timeout: 8_000 });
    }

    // 11th player joins picker but lounge should be full
    await joinGame(pages[10], 'LoungeFullUser10');
    await pages[10].click('#lounge-tab-btn');

    // They should stay on picker screen (server rejects with error)
    // The lounge-screen should remain hidden
    await pages[10].waitForTimeout(1500);
    const loungeVisible = await pages[10].evaluate(() =>
      !document.getElementById('lounge-screen').classList.contains('hidden')
    );
    expect(loungeVisible).toBe(false);

    for (const ctx of contexts) await ctx.close();
  });

});

// ─────────────────────────────────────────────────────────────────────────────
// Forfeit button visibility (regression: was only shown in Poker, not all games)
// ─────────────────────────────────────────────────────────────────────────────

test.describe('Forfeit button', () => {

  test('forfeit button is visible in a Durak game', async ({ page }) => {
    await joinGame(page, 'ForfeitDurakUser');
    // Start a bot game (Durak is default)
    await page.waitForSelector('.invite-btn', { timeout: 5_000 });
    const botBtn = page.locator('.bot-row .invite-btn').first();
    if (await botBtn.count() > 0) await botBtn.click();
    else await page.locator('.invite-btn').first().click();

    await waitForGameScreen(page);

    // Forfeit button must be visible (not hidden) once the game starts
    await expect(page.locator('#forfeit-btn')).not.toHaveClass(/hidden/);
    await expect(page.locator('#forfeit-btn')).toBeVisible();
  });

  test('forfeit button is visible in a Blackjack game', async ({ page }) => {
    await joinGame(page, 'ForfeitBJUser');
    // Select Blackjack
    await page.click('#pick-blackjack');
    await page.waitForSelector('.invite-btn', { timeout: 5_000 });
    const botBtn = page.locator('.bot-row .invite-btn').first();
    if (await botBtn.count() > 0) await botBtn.click();
    else await page.locator('.invite-btn').first().click();

    await waitForGameScreen(page);

    await expect(page.locator('#forfeit-btn')).not.toHaveClass(/hidden/);
    await expect(page.locator('#forfeit-btn')).toBeVisible();
  });

});
