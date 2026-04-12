/**
 * Voice signal relay integration test.
 *
 * Connects two WebSocket clients, starts a Durak game between them,
 * then verifies the server correctly relays VoiceSignal messages
 * between players in the same room.
 *
 * Run with: node relay.test.js  (from test/ with server running)
 * Or via docker-compose: see docker-compose.test.yml
 */

'use strict';

const WebSocket = require('ws');

const SERVER_URL = process.env.SERVER_URL || 'ws://localhost:3000/ws';
const TIMEOUT_MS = 10_000;

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) {
    console.log(`  PASS: ${msg}`);
    passed++;
  } else {
    console.error(`  FAIL: ${msg}`);
    failed++;
  }
}

function openWs() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(SERVER_URL);
    const timer = setTimeout(() => reject(new Error('WS connect timeout')), TIMEOUT_MS);
    ws.on('open', () => { clearTimeout(timer); resolve(ws); });
    ws.on('error', err => { clearTimeout(timer); reject(err); });
  });
}

function waitForMessage(ws, predicate) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Message wait timeout')), TIMEOUT_MS);
    function handler(data) {
      let msg;
      try { msg = JSON.parse(data.toString()); } catch { return; }
      if (predicate(msg)) {
        clearTimeout(timer);
        ws.off('message', handler);
        resolve(msg);
      }
    }
    ws.on('message', handler);
  });
}

function send(ws, obj) {
  ws.send(JSON.stringify(obj));
}

async function run() {
  console.log('=== Voice Signal Relay Tests ===\n');

  let wsA, wsB;
  try {
    wsA = await openWs();
    wsB = await openWs();
  } catch (e) {
    console.error('Could not connect to server:', e.message);
    console.error('Make sure the server is running at', SERVER_URL);
    process.exit(1);
  }

  // ── Test 1: Join and get Welcome ────────────────────────────────────────────
  console.log('Test 1: Join handshake');
  const joinDone = Promise.all([
    waitForMessage(wsA, m => m.type === 'Welcome'),
    waitForMessage(wsB, m => m.type === 'Welcome'),
  ]);
  send(wsA, { type: 'Join', username: 'VoiceTestA' });
  send(wsB, { type: 'Join', username: 'VoiceTestB' });

  const [welcomeA, welcomeB] = await joinDone;
  assert(typeof welcomeA.player_id === 'string', 'Player A gets UUID');
  assert(typeof welcomeB.player_id === 'string', 'Player B gets UUID');

  const idA = welcomeA.player_id;
  const idB = welcomeB.player_id;

  // ── Test 2: Invite and accept → GameStarted ─────────────────────────────────
  console.log('\nTest 2: Invite/Accept → GameStarted');
  const gameStarted = Promise.all([
    waitForMessage(wsA, m => m.type === 'GameStarted'),
    waitForMessage(wsB, m => m.type === 'GameStarted'),
  ]);
  const incomingInvite = waitForMessage(wsB, m => m.type === 'IncomingInvite');

  send(wsA, { type: 'InvitePlayer', target_id: idB, game: { type: 'Durak' } });

  const invite = await incomingInvite;
  assert(invite.from_id === idA, 'Invite shows correct sender ID');

  send(wsB, { type: 'AcceptInvite', room_id: invite.room_id });

  const [gsA, gsB] = await gameStarted;
  assert(gsA.type === 'GameStarted', 'Player A receives GameStarted');
  assert(gsB.type === 'GameStarted', 'Player B receives GameStarted');

  // ── Test 3: VoiceSignal A→B is relayed ─────────────────────────────────────
  console.log('\nTest 3: VoiceSignal A→B relayed');
  const relayedAtB = waitForMessage(wsB, m => m.type === 'VoiceSignalRelayed');

  send(wsA, {
    type: 'VoiceSignal',
    to: idB,
    signal_type: 'offer',
    payload: JSON.stringify({ type: 'offer', sdp: 'v=0\r\n...' }),
  });

  const relay = await relayedAtB;
  assert(relay.from === idA, 'Relayed signal has correct from ID');
  assert(relay.signal_type === 'offer', 'Signal type preserved');
  assert(relay.payload.includes('"offer"'), 'Payload JSON preserved');

  // ── Test 4: VoiceSignal B→A is relayed ─────────────────────────────────────
  console.log('\nTest 4: VoiceSignal B→A relayed');
  const relayedAtA = waitForMessage(wsA, m => m.type === 'VoiceSignalRelayed');

  send(wsB, {
    type: 'VoiceSignal',
    to: idA,
    signal_type: 'answer',
    payload: JSON.stringify({ type: 'answer', sdp: 'v=0\r\n...' }),
  });

  const relayAns = await relayedAtA;
  assert(relayAns.from === idB, 'Answer relayed with correct from ID');
  assert(relayAns.signal_type === 'answer', 'Answer signal type preserved');

  // ── Test 5: VoiceSignal ICE candidate relayed ───────────────────────────────
  console.log('\nTest 5: ICE candidate relayed');
  const iceAtB = waitForMessage(wsB, m =>
    m.type === 'VoiceSignalRelayed' && m.signal_type === 'ice',
  );
  send(wsA, {
    type: 'VoiceSignal',
    to: idB,
    signal_type: 'ice',
    payload: JSON.stringify({
      candidate: 'candidate:1 1 UDP 2122252543 127.0.0.1 9 typ host',
      sdpMid: '0',
      sdpMLineIndex: 0,
    }),
  });
  const iceRelay = await iceAtB;
  assert(iceRelay.signal_type === 'ice', 'ICE signal type preserved');
  assert(iceRelay.payload.includes('127.0.0.1'), 'ICE candidate payload preserved');

  // ── Test 6: Signal to unknown player ID is silently dropped ─────────────────
  // Sending to a UUID not registered in the server should produce no relay.
  console.log('\nTest 6: Signal to unknown player ID is silently dropped');
  const fakeId = '00000000-0000-0000-0000-000000000000';

  // Listen for any unexpected relay arriving at B (should get nothing)
  const unexpectedAtB = new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 1_500); // 1.5s = nothing arrived
    function handler(data) {
      let m; try { m = JSON.parse(data.toString()); } catch { return; }
      if (m.type === 'VoiceSignalRelayed' && m.from === fakeId) {
        clearTimeout(timer); wsB.off('message', handler); resolve(m);
      }
    }
    wsB.on('message', handler);
  });

  send(wsA, {
    type: 'VoiceSignal',
    to: fakeId,
    signal_type: 'offer',
    payload: '{}',
  });

  const bad = await unexpectedAtB;
  assert(bad === null, 'Signal to unknown player ID is silently dropped');

  // ── Done ───────────────────────────────────────────────────────────────────
  wsA.close();
  wsB.close();

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
  process.exit(failed > 0 ? 1 : 0);
}

run().catch(e => {
  console.error('Unexpected error:', e);
  process.exit(1);
});
