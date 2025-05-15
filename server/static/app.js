'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let myId = null;
let myName = null;
let ws = null;
let wsRetryDelay = 1000;
let wsRetrying = false;
let peers = {};           // {peerId: RTCPeerConnection}
let peerStreams = {};     // {peerId: MediaStream}
let localStream = null;
let localScreenStream = null;
let micOn = false;
let camOn = false;
let screenOn = false;
let stunServers = [{urls: 'stun:stun.l.google.com:19302'}];
let newMsgCount = 0;
let atBottom = true;
let uploadSeq = 0;

// ── Avatar colors (8 warm palette) ────────────────────────────────────────
const AVATAR_COLORS = [
  '#D97706', '#0D9488', '#DB2777', '#7C3AED',
  '#0284C7', '#65A30D', '#EA580C', '#EC4899',
];
function avatarColor(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}
function avatarInitial(name) {
  return (name || '?').charAt(0).toUpperCase();
}

// ── Relative timestamps ────────────────────────────────────────────────────
function relTime(ts) {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, {weekday: 'short', day: 'numeric', month: 'short'});
}
function absTime(ts) {
  return new Date(ts * 1000).toLocaleString();
}

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const joinScreen    = $('join-screen');
const app           = $('app');
const joinForm      = $('join-form');
const nameInput     = $('name-input');
const codeInput     = $('code-input');
const nameError     = $('name-error');
const codeError     = $('code-error');
const joinBtn       = $('join-btn');
const presenceList  = $('presence-list');
const videoGrid     = $('video-grid');
const soloHint      = $('solo-hint');
const msgContainer  = $('messages');
const chatLoading   = $('chat-loading');
const emptyChat     = $('empty-chat');
const newMsgsBtn    = $('new-msgs-btn');
const msgInput      = $('msg-input');
const sendBtn       = $('send-btn');
const attachBtn     = $('attach-btn');
const fileInput     = $('file-input');
const connBanner    = $('conn-banner');

// Mic/cam buttons (desktop + mobile mirrors)
const btnMic    = $('btn-mic');
const btnCam    = $('btn-cam');
const btnScreen = $('btn-screen');
const btnMicM   = $('btn-mic-m');
const btnCamM   = $('btn-cam-m');
const btnScreenM = $('btn-screen-m');

// ── Join form ──────────────────────────────────────────────────────────────
joinForm.addEventListener('submit', async e => {
  e.preventDefault();
  nameError.className = 'field-error';
  codeError.className = 'field-error';

  const name = nameInput.value.trim();
  const code = codeInput.value.trim();
  if (!name) { showError(nameError, 'Name required'); return; }
  if (!code) { showError(codeError, 'Invite code required'); return; }

  joinBtn.disabled = true;
  joinBtn.textContent = 'Joining…';
  try {
    const res = await fetch('/api/join', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, invite_code: code}),
    });
    const json = await res.json();
    if (!res.ok) {
      if (res.status === 409) showError(nameError, json.error);
      else showError(codeError, json.error || 'Join failed');
      return;
    }
    myName = json.name;
    enterApp();
  } catch {
    showError(codeError, 'Network error — is the server running?');
  } finally {
    joinBtn.disabled = false;
    joinBtn.textContent = 'Join the room';
  }
});

function showError(el, msg) {
  el.textContent = msg;
  el.className = 'field-error visible';
}

// ── Check existing session cookie ──────────────────────────────────────────
async function checkSession() {
  try {
    const res = await fetch('/api/messages');
    if (res.ok) {
      // Cookie is valid — skip join form
      // We'll get our name from the welcome WS message
      enterApp();
      return;
    }
  } catch {}
  // Show join screen
  joinScreen.style.display = 'flex';
}

function enterApp() {
  joinScreen.style.display = 'none';
  app.classList.add('visible');
  loadHistory();
  connectWS();
}

// ── Message history ────────────────────────────────────────────────────────
async function loadHistory() {
  chatLoading.style.display = 'flex';
  chatLoading.style.flexDirection = 'column';
  chatLoading.style.gap = '8px';
  try {
    const res = await fetch('/api/messages');
    if (!res.ok) return;
    const msgs = await res.json();
    chatLoading.remove();
    if (msgs.length === 0) {
      emptyChat.classList.add('visible');
    } else {
      for (const m of msgs) appendChatMsg(m, false);
      scrollToBottom();
    }
  } catch {
    chatLoading.remove();
  }
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.addEventListener('open', () => {
    wsRetryDelay = 1000;
    wsRetrying = false;
    connBanner.classList.remove('visible');
  });

  ws.addEventListener('message', e => handleWS(JSON.parse(e.data)));

  ws.addEventListener('close', () => {
    if (!wsRetrying) scheduleReconnect();
  });
  ws.addEventListener('error', () => {
    ws.close();
  });
}

function scheduleReconnect() {
  wsRetrying = true;
  connBanner.classList.add('visible');
  connBanner.textContent = 'Reconnecting…';
  const delay = wsRetryDelay;
  wsRetryDelay = Math.min(wsRetryDelay * 2, 30000);

  const elapsed = { start: Date.now(), limit: 5 * 60 * 1000 };
  setTimeout(() => {
    if (Date.now() - elapsed.start > elapsed.limit) {
      connBanner.textContent = 'Connection lost — reload page.';
      return;
    }
    // Close stale peer connections
    for (const id of Object.keys(peers)) closePeer(id);
    connectWS();
  }, delay);
}

function handleWS(msg) {
  switch (msg.type) {
    case 'welcome':
      myId = msg.id;
      myName = myName || msg.name;
      stunServers = msg.stun || stunServers;
      addMyTile();
      // As newcomer: offer to all existing peers
      for (const peer of msg.peers) {
        addPeerTile(peer.id, peer.name);
        startOffer(peer.id);
      }
      updateGridCols();
      updatePresence(msg.peers);
      break;

    case 'presence':
      updatePresence(msg.peers);
      break;

    case 'peer_joined':
      if (msg.id !== myId) {
        addPeerTile(msg.id, msg.name);
        updateGridCols();
        appendSysMsg(`${msg.name} joined.`);
      }
      break;

    case 'peer_left':
      removePeerTile(msg.id);
      closePeer(msg.id);
      updateGridCols();
      appendSysMsg(`${msg.name} left.`);
      break;

    case 'offer':
      handleOffer(msg);
      break;
    case 'answer':
      handleAnswer(msg);
      break;
    case 'ice-candidate':
      handleIce(msg);
      break;

    case 'chat':
      appendChatMsg({
        user: msg.user,
        content: msg.content,
        msg_type: msg.msg_type || 'text',
        filename: msg.filename,
        ts: msg.ts,
      }, true);
      break;
  }
}

// ── WebRTC ────────────────────────────────────────────────────────────────
function createPeerConnection(peerId) {
  const pc = new RTCPeerConnection({iceServers: stunServers});
  peers[peerId] = pc;

  if (localStream) {
    localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
  }

  pc.onicecandidate = e => {
    if (e.candidate) {
      ws.send(JSON.stringify({type: 'ice-candidate', target: peerId, candidate: e.candidate}));
    }
  };

  pc.ontrack = e => {
    peerStreams[peerId] = e.streams[0];
    const tile = document.querySelector(`[data-peer="${peerId}"]`);
    if (tile) attachVideo(tile, e.streams[0]);
  };

  pc.oniceconnectionstatechange = () => {
    console.log(`[WebRTC] peer ${peerId} ice: ${pc.iceConnectionState}`);
  };

  return pc;
}

async function startOffer(peerId) {
  const pc = createPeerConnection(peerId);
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws.send(JSON.stringify({type: 'offer', target: peerId, sdp: pc.localDescription}));
}

async function handleOffer(msg) {
  const pc = createPeerConnection(msg.from);
  await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  ws.send(JSON.stringify({type: 'answer', target: msg.from, sdp: pc.localDescription}));
}

async function handleAnswer(msg) {
  const pc = peers[msg.from];
  if (pc) await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));
}

async function handleIce(msg) {
  const pc = peers[msg.from];
  if (pc && msg.candidate) {
    try { await pc.addIceCandidate(new RTCIceCandidate(msg.candidate)); } catch {}
  }
}

function closePeer(peerId) {
  const pc = peers[peerId];
  if (pc) { pc.close(); delete peers[peerId]; }
  delete peerStreams[peerId];
}

// ── Video tiles ───────────────────────────────────────────────────────────
function addMyTile() {
  if (document.querySelector('[data-peer="me"]')) return;
  const tile = makeTile('me', myName, true);
  videoGrid.insertBefore(tile, videoGrid.firstChild);
  updateGridCols();
}

function addPeerTile(peerId, peerName) {
  if (document.querySelector(`[data-peer="${peerId}"]`)) return;
  const tile = makeTile(peerId, peerName, false);
  videoGrid.appendChild(tile);
}

function removePeerTile(peerId) {
  const tile = document.querySelector(`[data-peer="${peerId}"]`);
  if (tile) {
    tile.style.opacity = '0';
    tile.style.transition = 'opacity .2s';
    setTimeout(() => tile.remove(), 200);
  }
}

function makeTile(id, name, isMe) {
  const tile = document.createElement('div');
  tile.className = 'video-tile' + (isMe ? ' you' : '');
  tile.dataset.peer = id;
  tile.setAttribute('aria-label', `${isMe ? 'You' : name} — camera off, mic off`);

  const avatar = document.createElement('div');
  avatar.className = 'tile-avatar';
  avatar.style.background = avatarColor(name);
  avatar.textContent = avatarInitial(name);

  const nameEl = document.createElement('div');
  nameEl.className = 'tile-name';
  nameEl.textContent = isMe ? `You (${name})` : name;

  const status = document.createElement('div');
  status.className = 'tile-status';

  tile.appendChild(avatar);
  tile.appendChild(nameEl);
  tile.appendChild(status);

  return tile;
}

function attachVideo(tile, stream, mirror = false) {
  let video = tile.querySelector('video');
  if (!video) {
    video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;
    video.muted = (tile.dataset.peer === 'me');
    if (mirror) video.style.transform = 'scaleX(-1)';
    tile.insertBefore(video, tile.firstChild);
    const nameEl = tile.querySelector('.tile-name');
    if (nameEl) nameEl.className = 'tile-name on-video';
  }
  video.srcObject = stream;
}

function removeVideo(tile) {
  const video = tile.querySelector('video');
  if (video) {
    video.srcObject = null;
    video.remove();
  }
  const nameEl = tile.querySelector('.tile-name');
  if (nameEl) nameEl.className = 'tile-name';
}

function updateGridCols() {
  const count = videoGrid.children.length;
  let cols = 2;
  if (count <= 1) cols = 1;
  else if (count <= 4) cols = 2;
  else if (count <= 9) cols = 3;
  else cols = 4;
  videoGrid.style.setProperty('--grid-cols', cols);

  const onlyMe = count === 1 && document.querySelector('[data-peer="me"]');
  soloHint.classList.toggle('visible', !!onlyMe);
}

// ── Presence header ───────────────────────────────────────────────────────
function updatePresence(peerList) {
  presenceList.innerHTML = '';
  const all = myName
    ? [{id: 'me', name: myName}, ...peerList.filter(p => p.id !== myId)]
    : peerList;
  for (const p of all) {
    const item = document.createElement('span');
    item.className = 'presence-item';
    item.innerHTML = `<span class="presence-dot"></span>${escHtml(p.name)}`;
    presenceList.appendChild(item);
  }
}

// ── Camera / Mic ──────────────────────────────────────────────────────────
async function toggleMic() {
  if (!localStream) {
    if (!await startLocalMedia()) return;
  }
  micOn = !micOn;
  localStream.getAudioTracks().forEach(t => { t.enabled = micOn; });
  setCtrlActive(btnMic, btnMicM, micOn);
  broadcastMediaState();
}

async function toggleCam() {
  if (!localStream) {
    if (!await startLocalMedia()) return;
  }
  camOn = !camOn;
  localStream.getVideoTracks().forEach(t => { t.enabled = camOn; });
  const myTile = document.querySelector('[data-peer="me"]');
  if (myTile) {
    if (camOn) attachVideo(myTile, localStream, true);
    else removeVideo(myTile);
  }
  setCtrlActive(btnCam, btnCamM, camOn);
  broadcastMediaState();
}

async function toggleScreen() {
  if (screenOn) {
    stopScreen();
    return;
  }
  try {
    localScreenStream = await navigator.mediaDevices.getDisplayMedia({video: true});
    screenOn = true;
    setCtrlActive(btnScreen, btnScreenM, true);
    const track = localScreenStream.getVideoTracks()[0];
    for (const pc of Object.values(peers)) {
      const sender = pc.getSenders().find(s => s.track && s.track.kind === 'video');
      if (sender) sender.replaceTrack(track);
      else pc.addTrack(track, localScreenStream);
    }
    track.onended = stopScreen;
  } catch (err) {
    if (err.name !== 'NotAllowedError') console.warn('[screen]', err);
  }
}

function stopScreen() {
  if (localScreenStream) {
    localScreenStream.getTracks().forEach(t => t.stop());
    localScreenStream = null;
  }
  screenOn = false;
  setCtrlActive(btnScreen, btnScreenM, false);
  // Restore camera track to peers
  const camTrack = localStream && localStream.getVideoTracks()[0];
  if (camTrack) {
    for (const pc of Object.values(peers)) {
      const sender = pc.getSenders().find(s => s.track && s.track.kind === 'video');
      if (sender) sender.replaceTrack(camTrack);
    }
  }
}

async function startLocalMedia() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({audio: true, video: true});
    micOn = true;
    camOn = true;
    // Add tracks to existing peer connections
    for (const pc of Object.values(peers)) {
      localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
    }
    return true;
  } catch (err) {
    const myTile = document.querySelector('[data-peer="me"]');
    if (myTile) {
      const avatar = myTile.querySelector('.tile-avatar');
      if (avatar) {
        avatar.title = 'Camera access denied — check browser settings';
      }
    }
    console.warn('[media]', err.name, err.message);
    return false;
  }
}

function broadcastMediaState() {
  // Notify peers of our media state via a chat-channel-style WS message
  // (peers update our tile's mic/cam icons)
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'media-state',
      mic: micOn,
      cam: camOn,
    }));
  }
}

function setCtrlActive(btn1, btn2, active) {
  [btn1, btn2].forEach(b => {
    if (!b) return;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

btnMic.addEventListener('click', toggleMic);
btnCam.addEventListener('click', toggleCam);
btnScreen.addEventListener('click', toggleScreen);
btnMicM.addEventListener('click', toggleMic);
btnCamM.addEventListener('click', toggleCam);
btnScreenM.addEventListener('click', toggleScreen);

// ── Chat ──────────────────────────────────────────────────────────────────
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + 'px';
});

sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
  const content = msgInput.value.trim();
  if (!content) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({type: 'chat', content}));
  msgInput.value = '';
  msgInput.style.height = 'auto';
}

// ── Append chat messages ──────────────────────────────────────────────────
function appendChatMsg(m, live) {
  emptyChat.classList.remove('visible');

  const isMe = m.user === myName;
  const div = document.createElement('div');
  div.className = `msg ${isMe ? 'mine' : 'theirs'}`;

  const meta = document.createElement('div');
  meta.className = 'meta';
  const tsSpan = document.createElement('span');
  tsSpan.textContent = `${escHtml(m.user)} · ${relTime(m.ts)}`;
  tsSpan.title = absTime(m.ts);
  meta.appendChild(tsSpan);
  div.appendChild(meta);

  if (m.msg_type === 'image') {
    const img = document.createElement('img');
    img.className = 'msg-image';
    img.src = m.content;
    img.alt = m.filename || 'image';
    img.loading = 'lazy';
    img.tabIndex = 0;
    div.appendChild(img);
  } else if (m.msg_type === 'file') {
    const a = document.createElement('a');
    a.className = 'file-chip';
    a.href = m.content;
    a.download = m.filename || 'file';
    a.textContent = `📎 ${m.filename || 'file'}`;
    div.appendChild(a);
  } else {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = m.content;
    div.appendChild(bubble);
  }

  msgContainer.appendChild(div);

  if (live && !atBottom) {
    newMsgCount++;
    newMsgsBtn.textContent = `↓ ${newMsgCount} new message${newMsgCount === 1 ? '' : 's'}`;
    newMsgsBtn.classList.add('visible');
  } else if (atBottom) {
    scrollToBottom();
  }
}

function appendSysMsg(text) {
  const el = document.createElement('div');
  el.className = 'sys-msg';
  el.textContent = text;
  msgContainer.appendChild(el);
  if (atBottom) scrollToBottom();
}

function scrollToBottom() {
  msgContainer.scrollTop = msgContainer.scrollHeight;
}

// ── New messages button ───────────────────────────────────────────────────
msgContainer.addEventListener('scroll', () => {
  const threshold = 80;
  const distFromBottom = msgContainer.scrollHeight - msgContainer.scrollTop - msgContainer.clientHeight;
  atBottom = distFromBottom < threshold;
  if (atBottom) {
    newMsgCount = 0;
    newMsgsBtn.classList.remove('visible');
  }
});

newMsgsBtn.addEventListener('click', () => {
  scrollToBottom();
  newMsgCount = 0;
  newMsgsBtn.classList.remove('visible');
});

// ── File upload ───────────────────────────────────────────────────────────
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = '';
});

async function uploadFile(file) {
  const seq = ++uploadSeq;
  const isImage = file.type.startsWith('image/');

  // Show optimistic chip in chat
  const container = document.createElement('div');
  container.className = 'msg mine';
  container.id = `upload-${seq}`;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${myName} · just now`;
  const chip = document.createElement('div');
  chip.className = 'file-chip uploading';
  chip.textContent = `📎 ${escHtml(file.name)} · Uploading…`;
  const bar = document.createElement('div');
  bar.className = 'upload-progress';
  const barInner = document.createElement('div');
  barInner.className = 'upload-progress-bar';
  barInner.style.width = '0%';
  bar.appendChild(barInner);
  container.appendChild(meta);
  container.appendChild(chip);
  container.appendChild(bar);
  emptyChat.classList.remove('visible');
  msgContainer.appendChild(container);
  scrollToBottom();

  const form = new FormData();
  form.append('file', file);

  try {
    const xhr = new XMLHttpRequest();
    await new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', e => {
        if (e.lengthComputable) {
          barInner.style.width = `${Math.round(e.loaded / e.total * 100)}%`;
        }
      });
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else reject(new Error(xhr.responseText));
      });
      xhr.addEventListener('error', () => reject(new Error('Network error')));
      xhr.open('POST', '/upload');
      xhr.send(form);
    });
    // Server will broadcast the file message; remove optimistic chip
    container.remove();
  } catch (err) {
    bar.remove();
    chip.textContent = `📎 ${escHtml(file.name)}`;
    chip.classList.remove('uploading');
    const errEl = document.createElement('div');
    errEl.className = 'upload-error';
    errEl.innerHTML = `⚠️ Upload failed · <button style="background:none;border:none;color:#D97706;cursor:pointer;padding:0;font-size:12px" onclick="retryUpload(${JSON.stringify(file.name)})">Retry</button>`;
    container.appendChild(errEl);
    console.warn('[upload]', err);
  }
}

// ── Security helpers ──────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ──────────────────────────────────────────────────────────────────
checkSession();
