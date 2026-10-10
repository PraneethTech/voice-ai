'use strict';
console.log('SPAN build v6');
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const root = document.documentElement;
const screen = $('screen');
const chat = $('chat');
const emptyTpl = $('empty').cloneNode(true);
const MAX_ITEMS = 60;

let S = null;            // last state from the server
let connected = false;
let toolName = '';
let recTimer = null;
let wasRecording = false;
let rec = {finals: [], partial: ''};

if (params.has('kiosk')) root.classList.add('kiosk');

// ---------- scale the device/screen uniformly to the viewport ----------
function fit() {
  const d = $('device');
  const s = params.get('scale') === '1' ? 1
    : Math.min(innerWidth / d.offsetWidth, innerHeight / d.offsetHeight) * (params.has('kiosk') ? 1 : 0.94);
  d.style.transform = `scale(${s})`;
}
addEventListener('resize', fit);
fit();

// ---------- helpers ----------
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const two = (n) => String(n).padStart(2, '0');
const clock = () => { const d = new Date(); return `${two(d.getHours())}:${two(d.getMinutes())}`; };
const ms = (v) => v < 1000 ? `${Math.round(v)}ms` : `${(v / 1000).toFixed(1)}s`;
const elapsed = () => {
  const t = Math.max(0, Math.floor(Date.now() / 1000 - (S && S.recording_started_at || Date.now() / 1000)));
  return `${two(Math.floor(t / 60))}:${two(t % 60)}`;
};

function nearBottom() { return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 24; }

function add(node) {
  const stick = nearBottom();
  const empty = $('empty');
  if (empty) empty.remove();
  chat.appendChild(node);
  while (chat.children.length > MAX_ITEMS) chat.firstChild.remove();
  if (stick) chat.scrollTop = chat.scrollHeight;
  return node;
}

// Bubbles still being streamed, by event id. An id is released once its bubble is
// final, so a pipeline that reuses ids (u1, a1…) every turn starts a new bubble.
const open = new Map();
function item(id, cls) {
  let n = open.get(id);
  if (!n) {
    n = add(el('div', cls));
    open.set(id, n);
  }
  return n;
}

// ---------- state rendering ----------
const LABEL = {idle: 'Ready', listening: 'Listening', thinking: 'Thinking', reasoning: 'Reasoning', tool: 'Using tool', speaking: 'Responding'};
const LOADER = {idle: 'Waiting for wake word', listening: 'Listening…', thinking: 'Thinking…', reasoning: 'Reasoning…', speaking: 'Responding…'};

function render() {
  if (!S) return;
  const mode = S.mode;
  const recording = mode === 'recording';
  const s = S.mic_muted ? 'muted' : mode;
  screen.dataset.s = s;
  screen.classList.toggle('rec-on', recording);
  screen.classList.toggle('voice-on', mode === 'idle' && !recording);

  $('stateLabel').textContent = !connected ? 'Reconnecting…'
    : S.mic_muted ? 'Muted'
    : recording ? `● REC ${elapsed()}`
    : LABEL[mode] || mode;

  const loader = !connected ? 'Reconnecting…'
    : S.mic_muted ? 'Mic muted'
    : mode === 'tool' ? `Calling ${toolName || 'tool'}…` : LOADER[mode] || '';
  $('loaderText').textContent = loader;
  const sub = $('emptySub');
  if (sub) sub.textContent = S.mic_muted ? 'Mic muted' : 'Ready';
  $('voiceTitle').textContent = S.mic_muted ? 'Microphone muted' : 'Ready';
  $('voiceSub').textContent = S.mic_muted ? 'Use the device button' : 'Say “Hey Jarvis”';

  root.style.setProperty('--led', S.led.hex);
  const wifi = $('wifi');
  wifi.classList.toggle('off', !S.wifi);
  wifi.setAttribute('aria-label', S.wifi ? 'Wi-Fi connected' : 'Wi-Fi off');
  $('mic').classList.toggle('muted', S.mic_muted);
  $('mic').setAttribute('aria-label', S.mic_muted ? 'Microphone muted' : 'Microphone on');

  $('recClock').textContent = elapsed();
  if (recording && !recTimer) recTimer = setInterval(render, 1000);
  if (!recording && recTimer) { clearInterval(recTimer); recTimer = null; }
  if (recording && !wasRecording) { rec = {finals: [], partial: ''}; renderRec(); }
  if (!recording && wasRecording) archiveRecording();
  wasRecording = recording;
}

// ---------- recording view ----------
function renderRec() {
  const card = $('recCard');
  card.textContent = '';
  if (!rec.finals.length && !rec.partial) {
    card.appendChild(el('span', 'hint', 'Listening for speech…'));
    return;
  }
  if (rec.finals.length) card.appendChild(document.createTextNode(rec.finals.join(' ') + (rec.partial ? ' ' : '')));
  if (rec.partial) card.appendChild(el('span', 'partial', rec.partial));
  card.scrollTop = card.scrollHeight;
}

function archiveRecording() {
  const text = [...rec.finals, rec.partial].filter(Boolean).join(' ');
  if (!text) return;
  const n = el('div', 'msg rec');
  n.append(el('span', 'tag', `● REC · ${clock()}`), document.createTextNode(text));
  add(n);
  rec = {finals: [], partial: ''};
}

// ---------- streamed tokens, batched per frame ----------
const pending = new Map();
let raf = 0;
function flushTokens() {
  raf = 0;
  const stick = nearBottom();
  for (const [id, text] of pending) {
    const n = item(id, 'msg bot live');
    if (!n.firstChild) n.appendChild(document.createTextNode(''));
    n.firstChild.appendData(text);
  }
  pending.clear();
  if (stick) chat.scrollTop = chat.scrollHeight;
}

function metricsLine(m) {
  const n = el('div', 'meta');
  const part = (label, value, unit) => {
    if (label) n.append(label);
    n.append(el('b', null, value));
    if (unit) n.append(unit);
  };
  part('TTFT ', ms(m.ttft_ms), ' · ');
  part('total ', ms(m.total_ms), ' · ');
  part('', String(m.tokens_per_sec), ' tok/s');
  n.append(el('br'));
  part('in ', String(m.input_tokens), ' · ');
  part('out ', String(m.output_tokens), ' · ');
  part('avg ', `${m.tpot_ms}ms`, '/tok');
  n.title = [
    `TTFT (time to first token): ${m.ttft_ms} ms`,
    `Total (end-to-end latency, request → last token): ${m.total_ms} ms`,
    `Generation (first → last token): ${m.gen_ms} ms`,
    `TPOT (avg time per output token): ${m.tpot_ms} ms`,
    `Throughput: ${m.tokens_per_sec} output tokens/s`,
    `Input tokens: ${m.input_tokens} · Output tokens: ${m.output_tokens}`,
  ].join('\n');
  return n;
}

// ---------- thinking/reasoning inline loader ----------
function showThinking(kind) {
  removeThinking();
  const b = el('div', 'msg bot thinking-card');
  b.id = 'thinkingBubble';
  const orb = el('span', 'think-orb');
  const txt = el('span', 'think-text', kind === 'reasoning' ? 'Reasoning…' : 'Thinking…');
  const dots = el('span', 'think-dots');
  dots.appendChild(el('i'));
  dots.appendChild(el('i'));
  dots.appendChild(el('i'));
  b.append(orb, txt, dots);
  add(b);
}

function removeThinking() {
  const b = document.getElementById('thinkingBubble');
  if (b) b.remove();
}

// ---------- events ----------
function onEvent(e) {
  switch (e.type) {
    case 'state':
      S = e.state;
      render();
      return;
    case 'wake':
      removeThinking();
      pending.clear();
      open.clear();
      chat.replaceChildren();
      add(el('div', 'sep', `— Hey Jarvis · ${clock()} —`));
      return;
    case 'user_partial':
    case 'user_final':
      if (e.recording) {
        if (e.type === 'user_final') { rec.finals.push(e.text); rec.partial = ''; } else rec.partial = e.text;
        renderRec();
        return;
      }
      {
        const stick = nearBottom();
        const n = item(e.id, 'msg user');
        n.textContent = e.text;
        n.classList.toggle('partial', e.type === 'user_partial');
        if (e.type === 'user_final') {
          open.delete(e.id);
          showThinking('thinking');
        }
        if (stick) chat.scrollTop = chat.scrollHeight;
      }
      return;
    case 'assistant_start':
      removeThinking();
      item(e.id, 'msg bot live');
      return;
    case 'assistant_token':
      removeThinking();
      pending.set(e.id, (pending.get(e.id) || '') + e.text);
      if (!raf) raf = requestAnimationFrame(flushTokens);
      return;
    case 'assistant_done':
    case 'assistant_interrupted': {
      removeThinking();
      flushTokens();
      const stick = nearBottom();
      const n = item(e.id, 'msg bot');
      n.classList.remove('live');
      open.delete(e.id);
      if (e.type === 'assistant_interrupted') n.appendChild(el('span', 'cut', ' — interrupted'));
      else if (e.metrics) n.after(metricsLine(e.metrics));
      if (stick) chat.scrollTop = chat.scrollHeight;
      return;
    }
    case 'tool': {
      removeThinking();
      const n = item(e.id, 'chip');
      const isStart = e.status === 'start';
      const isOk = e.status === 'ok';
      n.className = `chip ${e.status}`;
      n.textContent = '';
      if (isStart) {
        const spinner = el('span', 'tool-spinner');
        n.appendChild(spinner);
        n.appendChild(document.createTextNode(` Calling ${e.name}…`));
      } else {
        const mark = isOk ? '✓' : '✕';
        const txt = `${mark} ${e.name}${e.detail ? ' · ' + e.detail : ''}`;
        n.textContent = txt;
        open.delete(e.id);
        if (S && (S.mode === 'thinking' || S.mode === 'reasoning')) showThinking(S.mode);
      }
      n.title = n.textContent;
      toolName = e.name;
      if (S && S.mode === 'tool') render();
      return;
    }
    case 'reasoning': {
      showThinking('reasoning');
      return;
    }
    case 'error':
      removeThinking();
      add(el('div', 'err', e.message));
  }
}

async function syncState() {
  // One-shot truth pull: paints the real state on load and rescues the screen
  // wherever the SSE stream is blocked (hotspot middleboxes buffer streams).
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (r.ok) { S = await r.json(); render(); }
  } catch {
    connected = false; render();
  }
}

function connect() {
  syncState();
  const es = new EventSource('/api/events');   // reconnects on its own (server sends retry: 2000)
  es.onopen = () => {
    // The server replays recent history on every (re)connect, so start from a clean log.
    pending.clear();
    open.clear();
    chat.replaceChildren(emptyTpl.cloneNode(true));
    connected = true;
    render();
  };
  es.onerror = () => { connected = false; render(); };
  es.onmessage = (m) => {
    try { onEvent(JSON.parse(m.data)); } catch (err) { console.error(err); }
  };
  // Fallback: while the stream is down, poll the truth every 3s so the
  // screen (wifi/mute/light) can never freeze on stale defaults.
  setInterval(() => { if (!connected) syncState(); }, 3000);
}

// ---------- controls ----------
document.addEventListener('visibilitychange', () => root.classList.toggle('paused', document.hidden));

// Waveform envelope for the recording view: tall in the middle, short at the edges.
document.querySelectorAll('.wave i').forEach((bar, i, all) => {
  const x = (i - (all.length - 1) / 2) / 5;
  bar.style.setProperty('--h', `${Math.round(8 + 46 * Math.exp(-x * x))}px`);
  bar.style.animationDelay = `${-((i * 37) % 70) / 100}s`;
});

connect();
