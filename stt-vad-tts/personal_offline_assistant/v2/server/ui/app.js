'use strict';

// tobi v2 web client — browser mic in, events + TTS audio out.
// The browser's echo cancellation makes voice barge-in safe on this path.

const els = {
  state: document.getElementById('state'),
  stateDot: document.getElementById('state-dot'),
  wakeHint: document.getElementById('wake-hint'),
  transcript: document.getElementById('transcript'),
  reply: document.getElementById('reply'),
  latency: document.getElementById('latency'),
  logDir: document.getElementById('log-dir'),
  meetingBtn: document.getElementById('meeting-btn'),
  muteBtn: document.getElementById('mute-btn'),
  meetingPanel: document.getElementById('meeting-panel'),
  meetingList: document.getElementById('meeting-list'),
  start: document.getElementById('start'),
  startBtn: document.getElementById('start-btn'),
};

const STATE_COLORS = {
  dormant: '#6b7280', listening: '#22c55e', thinking: '#f59e0b', speaking: '#3b82f6',
};
let meetingMode = false;

let ws = null;
let ctx = null, workletNode = null, mediaStream = null;
let sources = [], nextStartTime = 0;

// The mic stays OPEN while TTS plays so the user can interrupt: the browser's
// AEC (echoCancellation: true below) removes our own voice, and the server's
// barge-sustain guard ignores short speaker-echo blips. Flip suppressMic to
// true only to debug raw echo problems — it drops every frame during playback,
// which makes barge-in impossible.
let suppressMic = false;
let playing = false, playEndTimer = null;

function setPlaying(on) {
  playing = on;
  if (playEndTimer) { clearTimeout(playEndTimer); playEndTimer = null; }
  if (!on && ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'playback', value: false }));
  }
}

function setMicPaused(paused) {
  if (workletNode) workletNode.port.postMessage({ pause: paused });
}

function setState(s) {
  els.state.textContent = meetingMode ? `meeting · ${s}` : s;
  const color = meetingMode ? '#a855f7' : (STATE_COLORS[s] || '#6b7280');
  els.stateDot.style.background = color;
  els.wakeHint.style.display = s === 'dormant' ? 'block' : 'none';
}

function addLine(kind, text, cls) {
  const div = document.createElement('div');
  div.className = `line ${cls || ''}`;
  div.textContent = text;
  if (kind) {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = kind;
    div.prepend(tag);
  }
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return div;
}

let partialEl = null;
function showPartial(text) {
  if (!partialEl) partialEl = addLine('you', text, 'partial');
  else partialEl.textContent = text;
}
function showFinal(text) {
  if (partialEl) { partialEl.remove(); partialEl = null; }
  addLine('you', text, 'final');
  els.reply.textContent = '';   // a new answer starts streaming below
  replyEl = null;
}

let replyEl = null;
function showReplyChunk(text) {
  if (!replyEl || !replyEl.isConnected) {
    replyEl = document.createElement('div');
    replyEl.className = 'reply-line';
    els.reply.appendChild(replyEl);
  }
  replyEl.textContent += text;
}

function stopPlayback() {
  sources.forEach((s) => { try { s.stop(); } catch (e) {} });
  sources = [];
  nextStartTime = 0;
  setPlaying(false);
  setMicPaused(false);
}

function queueAudio(pcm16) {
  const n = pcm16.byteLength / 2;
  const f32 = new Float32Array(n);
  const i16 = new Int16Array(pcm16);
  for (let i = 0; i < n; i++) f32[i] = i16[i] / 32768;
  const buf = ctx.createBuffer(1, n, 16000);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const at = Math.max(ctx.currentTime + 0.02, nextStartTime);
  src.start(at);
  nextStartTime = at + buf.duration;
  sources.push(src);
  src.onended = () => {
    sources = sources.filter((s) => s !== src);
    if (!sources.length) {
      // playback finished; 300ms tail before we report playback-end to the
      // server (it keeps its playback-end echo window open that long)
      if (playEndTimer) clearTimeout(playEndTimer);
      playEndTimer = setTimeout(() => {
        setPlaying(false);
        setMicPaused(false);
      }, 300);
    }
  };
  if (!playing) {
    setPlaying(true);
    setMicPaused(suppressMic);
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'playback', value: true }));
  }
}

async function start() {
  els.startBtn.disabled = true;
  els.startBtn.textContent = 'loading models…';
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    els.startBtn.textContent = 'microphone blocked — allow it, then reload';
    els.start.querySelector('.fine').textContent =
      'Error: ' + (err && err.name ? err.name : 'mic unavailable');
    return;
  }
  ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule('/mic-worklet.js');
  workletNode = new AudioWorkletNode(ctx, 'mic-capture');
  workletNode.port.onmessage = (e) => {
    if (suppressMic && playing) return; // inert fallback echo gate (off by default)
    if (ws && ws.readyState === 1) ws.send(e.data);
  };
  ctx.createMediaStreamSource(mediaStream).connect(workletNode); // not to destination: no feedback loop

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return queueAudio(e.data);
    const m = JSON.parse(e.data);
    switch (m.type) {
      case 'hello':
        els.logDir.textContent = `logs: ${m.log_dir.split(/[\\/]/).slice(-1)[0]}`;
        if (m.wake_words) els.wakeHint.textContent = `say "hey ${m.wake_words[0].replaceAll('_', ' ')}"`;
        break;
      case 'state': setState(m.state); break;
      case 'wake': addLine('sys', 'wake word detected', 'sys'); els.reply.textContent = ''; replyEl = null; break;
      case 'partial': showPartial(m.text); break;
      case 'final': showFinal(m.text); break;
      case 'reply_chunk': showReplyChunk(m.text); break;
      case 'bargein': stopPlayback(); els.reply.textContent = ''; replyEl = null; break;
      case 'error': addLine('sys', `error: ${m.message}`, 'sys'); break;
      case 'latency': {
        const parts = [`${(m.eot_to_first_audio_ms / 1000).toFixed(2)}s voice`];
        if (m.llm_ttft_ms != null) parts.push(`llm ${(m.llm_ttft_ms / 1000).toFixed(2)}s`);
        if (m.stt_ms != null) parts.push(`stt ${(m.stt_ms / 1000).toFixed(2)}s`);
        if (m.turn_ms != null) parts.push(`turn ${m.turn_ms}ms`);
        els.latency.textContent = parts.join(' · ');
        break;
      }
      case 'meeting_segment': {
        const div = document.createElement('div');
        div.className = 'mseg';
        div.dataset.tms = m.t_ms;
        div.textContent = `+${Math.floor(m.t_ms / 60000)}:${String(Math.floor(m.t_ms / 1000) % 60).padStart(2, '0')}  ${m.text}`;
        els.meetingList.appendChild(div);
        els.meetingList.scrollTop = els.meetingList.scrollHeight;
        break;
      }
      case 'meeting_report': {
        els.meetingList.innerHTML = '';
        for (const s of m.segments) {
          const div = document.createElement('div');
          div.className = 'mseg';
          div.innerHTML = `<b>${s.speaker}</b> +${Math.floor(s.t_ms / 60000)}:${String(Math.floor(s.t_ms / 1000) % 60).padStart(2, '0')}  ${s.text}`;
          els.meetingList.appendChild(div);
        }
        const sum = document.createElement('div');
        sum.className = 'msummary';
        sum.textContent = m.summary;
        els.meetingList.appendChild(sum);
        els.meetingList.scrollTop = els.meetingList.scrollHeight;
        break;
      }
      case 'muted': els.muteBtn.textContent = m.muted ? 'unmute' : 'mute'; break;
    }
  };
  ws.onclose = () => setState('disconnected');

  els.start.style.display = 'none';
}

els.startBtn.addEventListener('click', start);

els.muteBtn.addEventListener('click', () => {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'mute', value: els.muteBtn.textContent === 'mute' }));
});

els.meetingBtn.addEventListener('click', () => {
  if (!ws || ws.readyState !== 1) return;
  const on = els.meetingBtn.textContent !== 'end meeting';
  ws.send(JSON.stringify({ type: 'meeting', value: on }));
  els.meetingBtn.textContent = on ? 'end meeting' : 'meeting mode';
  meetingMode = on;
  els.meetingPanel.style.display = on ? 'block' : 'none';
  els.reply.style.display = on ? 'none' : 'block';
  if (on) { stopPlayback(); els.meetingList.innerHTML = ''; setState('listening'); }
});

// Keep the state pill in sync during playback silence.
setInterval(() => {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
}, 25000);
