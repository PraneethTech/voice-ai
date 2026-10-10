// Mic capture worklet: context runs at 16 kHz, converts to int16 and posts
// exact 512-sample frames (32 ms) to the main thread. The main thread can
// pause capture (echo gate) via a {pause} message.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(512);
    this._n = 0;
    this._paused = false;
    this.port.onmessage = (e) => {
      if (e.data && typeof e.data.pause === 'boolean') this._paused = e.data.pause;
    };
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || this._paused) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this._n === 512) {
        const out = this._buf.slice();
        this.port.postMessage(out.buffer, [out.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCapture);
