// AudioWorkletProcessor: maintains a ring buffer of the last N seconds of audio
// so the main thread can extract a window around a detected onset.
//
// Protocol (messages from main thread -> worklet):
//   { cmd: 'extract', requestId, sampleIndex, length }
//     -> replies { requestId, samples: Float32Array, startSample, endSample }
//     -> or { requestId, error: 'out_of_range' } if the window has scrolled away
//
// Messages worklet -> main thread (unsolicited):
//   { type: 'progress', totalSamples }   // sent periodically for main-thread clock sync

class RingBuffer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    const seconds = opts.seconds || 3;
    // sampleRate is a global inside AudioWorkletGlobalScope
    this.capacity = Math.ceil(sampleRate * seconds);
    this.buf = new Float32Array(this.capacity);
    this.writePos = 0;
    this.totalSamples = 0;
    this.progressEvery = Math.ceil(sampleRate / 20); // ~20 Hz
    this.sinceLastProgress = 0;

    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.cmd === 'extract') this.extract(m);
    };
  }

  extract({ requestId, sampleIndex, length }) {
    const endSample = sampleIndex + length;
    const oldestAvailable = this.totalSamples - this.capacity;
    if (sampleIndex < oldestAvailable || endSample > this.totalSamples) {
      this.port.postMessage({ requestId, error: 'out_of_range',
        have: [oldestAvailable, this.totalSamples], want: [sampleIndex, endSample] });
      return;
    }
    const out = new Float32Array(length);
    // Map sampleIndex -> ring position
    const samplesBehindEnd = this.totalSamples - sampleIndex;
    let readPos = (this.writePos - samplesBehindEnd + this.capacity) % this.capacity;
    for (let i = 0; i < length; i++) {
      out[i] = this.buf[readPos];
      readPos++;
      if (readPos === this.capacity) readPos = 0;
    }
    this.port.postMessage({ requestId, samples: out, startSample: sampleIndex, endSample },
                          [out.buffer]);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.writePos] = ch[i];
      this.writePos++;
      if (this.writePos === this.capacity) this.writePos = 0;
    }
    this.totalSamples += ch.length;
    this.sinceLastProgress += ch.length;
    if (this.sinceLastProgress >= this.progressEvery) {
      this.sinceLastProgress = 0;
      this.port.postMessage({ type: 'progress', totalSamples: this.totalSamples });
    }
    return true;
  }
}

registerProcessor('ring-buffer', RingBuffer);
