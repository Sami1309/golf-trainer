// Encode a Float32Array mono signal as a 16-bit PCM WAV Blob.
// Samples are assumed to be in [-1, 1]; clipped if out of range.
export function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const byteLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + byteLength);
  const view = new DataView(buffer);

  const writeStr = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + byteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);           // subchunk1 size
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);           // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, byteLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// Resample a Float32Array from one sample rate to another using OfflineAudioContext.
// Returns a Promise<Float32Array>.
export async function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ctx = new OfflineAudioContext(1, Math.ceil(samples.length * toRate / fromRate), toRate);
  const buf = ctx.createBuffer(1, samples.length, fromRate);
  buf.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}
