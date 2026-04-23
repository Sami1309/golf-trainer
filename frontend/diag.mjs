// Diagnostic: for each file, report peak amplitude + its time,
// and try several flux configs to find what detects the shot.
import { readdir, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { magSpectrum, hann } from './fft.js';

const SAMPLES_DIR = '/Users/sam/Desktop/samples';
const TARGET_SR = 16000;

async function findM4a(root) {
  const out = [];
  for (const e of await readdir(root, { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;
    const p = join(root, e.name);
    if (e.isDirectory()) out.push(...await findM4a(p));
    else if (/\.m4a$/i.test(e.name)) out.push(p);
  }
  return out;
}

function decode(path) {
  return new Promise((resolve, reject) => {
    const ff = spawn('ffmpeg', ['-v', 'error', '-i', path, '-ac', '1', '-ar', String(TARGET_SR),
                                '-f', 'f32le', 'pipe:1']);
    const chunks = [];
    let err = '';
    ff.stdout.on('data', c => chunks.push(c));
    ff.stderr.on('data', c => err += c.toString());
    ff.on('close', c => {
      if (c !== 0) return reject(new Error(err));
      const buf = Buffer.concat(chunks);
      resolve(new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4));
    });
  });
}

// Flux variants
function flux(sig, fftSize, hop, normalize = 'none') {
  const win = hann(fftSize);
  const magNorm = 2 / fftSize;
  const frames = [];
  const times = [];
  let prevMag = null;
  for (let s = 0; s + fftSize <= sig.length; s += hop) {
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = sig[s + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;
    let f = 0;
    if (prevMag) for (let k = 0; k < mag.length; k++) {
      const d = mag[k] - prevMag[k];
      if (d > 0) f += d;
    }
    if (normalize === 'sqrt') f /= Math.sqrt(mag.length);
    frames.push(f);
    times.push((s + fftSize / 2) / TARGET_SR);
    prevMag = mag;
  }
  return { frames, times };
}

function rmsEnvelope(sig, win, hop) {
  const out = [];
  const times = [];
  for (let s = 0; s + win <= sig.length; s += hop) {
    let sum = 0;
    for (let i = 0; i < win; i++) sum += sig[s + i] * sig[s + i];
    out.push(Math.sqrt(sum / win));
    times.push((s + win / 2) / TARGET_SR);
  }
  return { frames: out, times };
}

const paths = await findM4a(SAMPLES_DIR);
console.log(`files: ${paths.length}`);
const rows = [];

for (const p of paths) {
  const sig = await decode(p);
  // Peak amplitude
  let peakAmp = 0, peakSample = 0;
  for (let i = 0; i < sig.length; i++) {
    const a = Math.abs(sig[i]);
    if (a > peakAmp) { peakAmp = a; peakSample = i; }
  }
  const peakTime = peakSample / TARGET_SR;

  // Flux @ 1024/256 (current)
  const f1024 = flux(sig, 1024, 256);
  // Flux @ 256/64 (short-window, better for transients)
  const f256 = flux(sig, 256, 64);
  // Flux @ 512/128
  const f512 = flux(sig, 512, 128);
  // RMS onset: diff of RMS envelope, ratio variant for gain invariance
  const rms = rmsEnvelope(sig, 256, 64);
  const rmsDiff = new Float32Array(rms.frames.length);
  for (let i = 1; i < rms.frames.length; i++) {
    rmsDiff[i] = Math.max(0, rms.frames[i] - rms.frames[i - 1]);
  }
  const rmsRatio = new Float32Array(rms.frames.length);
  for (let i = 1; i < rms.frames.length; i++) {
    const a = rms.frames[i], b = rms.frames[i - 1] + 1e-6;
    rmsRatio[i] = Math.max(0, Math.log(a / b));
  }

  const argmax = arr => {
    let mi = 0, mv = arr[0];
    for (let i = 1; i < arr.length; i++) if (arr[i] > mv) { mv = arr[i]; mi = i; }
    return [mi, mv];
  };
  const [i1024, v1024] = argmax(f1024.frames);
  const [i256, v256] = argmax(f256.frames);
  const [i512, v512] = argmax(f512.frames);
  const [irms, vrms] = argmax(rmsDiff);
  const [irat, vrat] = argmax(rmsRatio);

  const row = {
    file: p.replace(SAMPLES_DIR + '/', ''),
    dur: +(sig.length / TARGET_SR).toFixed(2),
    peak_amp: +peakAmp.toFixed(3),
    peak_time: +peakTime.toFixed(3),
    flux1024: { max: +v1024.toFixed(3), t: +f1024.times[i1024].toFixed(3) },
    flux512: { max: +v512.toFixed(3), t: +f512.times[i512].toFixed(3) },
    flux256: { max: +v256.toFixed(3), t: +f256.times[i256].toFixed(3) },
    rmsDiff: { max: +vrms.toFixed(3), t: +rms.times[irms].toFixed(3) },
    rmsRatio: { max: +vrat.toFixed(3), t: +rms.times[irat].toFixed(3) },
  };
  rows.push(row);
}

// Print summary
console.log('\nname, dur, peakAmp@t, flux1024max@t, flux512max@t, flux256max@t, rmsDiffMax@t, rmsRatioMax@t');
for (const r of rows) {
  console.log(`${r.file.slice(-55).padEnd(55)} ${String(r.dur).padStart(5)}s ` +
              `peak=${r.peak_amp.toFixed(2)}@${r.peak_time.toFixed(2)} ` +
              `f1024=${r.flux1024.max.toFixed(2)}@${r.flux1024.t.toFixed(2)} ` +
              `f512=${r.flux512.max.toFixed(2)}@${r.flux512.t.toFixed(2)} ` +
              `f256=${r.flux256.max.toFixed(2)}@${r.flux256.t.toFixed(2)} ` +
              `rms=${r.rmsDiff.max.toFixed(3)}@${r.rmsDiff.t.toFixed(2)} ` +
              `rat=${r.rmsRatio.max.toFixed(2)}@${r.rmsRatio.t.toFixed(2)}`);
}

// Agreement check: do flux/rms peak times cluster near each other and near peak_amp?
let agree = 0, disagree = 0;
for (const r of rows) {
  const refs = [r.peak_time, r.flux256.t, r.rmsDiff.t];
  const min = Math.min(...refs), max = Math.max(...refs);
  if (max - min < 0.2) agree++; else disagree++;
}
console.log(`\nConsensus across peak/flux256/rmsDiff within 200ms: ${agree}/${rows.length}`);

await writeFile(join(SAMPLES_DIR, 'frontend', 'diag_report.json'), JSON.stringify(rows, null, 2));
console.log('wrote diag_report.json');
