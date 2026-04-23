// Run the same flux algorithm (from fft.js) on the 28 .m4a files
// via ffmpeg decode, so we can validate detector behavior without a browser.
// Writes a JSON report and prints a summary.
import { readdir, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { magSpectrum, hann } from './fft.js';

const SAMPLES_DIR = '/Users/sam/Desktop/samples';
const TARGET_SR = 16000;
const FFT_SIZE = 1024;
const HOP = 256;
const MIN_GAP_SEC = 0.2;

async function findM4a(root) {
  const results = [];
  const entries = await readdir(root, { withFileTypes: true });
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const p = join(root, e.name);
    if (e.isDirectory()) results.push(...await findM4a(p));
    else if (/\.m4a$/i.test(e.name)) results.push(p);
  }
  return results;
}

function decodeToFloat32(path) {
  return new Promise((resolve, reject) => {
    const args = ['-v', 'error', '-i', path, '-ac', '1', '-ar', String(TARGET_SR),
                  '-f', 'f32le', 'pipe:1'];
    const ff = spawn('ffmpeg', args);
    const chunks = [];
    let errOut = '';
    ff.stdout.on('data', c => chunks.push(c));
    ff.stderr.on('data', c => errOut += c.toString());
    ff.on('close', code => {
      if (code !== 0) return reject(new Error(`ffmpeg exit ${code}: ${errOut}`));
      const total = chunks.reduce((a, b) => a + b.length, 0);
      const buf = Buffer.concat(chunks, total);
      resolve(new Float32Array(buf.buffer, buf.byteOffset, total / 4));
    });
  });
}

function computeFlux(sig, fftSize, hop) {
  const win = hann(fftSize);
  const magNorm = 2 / fftSize;
  const frameTimes = [], fluxCurve = [];
  let prevMag = null;
  for (let start = 0; start + fftSize <= sig.length; start += hop) {
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = sig[start + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;
    let f = 0;
    if (prevMag) for (let k = 0; k < mag.length; k++) {
      const d = mag[k] - prevMag[k];
      if (d > 0) f += d;
    }
    fluxCurve.push(f);
    frameTimes.push((start + fftSize / 2) / TARGET_SR);
    prevMag = mag;
  }
  return { fluxCurve, frameTimes };
}

function pickPeaks(flux, times, threshold, minGapSec) {
  const onsets = [];
  let lastTime = -Infinity;
  for (let i = 1; i < flux.length - 1; i++) {
    if (flux[i] < threshold) continue;
    if (flux[i] <= flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    if (times[i] - lastTime < minGapSec) continue;
    onsets.push({ time: +times[i].toFixed(4), flux: +flux[i].toFixed(3) });
    lastTime = times[i];
  }
  return onsets;
}

const paths = await findM4a(SAMPLES_DIR);
console.log(`Found ${paths.length} .m4a files under ${SAMPLES_DIR}`);

const report = [];
let globalMax = 0;
for (const p of paths) {
  try {
    const sig = await decodeToFloat32(p);
    const { fluxCurve, frameTimes } = computeFlux(sig, FFT_SIZE, HOP);
    const maxFlux = fluxCurve.reduce((a, b) => Math.max(a, b), 0);
    if (maxFlux > globalMax) globalMax = maxFlux;
    // Try 3 candidate thresholds
    const out = { file: p.replace(SAMPLES_DIR + '/', ''), duration: sig.length / TARGET_SR,
                  maxFlux: +maxFlux.toFixed(3), at: {} };
    for (const thr of [0.3, 0.5, 0.8, 1.2]) {
      const onsets = pickPeaks(fluxCurve, frameTimes, thr, MIN_GAP_SEC);
      out.at[`thr=${thr}`] = { count: onsets.length, times: onsets.map(o => o.time) };
    }
    report.push(out);
    console.log(`${out.file.padEnd(70)} max=${out.maxFlux.toFixed(3).padStart(6)}  ` +
                Object.entries(out.at).map(([k, v]) => `${k}:${v.count}`).join(' '));
  } catch (e) {
    console.error(`FAIL ${p}: ${e.message}`);
  }
}

await writeFile(join(SAMPLES_DIR, 'frontend', 'smoke_report.json'),
                JSON.stringify({ globalMax, report }, null, 2));
console.log(`\nGlobal max flux across all files: ${globalMax.toFixed(3)}`);
console.log(`Wrote smoke_report.json`);
