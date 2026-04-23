// Evaluate onset detector against ground-truth labels.
//
// Loads data/labels.json, runs flux-based detection on each referenced .m4a,
// matches detections to ground truth, reports per-file accuracy + timing offset,
// and sweeps thresholds to find the best operating point.
//
// Usage: node eval_labels.mjs [variant]
//   variant in: baseline | hf | adaptive | hf+adaptive | recentered | hf+recentered
//                (default: baseline)
//
// "recentered": after flux picks an onset at t_flux, search raw audio in
// [t_flux, t_flux + 120ms] for the peak |amplitude| sample and report THAT as the
// detection time. Aligns detections with perceptual impact (where labels sit).

import { readFile, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { magSpectrum, hann } from './fft.js';

const SAMPLES_DIR = '/Users/sam/Desktop/samples';
const LABELS_PATH = `${SAMPLES_DIR}/data/labels.json`;
const TARGET_SR = 16000;
const FFT_SIZE = 1024;
const HOP = 256;
const MIN_GAP_SEC = 0.2;
const TOLERANCE_SEC = 0.1;     // ±100ms match tolerance
const VARIANT = process.argv[2] || 'baseline';

// Low-frequency cutoff (Hz) for hf variants.
// bin_k = f * FFT_SIZE / SR  →  500 Hz ≈ bin 32
const HF_CUTOFF_HZ = 500;
const HF_CUTOFF_BIN = Math.round(HF_CUTOFF_HZ * FFT_SIZE / TARGET_SR);

// Adaptive threshold params
const ADAPTIVE_WIN_FRAMES = Math.round(0.5 * TARGET_SR / HOP);  // 500ms trailing window
const ADAPTIVE_K = 5;     // threshold = median + k * MAD

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

function computeFlux(sig, opts) {
  const win = hann(FFT_SIZE);
  const magNorm = 2 / FFT_SIZE;
  const times = [];
  const flux = [];
  let prevMag = null;
  const binStart = opts.hfOnly ? HF_CUTOFF_BIN : 0;

  for (let s = 0; s + FFT_SIZE <= sig.length; s += HOP) {
    const frame = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) frame[i] = sig[s + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;
    let f = 0;
    if (prevMag) {
      for (let k = binStart; k < mag.length; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) f += d;
      }
    }
    flux.push(f);
    times.push((s + FFT_SIZE / 2) / TARGET_SR);
    prevMag = mag;
  }
  return { flux, times };
}

// Adaptive threshold at each frame i: median of last WIN frames + K * MAD.
// Used as a per-frame detection threshold; returns a Float32Array same length as flux.
function adaptiveThreshold(flux) {
  const out = new Float32Array(flux.length);
  for (let i = 0; i < flux.length; i++) {
    const lo = Math.max(0, i - ADAPTIVE_WIN_FRAMES);
    const window = flux.slice(lo, i).sort((a, b) => a - b);
    if (window.length < 10) { out[i] = Infinity; continue; }  // not enough history yet
    const med = window[Math.floor(window.length / 2)];
    const absDev = window.map(v => Math.abs(v - med)).sort((a, b) => a - b);
    const mad = absDev[Math.floor(absDev.length / 2)];
    out[i] = med + ADAPTIVE_K * mad;
  }
  return out;
}

function pickPeaks(flux, times, threshold, minGapSec) {
  const onsets = [];
  let lastTime = -Infinity;
  const isArr = Array.isArray(threshold) || threshold instanceof Float32Array;
  for (let i = 1; i < flux.length - 1; i++) {
    const thr = isArr ? threshold[i] : threshold;
    if (flux[i] < thr) continue;
    if (flux[i] <= flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    if (times[i] - lastTime < minGapSec) continue;
    onsets.push({ time: +times[i].toFixed(4), flux: +flux[i].toFixed(3) });
    lastTime = times[i];
  }
  return onsets;
}

function matchOnsets(onsets, groundTruth, toleranceSec) {
  const gtMatched = new Array(groundTruth.length).fill(null);
  const detMatched = new Array(onsets.length).fill(false);
  for (let gi = 0; gi < groundTruth.length; gi++) {
    let bestIdx = -1, bestDt = Infinity;
    for (let di = 0; di < onsets.length; di++) {
      if (detMatched[di]) continue;
      const dt = Math.abs(onsets[di].time - groundTruth[gi]);
      if (dt < bestDt) { bestDt = dt; bestIdx = di; }
    }
    if (bestIdx >= 0 && bestDt <= toleranceSec) {
      gtMatched[gi] = { detIdx: bestIdx, offsetSec: onsets[bestIdx].time - groundTruth[gi] };
      detMatched[bestIdx] = true;
    }
  }
  return { gtMatched, detMatched };
}

const labelsDoc = JSON.parse(await readFile(LABELS_PATH, 'utf8'));
const entries = Object.entries(labelsDoc.labels)
  .filter(([, v]) => v.shotTimes && v.shotTimes.length > 0);

console.log(`Variant: ${VARIANT}  ·  ${entries.length} labeled files  ·  tolerance ±${TOLERANCE_SEC*1000}ms`);
console.log(`Params: FFT=${FFT_SIZE} hop=${HOP} sr=${TARGET_SR}` +
            (VARIANT.includes('hf') ? `  hfCutoff=${HF_CUTOFF_HZ}Hz (bin ${HF_CUTOFF_BIN})` : '') +
            (VARIANT.includes('adaptive') ? `  adaptiveWin=${ADAPTIVE_WIN_FRAMES}fr K=${ADAPTIVE_K}` : ''));
console.log('');

const perFile = [];
for (const [, v] of entries) {
  const absPath = `${SAMPLES_DIR}/${v.path.replace(/^samples\//, '')}`;
  const sig = await decode(absPath);
  const { flux, times } = computeFlux(sig, { hfOnly: VARIANT.includes('hf') });
  perFile.push({ file: absPath.replace(SAMPLES_DIR + '/', ''), gt: v.shotTimes, flux, times, sig,
                 duration: sig.length / TARGET_SR });
}

// Re-center an onset on the peak |amplitude| sample within a forward search window.
function recenter(sig, tSec, searchMs = 120) {
  const startSample = Math.max(0, Math.floor(tSec * TARGET_SR));
  const endSample = Math.min(sig.length, startSample + Math.floor(searchMs * TARGET_SR / 1000));
  let peakIdx = startSample, peakVal = 0;
  for (let i = startSample; i < endSample; i++) {
    const v = Math.abs(sig[i]);
    if (v > peakVal) { peakVal = v; peakIdx = i; }
  }
  return peakIdx / TARGET_SR;
}

function maybeRecenter(onsets, sig) {
  if (!VARIANT.includes('recentered')) return onsets;
  return onsets.map(o => ({ time: +recenter(sig, o.time).toFixed(4), flux: o.flux }));
}

if (VARIANT.includes('adaptive')) {
  // Single detection run per file using adaptive threshold. Report at "threshold = adaptive".
  let tp = 0, fp = 0, fn = 0, offsets = [];
  const rows = [];
  for (const f of perFile) {
    const thr = adaptiveThreshold(f.flux);
    const onsetsRaw = pickPeaks(f.flux, f.times, thr, MIN_GAP_SEC);
    const onsets = maybeRecenter(onsetsRaw, f.sig);
    const m = matchOnsets(onsets, f.gt, TOLERANCE_SEC);
    const fileTP = m.gtMatched.filter(x => x !== null).length;
    const fileFN = m.gtMatched.filter(x => x === null).length;
    const fileFP = m.detMatched.filter(x => !x).length;
    tp += fileTP; fn += fileFN; fp += fileFP;
    for (const mm of m.gtMatched) if (mm) offsets.push(mm.offsetSec);
    rows.push({ file: f.file, gt: f.gt, onsets: onsets.map(o => o.time),
                fileTP, fileFP, fileFN });
  }
  const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);
  const recall = (tp + fn) === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);
  const meanOffMs = offsets.reduce((a, b) => a + b, 0) / offsets.length * 1000;
  const absOffMs = offsets.map(o => Math.abs(o) * 1000).sort((a, b) => a - b);
  const medAbsOffMs = absOffMs[Math.floor(absOffMs.length / 2)];
  const maxAbsOffMs = absOffMs[absOffMs.length - 1];

  console.log(`adaptive threshold  TP=${tp} FP=${fp} FN=${fn}  P=${precision.toFixed(3)}  R=${recall.toFixed(3)}  F1=${f1.toFixed(3)}`);
  console.log(`mean detection offset: ${meanOffMs.toFixed(1)}ms  median |offset|: ${medAbsOffMs.toFixed(1)}ms  max |offset|: ${maxAbsOffMs.toFixed(1)}ms`);
  console.log('\nPer-file (red = miss):');
  for (const r of rows) {
    const bad = r.fileFN > 0 || r.fileFP > 0;
    const mark = bad ? '!' : ' ';
    console.log(`${mark} ${r.file.slice(-55).padEnd(55)} gt=[${r.gt.map(t => t.toFixed(3))}] ` +
                `det=[${r.onsets.map(t => t.toFixed(3)).join(',')}]  TP=${r.fileTP} FP=${r.fileFP} FN=${r.fileFN}`);
  }
  await writeFile(`${SAMPLES_DIR}/frontend/eval_${VARIANT}.json`,
                  JSON.stringify({ tp, fp, fn, precision, recall, f1, meanOffMs, medAbsOffMs, maxAbsOffMs, rows }, null, 2));
  process.exit(0);
}

// Fixed-threshold sweep
const thresholds = [];
for (let t = 0.1; t <= 2.0; t += 0.05) thresholds.push(+t.toFixed(2));

const sweep = thresholds.map(threshold => {
  let tp = 0, fp = 0, fn = 0, offsets = [];
  for (const f of perFile) {
    const onsetsRaw = pickPeaks(f.flux, f.times, threshold, MIN_GAP_SEC);
    const onsets = maybeRecenter(onsetsRaw, f.sig);
    const m = matchOnsets(onsets, f.gt, TOLERANCE_SEC);
    tp += m.gtMatched.filter(x => x !== null).length;
    fn += m.gtMatched.filter(x => x === null).length;
    fp += m.detMatched.filter(x => !x).length;
    for (const mm of m.gtMatched) if (mm) offsets.push(mm.offsetSec);
  }
  const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);
  const recall = (tp + fn) === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);
  const meanOffMs = offsets.length ? offsets.reduce((a, b) => a + b, 0) / offsets.length * 1000 : 0;
  return { threshold, tp, fp, fn, precision, recall, f1, meanOffMs, offsets };
});

const bestF1 = sweep.reduce((a, b) => b.f1 > a.f1 ? b : a);
const perfect = sweep.filter(s => s.recall >= 0.999);
const bestPerfectRecall = perfect.length ? perfect.reduce((a, b) => b.threshold > a.threshold ? b : a) : null;

console.log('threshold  TP  FP  FN   P     R     F1    meanOff(ms)');
for (const s of sweep) {
  if (sweep.indexOf(s) % 2 !== 0 && s !== bestF1 && s !== bestPerfectRecall) continue;
  const mark = s === bestF1 ? ' ← bestF1' : s === bestPerfectRecall ? ' ← highest thr @ R=1' : '';
  console.log(` ${s.threshold.toFixed(2)}     ${String(s.tp).padStart(2)}  ${String(s.fp).padStart(2)}  ${String(s.fn).padStart(2)}   ` +
              `${s.precision.toFixed(2)}  ${s.recall.toFixed(2)}  ${s.f1.toFixed(3)}   ${s.meanOffMs.toFixed(1)}${mark}`);
}

// At best F1 threshold, show per-file + offset stats
console.log(`\nAt threshold ${bestF1.threshold}:`);
console.log(`TP=${bestF1.tp} FP=${bestF1.fp} FN=${bestF1.fn}  P=${bestF1.precision.toFixed(3)}  R=${bestF1.recall.toFixed(3)}  F1=${bestF1.f1.toFixed(3)}`);
const absOff = bestF1.offsets.map(o => Math.abs(o) * 1000).sort((a, b) => a - b);
if (absOff.length) {
  console.log(`mean signed offset: ${bestF1.meanOffMs.toFixed(1)}ms ` +
              `(positive = detection AFTER label; negative = BEFORE)`);
  console.log(`median |offset|: ${absOff[Math.floor(absOff.length / 2)].toFixed(1)}ms ` +
              `max |offset|: ${absOff[absOff.length - 1].toFixed(1)}ms`);
}
console.log('\nPer-file at best F1:');
for (const f of perFile) {
  const onsetsRaw = pickPeaks(f.flux, f.times, bestF1.threshold, MIN_GAP_SEC);
  const onsets = maybeRecenter(onsetsRaw, f.sig);
  const m = matchOnsets(onsets, f.gt, TOLERANCE_SEC);
  const fileTP = m.gtMatched.filter(x => x !== null).length;
  const fileFN = m.gtMatched.filter(x => x === null).length;
  const fileFP = m.detMatched.filter(x => !x).length;
  const bad = fileFN > 0 || fileFP > 0;
  const mark = bad ? '!' : ' ';
  console.log(`${mark} ${f.file.slice(-55).padEnd(55)} gt=[${f.gt.map(t => t.toFixed(3))}] ` +
              `det=[${onsets.map(o => o.time.toFixed(3)).join(',')}]  TP=${fileTP} FP=${fileFP} FN=${fileFN}`);
}

await writeFile(`${SAMPLES_DIR}/frontend/eval_${VARIANT}.json`,
                JSON.stringify({ variant: VARIANT, sweep, bestF1, bestPerfectRecall, toleranceSec: TOLERANCE_SEC }, null, 2));
console.log(`\nWrote eval_${VARIANT}.json`);
