import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MODEL_CLIP_SAMPLES,
  MODEL_SAMPLE_RATE,
  STAGE1_FEATURE_NAMES,
  extractStage1Features,
  prepareModelClip,
} from '../frontend/audio_features.js';
import { magSpectrum, hann } from '../frontend/fft.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LABELS_PATH = join(ROOT, 'data', 'labels.json');
const EXTERNAL_MANIFEST_PATH = join(ROOT, 'data', 'external', 'manifest.jsonl');
const PREPARED_DIR = join(ROOT, 'data', 'stage1b_prepared');
const PREPARED_MANIFEST_PATH = join(PREPARED_DIR, 'manifest.jsonl');
const HANDCRAFTED_MODEL_PATH = join(ROOT, 'frontend', 'models', 'stage1b_handcrafted.json');
const HANDCRAFTED_REPORT_PATH = join(ROOT, 'data', 'stage1b_handcrafted_report.json');
const CLIP_PRE_SAMPLES = Math.round(0.1 * MODEL_SAMPLE_RATE);
const CLIP_POST_SAMPLES = MODEL_CLIP_SAMPLES - CLIP_PRE_SAMPLES;
const NEGATIVE_MIN_PEAK_DBFS = -35;
const MAX_EXTERNAL_NEGATIVES = Infinity;
const EXTERNAL_NEGATIVE_FFT_SIZE = 1024;
const EXTERNAL_NEGATIVE_HOP_SIZE = 256;
const EXTERNAL_NEGATIVE_ONSET_THRESHOLD = 0.5;
const EXTERNAL_NEGATIVE_MIN_GAP_SEC = 0.2;
const EXTERNAL_NEGATIVES_PER_FILE = 2;
const LOCAL_PRESHOT_NEGATIVES_PER_FILE = 2;
const LOCAL_PRESHOT_ONSET_THRESHOLD = 0.15;
const LOCAL_PRESHOT_IMPACT_GAP_SEC = 0.25;

function decode(path) {
  return new Promise((resolve, reject) => {
    const ff = spawn('ffmpeg', [
      '-v', 'error',
      '-i', path,
      '-ac', '1',
      '-ar', String(MODEL_SAMPLE_RATE),
      '-f', 'f32le',
      'pipe:1',
    ]);
    const chunks = [];
    let err = '';
    ff.stdout.on('data', c => chunks.push(c));
    ff.stderr.on('data', c => { err += c.toString(); });
    ff.on('close', code => {
      if (code !== 0) return reject(new Error(err || `ffmpeg exited ${code}`));
      const buf = Buffer.concat(chunks);
      resolve(new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4));
    });
  });
}

function sourceIdFor(relPath) {
  return createHash('sha1').update(relPath).digest('hex').slice(0, 12);
}

function peakDbfs(samples, start, end) {
  let peak = 0;
  for (let i = start; i < end; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  return 20 * Math.log10(Math.max(peak, 1e-8));
}

function safeSlug(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 90) || 'clip';
}

function cropAroundStrict(samples, centerSample) {
  const start = centerSample - CLIP_PRE_SAMPLES;
  const end = centerSample + CLIP_POST_SAMPLES;
  if (start < 0 || end > samples.length) return null;
  return samples.slice(start, end);
}

function cropAroundPadded(samples, centerSample) {
  const out = new Float32Array(MODEL_CLIP_SAMPLES);
  const srcStartWanted = centerSample - CLIP_PRE_SAMPLES;
  const srcStart = Math.max(0, srcStartWanted);
  const dstStart = Math.max(0, -srcStartWanted);
  const copyLen = Math.min(samples.length - srcStart, MODEL_CLIP_SAMPLES - dstStart);
  if (copyLen > 0) out.set(samples.subarray(srcStart, srcStart + copyLen), dstStart);
  return out;
}

function recenterNear(samples, labelTimeSec, searchMs = 100) {
  const center = Math.round(labelTimeSec * MODEL_SAMPLE_RATE);
  const radius = Math.round((searchMs / 1000) * MODEL_SAMPLE_RATE);
  const lo = Math.max(0, center - radius);
  const hi = Math.min(samples.length, center + radius);
  let peakIdx = center;
  let peak = 0;
  for (let i = lo; i < hi; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) {
      peak = v;
      peakIdx = i;
    }
  }
  return peakIdx;
}

function makePositive(entry, relPath, samples, knownCenter = null) {
  const shotTime = entry.shotTimes[0];
  const center = knownCenter ?? recenterNear(samples, shotTime);
  const clip = cropAroundStrict(samples, center);
  if (!clip) throw new Error(`positive crop out of range: ${relPath}`);
  return {
    y: 1,
    sourceId: sourceIdFor(relPath),
    groupId: sourceIdFor(relPath),
    sourcePath: relPath,
    kind: 'shot_local_cropped',
    category: 'golf_shot',
    sourceName: 'self_recorded',
    clip: prepareModelClip(clip),
    centerSec: center / MODEL_SAMPLE_RATE,
    notes: 'Cropped from labeled shot time; spoken pre-roll is intentionally excluded.',
  };
}

function makeLocalPreShotNegatives(relPath, samples, shotCenter) {
  const cutoff = shotCenter - Math.round((LOCAL_PRESHOT_IMPACT_GAP_SEC + CLIP_POST_SAMPLES / MODEL_SAMPLE_RATE) * MODEL_SAMPLE_RATE);
  if (cutoff < Math.round(0.5 * MODEL_SAMPLE_RATE)) return [];

  const preShot = samples.subarray(0, cutoff);
  const { flux, times } = computeFlux(preShot);
  const peaks = pickPeaks(
    flux,
    times,
    LOCAL_PRESHOT_ONSET_THRESHOLD,
    EXTERNAL_NEGATIVE_MIN_GAP_SEC
  )
    .sort((a, b) => b.flux - a.flux)
    .slice(0, LOCAL_PRESHOT_NEGATIVES_PER_FILE);

  const out = [];
  const usedCenters = [];
  const addCandidate = (center, notes, fluxValue = null) => {
    if (center + CLIP_POST_SAMPLES > shotCenter - Math.round(LOCAL_PRESHOT_IMPACT_GAP_SEC * MODEL_SAMPLE_RATE)) return;
    if (usedCenters.some(existing => Math.abs(existing - center) < Math.round(0.25 * MODEL_SAMPLE_RATE))) return;
    const clip = cropAroundPadded(samples, center);
    const dbfs = peakDbfs(clip, 0, clip.length);
    if (dbfs < -45) return;
    usedCenters.push(center);
    out.push({
      y: 0,
      sourceId: sourceIdFor(`${relPath}#preshot#${center}`),
      groupId: sourceIdFor(relPath),
      sourcePath: relPath,
      kind: 'local_preshot_negative',
      category: 'local_preshot_voice_ambient',
      sourceName: 'self_recorded',
      clip: prepareModelClip(clip),
      centerSec: center / MODEL_SAMPLE_RATE,
      flux: fluxValue,
      notes,
    });
  };

  for (const peak of peaks) {
    addCandidate(
      recenterForward(samples, peak.time),
      'Pre-impact crop from a labeled local recording; may include spoken shot-name pre-roll.',
      peak.flux
    );
  }

  const fallbackCenters = [
    Math.round(0.75 * MODEL_SAMPLE_RATE),
    Math.round(cutoff * 0.45),
    Math.round(cutoff * 0.8),
  ];
  for (const center of fallbackCenters) {
    if (out.length >= LOCAL_PRESHOT_NEGATIVES_PER_FILE) break;
    addCandidate(
      center,
      'Fallback pre-impact crop from a labeled local recording; used to reduce local-vs-external leakage.'
    );
  }

  return out.slice(0, LOCAL_PRESHOT_NEGATIVES_PER_FILE);
}

function loudestSampleIndex(samples) {
  let peak = 0;
  let peakIdx = Math.min(CLIP_PRE_SAMPLES, Math.max(0, samples.length - 1));
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) {
      peak = v;
      peakIdx = i;
    }
  }
  return peakIdx;
}

function computeFlux(samples, fftSize = EXTERNAL_NEGATIVE_FFT_SIZE, hop = EXTERNAL_NEGATIVE_HOP_SIZE) {
  const win = hann(fftSize);
  const magNorm = 2 / fftSize;
  const flux = [];
  const times = [];
  let prevMag = null;
  for (let start = 0; start + fftSize <= samples.length; start += hop) {
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = samples[start + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;
    let f = 0;
    if (prevMag) {
      for (let k = 0; k < mag.length; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) f += d;
      }
    }
    flux.push(f);
    times.push((start + fftSize / 2) / MODEL_SAMPLE_RATE);
    prevMag = mag;
  }
  return { flux, times };
}

function pickPeaks(flux, times, threshold, minGapSec) {
  const peaks = [];
  let lastTime = -Infinity;
  for (let i = 1; i < flux.length - 1; i++) {
    if (flux[i] < threshold) continue;
    if (flux[i] <= flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    if (times[i] - lastTime < minGapSec) continue;
    peaks.push({ time: times[i], flux: flux[i] });
    lastTime = times[i];
  }
  return peaks;
}

function recenterForward(samples, timeSec, searchMs = 120) {
  const start = Math.max(0, Math.floor(timeSec * MODEL_SAMPLE_RATE));
  const end = Math.min(samples.length, start + Math.floor(searchMs * MODEL_SAMPLE_RATE / 1000));
  let peakIdx = start;
  let peak = 0;
  for (let i = start; i < end; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) {
      peak = v;
      peakIdx = i;
    }
  }
  return peakIdx;
}

function makeExternalNegatives(manifestRow, samples) {
  const relPath = manifestRow.local_path;
  const { flux, times } = computeFlux(samples);
  const peaks = pickPeaks(
    flux,
    times,
    EXTERNAL_NEGATIVE_ONSET_THRESHOLD,
    EXTERNAL_NEGATIVE_MIN_GAP_SEC
  )
    .sort((a, b) => b.flux - a.flux)
    .slice(0, EXTERNAL_NEGATIVES_PER_FILE);

  const out = [];
  for (const peak of peaks) {
    const center = recenterForward(samples, peak.time);
    const clip = cropAroundPadded(samples, center);
    const dbfs = peakDbfs(clip, 0, clip.length);
    if (dbfs < NEGATIVE_MIN_PEAK_DBFS) continue;
    out.push({
      y: 0,
      sourceId: sourceIdFor(`${relPath}#${peak.time.toFixed(4)}`),
      groupId: sourceIdFor(relPath),
      sourcePath: relPath,
      kind: 'external_onset_negative',
      category: manifestRow.category,
      sourceName: manifestRow.source_name,
      license: manifestRow.license,
      clip: prepareModelClip(clip),
      centerSec: center / MODEL_SAMPLE_RATE,
      flux: peak.flux,
      notes: manifestRow.notes || '',
    });
  }

  return out;
}

function makeFallbackExternalNegative(manifestRow, samples) {
  const relPath = manifestRow.local_path;
  const center = loudestSampleIndex(samples);
  const clip = cropAroundPadded(samples, center);
  const peak = peakDbfs(clip, 0, clip.length);
  if (peak < NEGATIVE_MIN_PEAK_DBFS) return null;
  return {
    y: 0,
    sourceId: sourceIdFor(relPath),
    groupId: sourceIdFor(relPath),
    sourcePath: relPath,
    kind: 'external_peak_negative',
    category: manifestRow.category,
    sourceName: manifestRow.source_name,
    license: manifestRow.license,
    clip: prepareModelClip(clip),
    centerSec: center / MODEL_SAMPLE_RATE,
    notes: `fallback peak crop; ${manifestRow.notes || ''}`,
  };
}

function featuresForExamples(examples) {
  return examples.map(ex => ({
    ...ex,
    x: extractStage1Features(ex.clip),
    clip: undefined,
  }));
}

function encodeWavBuffer(samples, sampleRate = MODEL_SAMPLE_RATE) {
  const bytesPerSample = 2;
  const byteLength = samples.length * bytesPerSample;
  const buffer = Buffer.alloc(44 + byteLength);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + byteLength, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * bytesPerSample, 28);
  buffer.writeUInt16LE(bytesPerSample, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(byteLength, 40);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff), offset);
    offset += 2;
  }
  return buffer;
}

async function writePreparedDataset(examples) {
  const shotDir = join(PREPARED_DIR, 'shot');
  const notShotDir = join(PREPARED_DIR, 'not_shot');
  await rm(PREPARED_DIR, { recursive: true, force: true });
  await mkdir(shotDir, { recursive: true });
  await mkdir(notShotDir, { recursive: true });

  const rows = [];
  for (const ex of examples) {
    const subdir = ex.y ? shotDir : notShotDir;
    const prefix = ex.y ? 'shot' : 'not_shot';
    const filename = `${prefix}__${ex.sourceId}__${safeSlug(basename(ex.sourcePath))}.wav`;
    const absPath = join(subdir, filename);
    const localPath = absPath.replace(ROOT + '/', '');
    await writeFile(absPath, encodeWavBuffer(ex.clip));
    rows.push({
      local_path: localPath,
      label: ex.y ? 'shot' : 'not_shot',
      source_path: ex.sourcePath,
      source_id: ex.sourceId,
      group_id: ex.groupId,
      source_name: ex.sourceName,
      category: ex.category,
      kind: ex.kind,
      center_sec: +ex.centerSec.toFixed(4),
      sample_rate_hz: MODEL_SAMPLE_RATE,
      clip_samples: MODEL_CLIP_SAMPLES,
      notes: ex.notes,
    });
  }

  await writeFile(PREPARED_MANIFEST_PATH, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  return rows;
}

function meanStd(rows, indices) {
  const n = indices.length;
  const d = rows[0].x.length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (const idx of indices) {
    for (let j = 0; j < d; j++) mean[j] += rows[idx].x[j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (const idx of indices) {
    for (let j = 0; j < d; j++) {
      const delta = rows[idx].x[j] - mean[j];
      std[j] += delta * delta;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / Math.max(1, n - 1)) || 1;
  return { mean, std };
}

function standardize(row, scaler) {
  return row.x.map((v, i) => (v - scaler.mean[i]) / scaler.std[i]);
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function trainLogistic(rows, indices, scaler, opts = {}) {
  const d = rows[0].x.length;
  const weights = new Array(d).fill(0);
  let bias = 0;
  const epochs = opts.epochs ?? 2500;
  const baseLr = opts.lr ?? 0.08;
  const l2 = opts.l2 ?? 0.01;
  const ys = indices.map(i => rows[i].y);
  const nPos = ys.filter(Boolean).length;
  const nNeg = ys.length - nPos;
  const posWeight = ys.length / Math.max(1, 2 * nPos);
  const negWeight = ys.length / Math.max(1, 2 * nNeg);
  const norm = indices.reduce((acc, idx) => acc + (rows[idx].y ? posWeight : negWeight), 0);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = weights.map(w => l2 * w);
    let gradB = 0;
    for (const idx of indices) {
      const row = rows[idx];
      const x = standardize(row, scaler);
      let z = bias;
      for (let j = 0; j < d; j++) z += weights[j] * x[j];
      const p = sigmoid(z);
      const sampleWeight = row.y ? posWeight : negWeight;
      const err = (p - row.y) * sampleWeight;
      for (let j = 0; j < d; j++) gradW[j] += (err * x[j]) / norm;
      gradB += err / norm;
    }
    const lr = baseLr / (1 + epoch / 1000);
    for (let j = 0; j < d; j++) weights[j] -= lr * gradW[j];
    bias -= lr * gradB;
  }
  return { weights, bias };
}

function predict(row, model, scaler) {
  const x = standardize(row, scaler);
  let z = model.bias;
  for (let j = 0; j < x.length; j++) z += model.weights[j] * x[j];
  return sigmoid(z);
}

function groupKey(row) {
  return row.groupId ?? row.sourceId ?? row.sourcePath;
}

function makeFolds(rows, k = 5) {
  const groups = new Map();
  rows.forEach((row, idx) => {
    const key = groupKey(row);
    if (!groups.has(key)) {
      groups.set(key, { key, indices: [], positives: 0, negatives: 0 });
    }
    const group = groups.get(key);
    group.indices.push(idx);
    if (row.y) group.positives++;
    else group.negatives++;
  });

  const foldStates = Array.from({ length: k }, (_, i) => ({
    i,
    groups: new Set(),
    positives: 0,
    negatives: 0,
    total: 0,
  }));

  const assign = (group, compare) => {
    const fold = [...foldStates].sort(compare)[0];
    fold.groups.add(group.key);
    fold.positives += group.positives;
    fold.negatives += group.negatives;
    fold.total += group.indices.length;
  };

  const positiveGroups = [...groups.values()]
    .filter(g => g.positives > 0)
    .sort((a, b) => b.positives - a.positives || b.indices.length - a.indices.length || a.key.localeCompare(b.key));
  const negativeOnlyGroups = [...groups.values()]
    .filter(g => g.positives === 0)
    .sort((a, b) => b.indices.length - a.indices.length || a.key.localeCompare(b.key));

  for (const group of positiveGroups) {
    assign(group, (a, b) =>
      a.positives - b.positives ||
      a.total - b.total ||
      a.negatives - b.negatives ||
      a.i - b.i
    );
  }

  for (const group of negativeOnlyGroups) {
    assign(group, (a, b) =>
      a.total - b.total ||
      a.negatives - b.negatives ||
      a.i - b.i
    );
  }

  return foldStates.map(fold => ({
    val: rows.map((r, i) => fold.groups.has(groupKey(r)) ? i : -1).filter(i => i >= 0),
    train: rows.map((r, i) => fold.groups.has(groupKey(r)) ? -1 : i).filter(i => i >= 0),
    summary: {
      groups: fold.groups.size,
      positives: fold.positives,
      negatives: fold.negatives,
      total: fold.total,
    },
  }));
}

function metrics(scored, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const s of scored) {
    const pred = s.p >= threshold ? 1 : 0;
    if (pred === 1 && s.y === 1) tp++;
    else if (pred === 1 && s.y === 0) fp++;
    else if (pred === 0 && s.y === 0) tn++;
    else fn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const specificity = tn + fp ? tn / (tn + fp) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return { threshold, tp, fp, tn, fn, precision, recall, specificity, f1 };
}

function chooseThreshold(scored) {
  const candidates = [];
  for (let t = 0.05; t <= 0.95; t += 0.01) candidates.push(+t.toFixed(2));
  const evaluated = candidates.map(t => metrics(scored, t));
  const perfectRecall = evaluated.filter(m => m.recall >= 0.999);
  if (perfectRecall.length) {
    return perfectRecall.reduce((best, cur) => {
      if (cur.precision !== best.precision) return cur.precision > best.precision ? cur : best;
      if (cur.f1 !== best.f1) return cur.f1 > best.f1 ? cur : best;
      return cur.threshold > best.threshold ? cur : best;
    });
  }
  const highRecall = evaluated.filter(m => m.recall >= 0.95);
  if (highRecall.length) {
    return highRecall.reduce((best, cur) => {
      if (cur.precision !== best.precision) return cur.precision > best.precision ? cur : best;
      if (cur.f1 !== best.f1) return cur.f1 > best.f1 ? cur : best;
      return cur.threshold > best.threshold ? cur : best;
    });
  }
  return evaluated.reduce((best, cur) => cur.f1 > best.f1 ? cur : best);
}

function chooseThresholdByWorstFold(foldScored, minRecall = 0.95) {
  const candidates = [];
  for (let t = 0.05; t <= 0.95; t += 0.01) candidates.push(+t.toFixed(2));
  const evaluated = candidates.map(threshold => {
    const folds = foldScored.map(scored => metrics(scored, threshold));
    const pooled = metrics(foldScored.flat(), threshold);
    const worstRecall = Math.min(...folds.map(f => f.recall));
    const worstSpecificity = Math.min(...folds.map(f => f.specificity));
    return { ...pooled, folds, worstRecall, worstSpecificity };
  });

  const safe = evaluated.filter(e => e.worstRecall >= minRecall);
  if (safe.length) {
    // Highest threshold that preserves per-fold recall gives the lowest FP rate
    // without hiding a weak validation fold behind pooled metrics.
    return safe.reduce((best, cur) => cur.threshold > best.threshold ? cur : best);
  }

  return evaluated.reduce((best, cur) => {
    if (cur.worstRecall !== best.worstRecall) return cur.worstRecall > best.worstRecall ? cur : best;
    return cur.f1 > best.f1 ? cur : best;
  });
}

async function loadExternalNegativeRows() {
  try {
    const raw = await readFile(EXTERNAL_MANIFEST_PATH, 'utf8');
    return raw
      .split('\n')
      .filter(Boolean)
      .map(line => JSON.parse(line))
      .filter(row =>
        row.polarity === 'negative' &&
        row.split_candidate === 'trainable' &&
        row.ai_training_permission === 'yes'
      );
  } catch (e) {
    console.warn(`No external manifest loaded: ${e.message}`);
    return [];
  }
}

function countBy(rows, key) {
  return rows.reduce((acc, row) => {
    const value = row[key] ?? 'unknown';
    acc[value] = (acc[value] || 0) + 1;
    return acc;
  }, {});
}

function summarizeScores(values) {
  if (!values.length) return { count: 0, min: null, p05: null, p50: null, p95: null, max: null, mean: null };
  const sorted = values.slice().sort((a, b) => a - b);
  const quantile = q => {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  return {
    count: sorted.length,
    min: sorted[0],
    p05: quantile(0.05),
    p50: quantile(0.5),
    p95: quantile(0.95),
    max: sorted[sorted.length - 1],
    mean: values.reduce((acc, v) => acc + v, 0) / values.length,
  };
}

function scoreSeparation(scored) {
  const positiveScores = scored.filter(s => s.y === 1).map(s => s.p);
  const negativeScores = scored.filter(s => s.y === 0).map(s => s.p);
  const positives = summarizeScores(positiveScores);
  const negatives = summarizeScores(negativeScores);
  const minPositiveP = positives.min;
  const maxNegativeP = negatives.max;
  return {
    positives,
    negatives,
    minPositiveP,
    maxNegativeP,
    separationMargin: minPositiveP == null || maxNegativeP == null ? null : minPositiveP - maxNegativeP,
  };
}

const labelsDoc = JSON.parse(await readFile(LABELS_PATH, 'utf8'));
const rawEntries = Object.values(labelsDoc.labels).filter(e => e.shotTimes?.length);
const externalNegativeRows = (await loadExternalNegativeRows()).slice(0, MAX_EXTERNAL_NEGATIVES);
const examples = [];
const skippedExternalNegatives = [];

for (const entry of rawEntries) {
  const relPath = entry.path.replace(/^samples\//, '');
  const absPath = join(ROOT, relPath);
  const samples = await decode(absPath);
  const shotCenter = recenterNear(samples, entry.shotTimes[0]);
  examples.push(makePositive(entry, relPath, samples, shotCenter));
  examples.push(...makeLocalPreShotNegatives(relPath, samples, shotCenter));
}

let externalDone = 0;
for (const row of externalNegativeRows) {
  externalDone++;
  if (externalDone % 100 === 0) {
    console.log(`Decoded external negatives ${externalDone}/${externalNegativeRows.length}`);
  }
  try {
    const samples = await decode(join(ROOT, row.local_path));
    const negativeExamples = makeExternalNegatives(row, samples);
    if (negativeExamples.length) {
      examples.push(...negativeExamples);
    } else {
      const fallback = makeFallbackExternalNegative(row, samples);
      if (fallback) skippedExternalNegatives.push({ local_path: row.local_path, reason: 'no_flux_onset_candidate' });
      else skippedExternalNegatives.push({ local_path: row.local_path, reason: 'below_peak_threshold' });
    }
  } catch (e) {
    skippedExternalNegatives.push({ local_path: row.local_path, reason: e.message });
  }
}

const preparedManifestRows = await writePreparedDataset(examples);
const rows = featuresForExamples(examples);
const nPos = rows.filter(r => r.y === 1).length;
const nNeg = rows.filter(r => r.y === 0).length;
const folds = makeFolds(rows, 5);
const oof = [];
const foldScored = [];
const foldReports = [];

for (let i = 0; i < folds.length; i++) {
  const fold = folds[i];
  const scaler = meanStd(rows, fold.train);
  const model = trainLogistic(rows, fold.train, scaler);
  const scored = fold.val.map(idx => ({ idx, y: rows[idx].y, p: predict(rows[idx], model, scaler) }));
  oof.push(...scored);
  foldScored.push(scored);
  foldReports.push({
    fold: i + 1,
    train: fold.train.length,
    val: fold.val.length,
    valGroups: fold.summary.groups,
    valPositives: fold.summary.positives,
    valNegatives: fold.summary.negatives,
    at_050: metrics(scored, 0.5),
  });
}

const pooledThresholdMetrics = chooseThreshold(oof);
const thresholdMetrics = chooseThresholdByWorstFold(foldScored, 0.95);
const oofScoreSeparation = scoreSeparation(oof);
const allIndices = rows.map((_, i) => i);
const finalScaler = meanStd(rows, allIndices);
const finalModel = trainLogistic(rows, allIndices, finalScaler);
const allScored = rows.map((row, idx) => ({ idx, y: row.y, p: predict(row, finalModel, finalScaler) }));
const trainScoreSeparation = scoreSeparation(allScored);
const trainOnlyMetrics = chooseThreshold(allScored);
const trainMetricsAtCvThreshold = metrics(allScored, thresholdMetrics.threshold);

const modelOut = {
  version: 1,
  type: 'standardized_logistic_regression',
  task: 'stage1b_shot_verifier',
  featureExtractor: 'stage1_handcrafted',
  sampleRate: MODEL_SAMPLE_RATE,
  clipSamples: MODEL_CLIP_SAMPLES,
  threshold: thresholdMetrics.threshold,
  features: STAGE1_FEATURE_NAMES,
  mean: finalScaler.mean,
  std: finalScaler.std,
  weights: finalModel.weights,
  bias: finalModel.bias,
  training: {
    generatedAt: new Date().toISOString(),
    positives: nPos,
    negatives: nNeg,
    localPositiveSourceFiles: rawEntries.length,
    externalNegativeSourceFiles: externalNegativeRows.length,
    skippedExternalNegatives: skippedExternalNegatives.length,
    negativeMinPeakDbfs: NEGATIVE_MIN_PEAK_DBFS,
    preparedManifest: PREPARED_MANIFEST_PATH.replace(ROOT + '/', ''),
    thresholdSelection: 'cross_validation_worst_fold_recall',
    cvRecommendedThreshold: thresholdMetrics.threshold,
    pooledCvThreshold: pooledThresholdMetrics.threshold,
    cv: thresholdMetrics,
    pooledCv: pooledThresholdMetrics,
    oofScoreSeparation,
    trainAtCvThreshold: trainMetricsAtCvThreshold,
    trainScoreSeparation,
    trainOnlyMetrics,
  },
};

const report = {
  generatedAt: modelOut.training.generatedAt,
  positives: nPos,
  negatives: nNeg,
  preparedManifest: modelOut.training.preparedManifest,
  preparedCounts: countBy(preparedManifestRows, 'label'),
  negativeCategories: countBy(rows.filter(r => r.y === 0), 'category'),
  negativeSources: countBy(rows.filter(r => r.y === 0), 'sourceName'),
  positiveSources: countBy(rows.filter(r => r.y === 1), 'sourceName'),
  skippedExternalNegatives,
  threshold: thresholdMetrics.threshold,
  thresholdSelection: 'cross_validation_worst_fold_recall',
  cvRecommendedThreshold: thresholdMetrics.threshold,
  pooledCvThreshold: pooledThresholdMetrics.threshold,
  cv: thresholdMetrics,
  pooledCv: pooledThresholdMetrics,
  oofScoreSeparation,
  trainAtCvThreshold: trainMetricsAtCvThreshold,
  trainScoreSeparation,
  trainOnlyMetrics,
  foldDistribution: folds.map((fold, i) => ({
    fold: i + 1,
    ...fold.summary,
    train: fold.train.length,
    val: fold.val.length,
  })),
  folds: foldReports,
  hardestFalseNegativesAtThreshold: oof
    .filter(s => s.y === 1 && s.p < thresholdMetrics.threshold)
    .sort((a, b) => a.p - b.p)
    .map(s => ({ p: s.p, sourcePath: rows[s.idx].sourcePath, centerSec: rows[s.idx].centerSec })),
  hardestFalsePositivesAtThreshold: oof
    .filter(s => s.y === 0 && s.p >= thresholdMetrics.threshold)
    .sort((a, b) => b.p - a.p)
    .slice(0, 20)
    .map(s => ({ p: s.p, sourcePath: rows[s.idx].sourcePath, centerSec: rows[s.idx].centerSec })),
};

await mkdir(dirname(HANDCRAFTED_MODEL_PATH), { recursive: true });
await mkdir(dirname(HANDCRAFTED_REPORT_PATH), { recursive: true });
await writeFile(HANDCRAFTED_MODEL_PATH, JSON.stringify(modelOut, null, 2));
await writeFile(HANDCRAFTED_REPORT_PATH, JSON.stringify(report, null, 2));

console.log(`Stage 1b detector trained: ${nPos} positives, ${nNeg} negatives`);
console.log(`Prepared dataset written: ${PREPARED_MANIFEST_PATH}`);
console.log(`Skipped external negatives: ${skippedExternalNegatives.length}`);
console.log(`CV threshold ${thresholdMetrics.threshold.toFixed(2)}  P=${thresholdMetrics.precision.toFixed(3)} R=${thresholdMetrics.recall.toFixed(3)} F1=${thresholdMetrics.f1.toFixed(3)} specificity=${thresholdMetrics.specificity.toFixed(3)} worstFoldR=${thresholdMetrics.worstRecall.toFixed(3)} worstFoldSpec=${thresholdMetrics.worstSpecificity.toFixed(3)}`);
console.log(`OOF margin minPos=${oofScoreSeparation.minPositiveP?.toFixed(3) ?? 'n/a'} maxNeg=${oofScoreSeparation.maxNegativeP?.toFixed(3) ?? 'n/a'} margin=${oofScoreSeparation.separationMargin?.toFixed(3) ?? 'n/a'}`);
console.log(`Train-only threshold ${trainOnlyMetrics.threshold.toFixed(2)}  P=${trainOnlyMetrics.precision.toFixed(3)} R=${trainOnlyMetrics.recall.toFixed(3)} F1=${trainOnlyMetrics.f1.toFixed(3)} specificity=${trainOnlyMetrics.specificity.toFixed(3)} (not used for deployment)`);
console.log(`Wrote ${HANDCRAFTED_MODEL_PATH}`);
console.log(`Wrote ${HANDCRAFTED_REPORT_PATH}`);
