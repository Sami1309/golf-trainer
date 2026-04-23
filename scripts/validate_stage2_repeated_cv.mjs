import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  LOG_MEL_FEATURE_NAMES,
  MODEL_CLIP_SAMPLES,
  MODEL_SAMPLE_RATE,
  extractLogMelFeatures,
  prepareModelClip,
} from '../frontend/audio_features.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PREPARED_MANIFEST_PATH = join(ROOT, 'data', 'stage1b_prepared', 'manifest.jsonl');
const REPORT_PATH = join(ROOT, 'data', 'stage2_pure_fat_repeated_cv_report.json');
const REPEATS = Number(process.env.REPEATS || 200);
const FOLDS = Number(process.env.FOLDS || 5);
const BASE_SEED = Number(process.env.SEED || 20260423);

function parseWavPcm16Mono(buffer, relPath) {
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`not a RIFF/WAVE file: ${relPath}`);
  }

  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;
  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ') {
      fmt = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      dataOffset = start;
      dataSize = size;
    }
    offset = start + size + (size % 2);
  }

  if (!fmt || dataOffset < 0) throw new Error(`missing wav fmt/data chunk: ${relPath}`);
  if (fmt.audioFormat !== 1 || fmt.bitsPerSample !== 16) {
    throw new Error(`unsupported wav format in ${relPath}: format=${fmt.audioFormat} bits=${fmt.bitsPerSample}`);
  }
  if (fmt.sampleRate !== MODEL_SAMPLE_RATE) {
    throw new Error(`unexpected sample rate in ${relPath}: ${fmt.sampleRate}`);
  }

  const frames = Math.floor(dataSize / (2 * fmt.channels));
  const samples = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let ch = 0; ch < fmt.channels; ch++) {
      sum += buffer.readInt16LE(dataOffset + (i * fmt.channels + ch) * 2) / 32768;
    }
    samples[i] = sum / fmt.channels;
  }
  return prepareModelClip(samples);
}

async function readPreparedClip(relPath) {
  const buffer = await readFile(join(ROOT, relPath));
  const samples = parseWavPcm16Mono(buffer, relPath);
  if (samples.length !== MODEL_CLIP_SAMPLES) {
    throw new Error(`unexpected clip length in ${relPath}: ${samples.length}`);
  }
  return samples;
}

function classFromTitle(sourcePath) {
  const title = sourcePath.split('/')[0].toLowerCase();
  if (title.includes('topped')) return null;
  if (title.includes('1mm')) return null;
  if (title.includes('pure')) return 'pure';
  if (title.includes('fat')) return 'fat';
  return null;
}

function mulberry32(seed) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(values, random) {
  const out = values.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
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

function trainLogistic(rows, indices, scaler, opts = {}) {
  const d = rows[0].x.length;
  const weights = new Array(d).fill(0);
  let bias = 0;
  const epochs = opts.epochs ?? 1800;
  const baseLr = opts.lr ?? 0.035;
  const l2 = opts.l2 ?? 0.12;
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
    const lr = baseLr / (1 + epoch / 900);
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

function makeRandomStratifiedFolds(rows, repeats, k, seed) {
  const pure = rows.map((r, i) => r.y === 1 ? i : -1).filter(i => i >= 0);
  const fat = rows.map((r, i) => r.y === 0 ? i : -1).filter(i => i >= 0);
  const out = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    const random = mulberry32(seed + repeat);
    const pureShuffled = shuffle(pure, random);
    const fatShuffled = shuffle(fat, random);
    const foldSets = Array.from({ length: k }, () => new Set());
    pureShuffled.forEach((idx, i) => foldSets[i % k].add(idx));
    fatShuffled.forEach((idx, i) => foldSets[i % k].add(idx));
    out.push(foldSets.map(valSet => ({
      val: rows.map((_, i) => valSet.has(i) ? i : -1).filter(i => i >= 0),
      train: rows.map((_, i) => valSet.has(i) ? -1 : i).filter(i => i >= 0),
    })));
  }
  return out;
}

function metrics(scored, threshold = 0.5) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const s of scored) {
    const pred = s.pPure >= threshold ? 1 : 0;
    if (pred === 1 && s.y === 1) tp++;
    else if (pred === 1 && s.y === 0) fp++;
    else if (pred === 0 && s.y === 0) tn++;
    else fn++;
  }
  const accuracy = scored.length ? (tp + tn) / scored.length : 0;
  const pureRecall = tp + fn ? tp / (tp + fn) : 0;
  const fatRecall = tn + fp ? tn / (tn + fp) : 0;
  const balancedAccuracy = (pureRecall + fatRecall) / 2;
  return { tpPure: tp, fpPure: fp, tnFat: tn, fnPure: fn, accuracy, pureRecall, fatRecall, balancedAccuracy };
}

function confidenceMetrics(scored, confidenceThreshold = 0.6) {
  const kept = scored.filter(s => Math.max(s.pPure, 1 - s.pPure) >= confidenceThreshold);
  return {
    confidenceThreshold,
    kept: kept.length,
    unsure: scored.length - kept.length,
    coverage: scored.length ? kept.length / scored.length : 0,
    metrics: metrics(kept, 0.5),
  };
}

function summarize(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const q = p => {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * p;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  return {
    count: sorted.length,
    min: sorted[0] ?? null,
    p05: q(0.05),
    p25: q(0.25),
    median: q(0.5),
    p75: q(0.75),
    p95: q(0.95),
    max: sorted[sorted.length - 1] ?? null,
    mean: sorted.length ? values.reduce((acc, v) => acc + v, 0) / values.length : null,
  };
}

async function loadRows() {
  const manifestRows = (await readFile(PREPARED_MANIFEST_PATH, 'utf8'))
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .filter(row => row.label === 'shot');

  const rows = [];
  const excluded = [];
  for (const manifest of manifestRows) {
    const className = classFromTitle(manifest.source_path);
    if (!className) {
      excluded.push({ sourcePath: manifest.source_path, title: manifest.source_path.split('/')[0] });
      continue;
    }
    const clip = await readPreparedClip(manifest.local_path);
    rows.push({
      y: className === 'pure' ? 1 : 0,
      className,
      x: extractLogMelFeatures(clip),
      sourcePath: manifest.source_path,
      title: manifest.source_path.split('/')[0],
    });
  }
  return { rows, excluded };
}

const { rows, excluded } = await loadRows();
if (!rows.length) throw new Error(`No pure/fat rows loaded from ${PREPARED_MANIFEST_PATH}`);
if (rows[0].x.length !== LOG_MEL_FEATURE_NAMES.length) {
  throw new Error(`feature length mismatch: ${rows[0].x.length} != ${LOG_MEL_FEATURE_NAMES.length}`);
}

const foldsByRepeat = makeRandomStratifiedFolds(rows, REPEATS, FOLDS, BASE_SEED);
const repeatReports = [];
const perExample = rows.map(row => ({
  sourcePath: row.sourcePath,
  title: row.title,
  actual: row.className,
  seen: 0,
  correct: 0,
  confidentSeen: 0,
  confidentCorrect: 0,
  pPureValues: [],
}));

for (let repeat = 0; repeat < foldsByRepeat.length; repeat++) {
  const scored = [];
  const foldReports = [];
  for (let foldIndex = 0; foldIndex < foldsByRepeat[repeat].length; foldIndex++) {
    const fold = foldsByRepeat[repeat][foldIndex];
    const scaler = meanStd(rows, fold.train);
    const model = trainLogistic(rows, fold.train, scaler);
    const foldScored = fold.val.map(idx => ({ idx, y: rows[idx].y, pPure: predict(rows[idx], model, scaler) }));
    scored.push(...foldScored);
    foldReports.push({
      fold: foldIndex + 1,
      train: fold.train.length,
      val: fold.val.length,
      pure: fold.val.filter(i => rows[i].y === 1).length,
      fat: fold.val.filter(i => rows[i].y === 0).length,
      metrics: metrics(foldScored, 0.5),
    });
  }

  for (const s of scored) {
    const entry = perExample[s.idx];
    const predicted = s.pPure >= 0.5 ? 1 : 0;
    const confident = Math.max(s.pPure, 1 - s.pPure) >= 0.6;
    entry.seen++;
    if (predicted === s.y) entry.correct++;
    if (confident) {
      entry.confidentSeen++;
      if (predicted === s.y) entry.confidentCorrect++;
    }
    entry.pPureValues.push(s.pPure);
  }

  const base = metrics(scored, 0.5);
  const conf = confidenceMetrics(scored, 0.6);
  repeatReports.push({
    repeat: repeat + 1,
    seed: BASE_SEED + repeat,
    metrics: base,
    confidence060: conf,
    folds: foldReports,
  });
}

const accuracyValues = repeatReports.map(r => r.metrics.accuracy);
const balancedValues = repeatReports.map(r => r.metrics.balancedAccuracy);
const pureRecallValues = repeatReports.map(r => r.metrics.pureRecall);
const fatRecallValues = repeatReports.map(r => r.metrics.fatRecall);
const coverageValues = repeatReports.map(r => r.confidence060.coverage);
const keptAccuracyValues = repeatReports.map(r => r.confidence060.metrics.accuracy);

const perExampleSummary = perExample
  .map(e => ({
    sourcePath: e.sourcePath,
    title: e.title,
    actual: e.actual,
    seen: e.seen,
    accuracy: e.seen ? e.correct / e.seen : null,
    confidentSeen: e.confidentSeen,
    confidentAccuracy: e.confidentSeen ? e.confidentCorrect / e.confidentSeen : null,
    pPure: summarize(e.pPureValues),
  }))
  .sort((a, b) => (a.accuracy ?? 0) - (b.accuracy ?? 0) || a.sourcePath.localeCompare(b.sourcePath));

const report = {
  generatedAt: new Date().toISOString(),
  preparedManifest: PREPARED_MANIFEST_PATH.replace(ROOT + '/', ''),
  labelSource: 'source_path folder/title parsed from prepared manifest',
  featureExtractor: 'logmel_summary',
  rows: rows.length,
  pure: rows.filter(r => r.y === 1).length,
  fat: rows.filter(r => r.y === 0).length,
  excluded,
  repeats: REPEATS,
  folds: FOLDS,
  baseSeed: BASE_SEED,
  summary: {
    accuracy: summarize(accuracyValues),
    balancedAccuracy: summarize(balancedValues),
    pureRecall: summarize(pureRecallValues),
    fatRecall: summarize(fatRecallValues),
    confidence060Coverage: summarize(coverageValues),
    confidence060KeptAccuracy: summarize(keptAccuracyValues),
    repeatsAtOrAbove070: accuracyValues.filter(v => v >= 0.7).length,
    repeatsAtOrAbove080: accuracyValues.filter(v => v >= 0.8).length,
    repeatsAtOrAbove090: accuracyValues.filter(v => v >= 0.9).length,
  },
  weakestExamples: perExampleSummary.slice(0, 10),
  perExample: perExampleSummary,
  repeatReports,
};

await writeFile(REPORT_PATH, JSON.stringify(report, null, 2));

console.log(`Repeated Stage 2 pure/fat CV: ${REPEATS}x ${FOLDS}-fold, ${rows.length} examples (${report.pure} pure / ${report.fat} fat), excluded ${excluded.length}`);
console.log(`Accuracy mean=${report.summary.accuracy.mean.toFixed(3)} median=${report.summary.accuracy.median.toFixed(3)} min=${report.summary.accuracy.min.toFixed(3)} p05=${report.summary.accuracy.p05.toFixed(3)} p95=${report.summary.accuracy.p95.toFixed(3)} max=${report.summary.accuracy.max.toFixed(3)}`);
console.log(`Pure recall mean=${report.summary.pureRecall.mean.toFixed(3)} Fat recall mean=${report.summary.fatRecall.mean.toFixed(3)}`);
console.log(`Conf>=0.60 coverage mean=${report.summary.confidence060Coverage.mean.toFixed(3)} keptAcc mean=${report.summary.confidence060KeptAccuracy.mean.toFixed(3)}`);
console.log(`Repeats >=0.70 acc: ${report.summary.repeatsAtOrAbove070}/${REPEATS}; >=0.80: ${report.summary.repeatsAtOrAbove080}/${REPEATS}; >=0.90: ${report.summary.repeatsAtOrAbove090}/${REPEATS}`);
console.log(`Wrote ${REPORT_PATH}`);
