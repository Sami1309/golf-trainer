import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MODEL_CLIP_SAMPLES,
  MODEL_SAMPLE_RATE,
  STAGE1_FEATURE_NAMES,
  extractStage1Features,
  prepareModelClip,
} from '../frontend/audio_features.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PREPARED_MANIFEST_PATH = join(ROOT, 'data', 'stage1b_prepared', 'manifest.jsonl');
const HANDCRAFTED_MODEL_PATH = join(ROOT, 'frontend', 'models', 'stage1b_handcrafted.json');
const HANDCRAFTED_REPORT_PATH = join(ROOT, 'data', 'stage1b_handcrafted_report.json');

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

async function loadRows() {
  const raw = await readFile(PREPARED_MANIFEST_PATH, 'utf8');
  const manifestRows = raw.split('\n').filter(Boolean).map(line => JSON.parse(line));
  const rows = [];
  for (let i = 0; i < manifestRows.length; i++) {
    const manifest = manifestRows[i];
    if ((i + 1) % 100 === 0) console.log(`Extracted handcrafted features ${i + 1}/${manifestRows.length}`);
    const clip = await readPreparedClip(manifest.local_path);
    rows.push({
      y: manifest.label === 'shot' ? 1 : 0,
      x: extractStage1Features(clip),
      sourceId: manifest.source_id,
      groupId: manifest.group_id,
      sourcePath: manifest.source_path,
      localPath: manifest.local_path,
      sourceName: manifest.source_name,
      category: manifest.category,
      kind: manifest.kind,
      centerSec: manifest.center_sec,
    });
  }
  return rows;
}

const rows = await loadRows();
if (!rows.length) throw new Error(`No prepared examples found at ${PREPARED_MANIFEST_PATH}. Run "npm run prepare:stage1b" first.`);
if (rows[0].x.length !== STAGE1_FEATURE_NAMES.length) {
  throw new Error(`handcrafted feature length mismatch: ${rows[0].x.length} != ${STAGE1_FEATURE_NAMES.length}`);
}

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
  modelPath: HANDCRAFTED_MODEL_PATH.replace(ROOT + '/', ''),
  positives: nPos,
  negatives: nNeg,
  preparedManifest: modelOut.training.preparedManifest,
  preparedCounts: countBy(rows.map(r => ({ label: r.y ? 'shot' : 'not_shot' })), 'label'),
  negativeCategories: countBy(rows.filter(r => r.y === 0), 'category'),
  negativeSources: countBy(rows.filter(r => r.y === 0), 'sourceName'),
  positiveSources: countBy(rows.filter(r => r.y === 1), 'sourceName'),
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

console.log(`Stage 1b handcrafted baseline trained: ${nPos} positives, ${nNeg} negatives`);
console.log(`CV threshold ${thresholdMetrics.threshold.toFixed(2)}  P=${thresholdMetrics.precision.toFixed(3)} R=${thresholdMetrics.recall.toFixed(3)} F1=${thresholdMetrics.f1.toFixed(3)} specificity=${thresholdMetrics.specificity.toFixed(3)} worstFoldR=${thresholdMetrics.worstRecall.toFixed(3)} worstFoldSpec=${thresholdMetrics.worstSpecificity.toFixed(3)}`);
console.log(`OOF margin minPos=${oofScoreSeparation.minPositiveP?.toFixed(3) ?? 'n/a'} maxNeg=${oofScoreSeparation.maxNegativeP?.toFixed(3) ?? 'n/a'} margin=${oofScoreSeparation.separationMargin?.toFixed(3) ?? 'n/a'}`);
console.log(`Train-only threshold ${trainOnlyMetrics.threshold.toFixed(2)}  P=${trainOnlyMetrics.precision.toFixed(3)} R=${trainOnlyMetrics.recall.toFixed(3)} F1=${trainOnlyMetrics.f1.toFixed(3)} specificity=${trainOnlyMetrics.specificity.toFixed(3)} (not used for deployment)`);
console.log(`Wrote ${HANDCRAFTED_MODEL_PATH}`);
console.log(`Wrote ${HANDCRAFTED_REPORT_PATH}`);
