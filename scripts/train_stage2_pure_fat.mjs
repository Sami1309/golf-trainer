import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  LOG_MEL_FEATURE_NAMES,
  MODEL_CLIP_SAMPLES,
  MODEL_SAMPLE_RATE,
  extractLogMelFeatures,
  prepareModelClip,
} from '../frontend/audio_features.js';
import {
  classFromFolder,
  exclusionForFolder,
  loadStage2PureFatPolicy,
  summarizeStage2PureFatPolicy,
} from './stage2_pure_fat_policy.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LABELS_PATH = join(ROOT, 'data', 'labels.json');
const PREPARED_MANIFEST_PATH = join(ROOT, 'data', 'stage1b_prepared', 'manifest.jsonl');
const MODEL_PATH = join(ROOT, 'frontend', 'models', 'stage2_pure_fat.json');
const REPORT_PATH = join(ROOT, 'data', 'stage2_pure_fat_report.json');

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

function groupKey(row) {
  return row.groupId ?? row.sourceId ?? row.sourcePath;
}

function makeFolds(rows, k = 5) {
  const pureGroups = rows.filter(r => r.y === 1).map(groupKey).sort();
  const fatGroups = rows.filter(r => r.y === 0).map(groupKey).sort();
  const folds = Array.from({ length: k }, () => new Set());
  pureGroups.forEach((group, i) => folds[i % k].add(group));
  fatGroups.forEach((group, i) => folds[i % k].add(group));
  return folds.map(groupSet => {
    const val = rows.map((r, i) => groupSet.has(groupKey(r)) ? i : -1).filter(i => i >= 0);
    const train = rows.map((r, i) => groupSet.has(groupKey(r)) ? -1 : i).filter(i => i >= 0);
    return {
      val,
      train,
      summary: {
        pure: val.filter(i => rows[i].y === 1).length,
        fat: val.filter(i => rows[i].y === 0).length,
      },
    };
  });
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
  const purePrecision = tp + fp ? tp / (tp + fp) : 0;
  const pureRecall = tp + fn ? tp / (tp + fn) : 0;
  const fatPrecision = tn + fn ? tn / (tn + fn) : 0;
  const fatRecall = tn + fp ? tn / (tn + fp) : 0;
  const accuracy = scored.length ? (tp + tn) / scored.length : 0;
  return { threshold, tpPure: tp, fpPure: fp, tnFat: tn, fnPure: fn, accuracy, purePrecision, pureRecall, fatPrecision, fatRecall };
}

function confidenceMetrics(scored, confidenceThreshold = 0.6) {
  const kept = scored.filter(s => Math.max(s.pPure, 1 - s.pPure) >= confidenceThreshold);
  const unsure = scored.length - kept.length;
  return { confidenceThreshold, unsure, coverage: scored.length ? kept.length / scored.length : 0, kept: metrics(kept, 0.5) };
}

function scoreSummary(scored) {
  const pureScores = scored.filter(s => s.y === 1).map(s => s.pPure).sort((a, b) => a - b);
  const fatScores = scored.filter(s => s.y === 0).map(s => s.pPure).sort((a, b) => a - b);
  return {
    minPureP: pureScores[0] ?? null,
    maxFatP: fatScores[fatScores.length - 1] ?? null,
    separationMargin: pureScores.length && fatScores.length ? pureScores[0] - fatScores[fatScores.length - 1] : null,
    pureScores,
    fatScores,
  };
}

async function assertPreparedDataIsFresh() {
  let labelsStat;
  let manifestStat;
  try {
    labelsStat = await stat(LABELS_PATH);
  } catch (e) {
    throw new Error(`Cannot stat ${LABELS_PATH}: ${e.message}`);
  }
  try {
    manifestStat = await stat(PREPARED_MANIFEST_PATH);
  } catch (e) {
    throw new Error(`Prepared manifest missing at ${PREPARED_MANIFEST_PATH}. Run "npm run prepare:stage1b" first.`);
  }
  if (labelsStat.mtimeMs > manifestStat.mtimeMs) {
    throw new Error(
      `data/labels.json (${labelsStat.mtime.toISOString()}) is newer than ` +
      `data/stage1b_prepared/manifest.jsonl (${manifestStat.mtime.toISOString()}). ` +
      `Run "npm run prepare:stage1b" before retraining Stage 2 so cropped clips match current labels.`
    );
  }
}

async function loadRows() {
  await assertPreparedDataIsFresh();
  const exclusionPolicy = await loadStage2PureFatPolicy();
  const labelsDoc = JSON.parse(await readFile(LABELS_PATH, 'utf8'));
  const labelByRelPath = new Map();
  for (const entry of Object.values(labelsDoc.labels)) {
    const relPath = entry.path.replace(/^samples\//, '');
    labelByRelPath.set(relPath, entry);
  }

  const manifestRows = (await readFile(PREPARED_MANIFEST_PATH, 'utf8'))
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .filter(row => row.label === 'shot');

  const rows = [];
  const skipped = [];
  for (const manifest of manifestRows) {
    const labelEntry = labelByRelPath.get(manifest.source_path);
    const folderLabel = labelEntry?.folderLabel ?? null;
    const exclusion = exclusionForFolder(folderLabel, exclusionPolicy);
    const className = classFromFolder(folderLabel, exclusionPolicy);
    if (!className) {
      skipped.push({
        sourcePath: manifest.source_path,
        folderLabel,
        ...exclusion,
      });
      continue;
    }
    const clip = await readPreparedClip(manifest.local_path);
    rows.push({
      y: className === 'pure' ? 1 : 0,
      className,
      x: extractLogMelFeatures(clip),
      sourceId: manifest.source_id,
      groupId: manifest.group_id,
      sourcePath: manifest.source_path,
      localPath: manifest.local_path,
      folderLabel: labelEntry.folderLabel,
      centerSec: manifest.center_sec,
    });
  }
  return { rows, skipped, exclusionPolicy };
}

const { rows, skipped, exclusionPolicy } = await loadRows();
if (!rows.length) throw new Error('No pure/fat Stage 2 rows found. Run npm run train:stage1b first.');
if (rows[0].x.length !== LOG_MEL_FEATURE_NAMES.length) {
  throw new Error(`log-mel feature length mismatch: ${rows[0].x.length} != ${LOG_MEL_FEATURE_NAMES.length}`);
}

const folds = makeFolds(rows, 5);
const oof = [];
const foldReports = [];

for (let i = 0; i < folds.length; i++) {
  const fold = folds[i];
  const scaler = meanStd(rows, fold.train);
  const model = trainLogistic(rows, fold.train, scaler);
  const scored = fold.val.map(idx => ({ idx, y: rows[idx].y, pPure: predict(rows[idx], model, scaler) }));
  oof.push(...scored);
  foldReports.push({
    fold: i + 1,
    train: fold.train.length,
    val: fold.val.length,
    valPure: fold.summary.pure,
    valFat: fold.summary.fat,
    at_050: metrics(scored, 0.5),
    at_conf_060: confidenceMetrics(scored, 0.6),
  });
}

const allIndices = rows.map((_, i) => i);
const finalScaler = meanStd(rows, allIndices);
const finalModel = trainLogistic(rows, allIndices, finalScaler);
const allScored = rows.map((row, idx) => ({ idx, y: row.y, pPure: predict(row, finalModel, finalScaler) }));
const CONFIDENCE_THRESHOLD = 0.78;
const oofMetrics = metrics(oof, 0.5);
const oofConfidenceMetrics = confidenceMetrics(oof, CONFIDENCE_THRESHOLD);
const oofScoreSeparation = scoreSummary(oof);
const trainMetrics = metrics(allScored, 0.5);
const trainScoreSeparation = scoreSummary(allScored);

const nPure = rows.filter(r => r.y === 1).length;
const nFat = rows.filter(r => r.y === 0).length;
const generatedAt = new Date().toISOString();

const modelOut = {
  version: 1,
  type: 'standardized_logistic_regression',
  task: 'stage2_pure_fat_classifier',
  featureExtractor: 'logmel_summary',
  sampleRate: MODEL_SAMPLE_RATE,
  clipSamples: MODEL_CLIP_SAMPLES,
  classes: ['fat', 'pure'],
  positiveClass: 'pure',
  negativeClass: 'fat',
  confidenceThreshold: CONFIDENCE_THRESHOLD,
  features: LOG_MEL_FEATURE_NAMES,
  mean: finalScaler.mean,
  std: finalScaler.std,
  weights: finalModel.weights,
  bias: finalModel.bias,
  training: {
    generatedAt,
    pure: nPure,
    fat: nFat,
    excluded: skipped.length,
    excludedPolicy: 'manual bad-data exclusions plus topped and 1mm/borderline exclusions for pure-vs-fat v0',
    exclusionPolicy: summarizeStage2PureFatPolicy(exclusionPolicy),
    preparedManifest: PREPARED_MANIFEST_PATH.replace(ROOT + '/', ''),
    cv: oofMetrics,
    cvAtConfidence: oofConfidenceMetrics,
    oofScoreSeparation,
    trainOnlyMetrics: trainMetrics,
    trainScoreSeparation,
  },
};

const report = {
  generatedAt,
  modelPath: MODEL_PATH.replace(ROOT + '/', ''),
  featureExtractor: modelOut.featureExtractor,
  featureCount: LOG_MEL_FEATURE_NAMES.length,
  labelSource: 'data/labels.json folderLabel joined through prepared manifest source_path',
  exclusionPolicy: summarizeStage2PureFatPolicy(exclusionPolicy),
  pure: nPure,
  fat: nFat,
  excluded: skipped,
  cv: oofMetrics,
  cvAtConfidence: oofConfidenceMetrics,
  oofScoreSeparation,
  trainOnlyMetrics: trainMetrics,
  trainScoreSeparation,
  folds: foldReports,
  predictions: oof
    .sort((a, b) => rows[a.idx].sourcePath.localeCompare(rows[b.idx].sourcePath))
    .map(s => ({
      sourcePath: rows[s.idx].sourcePath,
      folderLabel: rows[s.idx].folderLabel,
      actual: rows[s.idx].className,
      pPure: s.pPure,
      predicted: s.pPure >= 0.5 ? 'pure' : 'fat',
      confidence: Math.max(s.pPure, 1 - s.pPure),
    })),
};

await mkdir(dirname(MODEL_PATH), { recursive: true });
await mkdir(dirname(REPORT_PATH), { recursive: true });
await writeFile(MODEL_PATH, JSON.stringify(modelOut, null, 2));
await writeFile(REPORT_PATH, JSON.stringify(report, null, 2));

console.log(`Stage 2 pure/fat trained: ${nPure} pure, ${nFat} fat, excluded ${skipped.length}`);
console.log(`CV accuracy ${oofMetrics.accuracy.toFixed(3)} pureRecall=${oofMetrics.pureRecall.toFixed(3)} fatRecall=${oofMetrics.fatRecall.toFixed(3)}`);
console.log(`CV @ confidence>=${CONFIDENCE_THRESHOLD.toFixed(2)} coverage=${oofConfidenceMetrics.coverage.toFixed(3)} unsure=${oofConfidenceMetrics.unsure} keptAccuracy=${oofConfidenceMetrics.kept.accuracy.toFixed(3)}`);
console.log(`OOF margin minPure=${oofScoreSeparation.minPureP?.toFixed(3) ?? 'n/a'} maxFat=${oofScoreSeparation.maxFatP?.toFixed(3) ?? 'n/a'} margin=${oofScoreSeparation.separationMargin?.toFixed(3) ?? 'n/a'}`);
console.log(`Wrote ${MODEL_PATH}`);
console.log(`Wrote ${REPORT_PATH}`);
