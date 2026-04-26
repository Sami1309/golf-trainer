import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MODEL_CLIP_SAMPLES,
  MODEL_SAMPLE_RATE,
  prepareModelClip,
} from '../frontend/audio_features.js';
import { magSpectrum, hann } from '../frontend/fft.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LABELS_PATH = join(ROOT, 'data', 'labels.json');
const EXTERNAL_MANIFEST_PATH = join(ROOT, 'data', 'external', 'manifest.jsonl');
const PREPARED_DIR = join(ROOT, 'data', 'stage1b_prepared');
const PREPARED_MANIFEST_PATH = join(PREPARED_DIR, 'manifest.jsonl');
const PREPARED_REPORT_PATH = join(PREPARED_DIR, 'prepare_report.json');
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

const report = {
  generatedAt: new Date().toISOString(),
  labelsPath: LABELS_PATH.replace(ROOT + '/', ''),
  preparedManifest: PREPARED_MANIFEST_PATH.replace(ROOT + '/', ''),
  positives: examples.filter(e => e.y === 1).length,
  negatives: examples.filter(e => e.y === 0).length,
  preparedCounts: countBy(preparedManifestRows, 'label'),
  negativeCategories: countBy(examples.filter(e => e.y === 0), 'category'),
  positiveSources: countBy(examples.filter(e => e.y === 1), 'sourceName'),
  negativeSources: countBy(examples.filter(e => e.y === 0), 'sourceName'),
  externalNegativeSourceFiles: externalNegativeRows.length,
  skippedExternalNegatives,
  config: {
    modelSampleRate: MODEL_SAMPLE_RATE,
    modelClipSamples: MODEL_CLIP_SAMPLES,
    clipPreSamples: CLIP_PRE_SAMPLES,
    clipPostSamples: CLIP_POST_SAMPLES,
    negativeMinPeakDbfs: NEGATIVE_MIN_PEAK_DBFS,
    externalNegativeOnsetThreshold: EXTERNAL_NEGATIVE_ONSET_THRESHOLD,
    externalNegativeMinGapSec: EXTERNAL_NEGATIVE_MIN_GAP_SEC,
    externalNegativesPerFile: EXTERNAL_NEGATIVES_PER_FILE,
    localPreShotNegativesPerFile: LOCAL_PRESHOT_NEGATIVES_PER_FILE,
    localPreShotOnsetThreshold: LOCAL_PRESHOT_ONSET_THRESHOLD,
    localPreShotImpactGapSec: LOCAL_PRESHOT_IMPACT_GAP_SEC,
  },
};

await writeFile(PREPARED_REPORT_PATH, JSON.stringify(report, null, 2));

console.log(`Stage 1b prepared: ${report.positives} positives, ${report.negatives} negatives`);
console.log(`Skipped external negatives: ${skippedExternalNegatives.length}`);
console.log(`Wrote ${PREPARED_MANIFEST_PATH}`);
console.log(`Wrote ${PREPARED_REPORT_PATH}`);
