import { readdir, readFile, rename, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { basename, dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { magSpectrum, hann } from '../frontend/fft.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_SOURCE_DIR = join(ROOT, 'new_data');
const LABELS_PATH = join(ROOT, 'data', 'labels.json');

const TARGET_SR = 16000;
const FFT_SIZE = 1024;
const HOP_SIZE = 256;
const MIN_GAP_SEC = 0.2;
const ONSET_THRESHOLD = 0.65;
const RECENTER_SEARCH_MS = 120;

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has('--dry-run');

function parseSourceDir() {
  const sourceArg = process.argv.find(arg => arg.startsWith('--source='));
  return sourceArg ? join(ROOT, sourceArg.slice('--source='.length)) : DEFAULT_SOURCE_DIR;
}

function decode(path) {
  return new Promise((resolve, reject) => {
    const ff = spawn('ffmpeg', [
      '-v', 'error',
      '-i', path,
      '-ac', '1',
      '-ar', String(TARGET_SR),
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

function computeFlux(samples) {
  const win = hann(FFT_SIZE);
  const magNorm = 2 / FFT_SIZE;
  const flux = [];
  const times = [];
  let prevMag = null;

  for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) frame[i] = samples[start + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;

    let value = 0;
    if (prevMag) {
      for (let k = 0; k < mag.length; k++) {
        const delta = mag[k] - prevMag[k];
        if (delta > 0) value += delta;
      }
    }

    flux.push(value);
    times.push((start + FFT_SIZE / 2) / TARGET_SR);
    prevMag = mag;
  }

  return { flux, times };
}

function pickPeaks(flux, times) {
  const peaks = [];
  let lastTime = -Infinity;

  for (let i = 1; i < flux.length - 1; i++) {
    if (flux[i] < ONSET_THRESHOLD) continue;
    if (flux[i] <= flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    if (times[i] - lastTime < MIN_GAP_SEC) continue;
    peaks.push({ time: times[i], flux: flux[i] });
    lastTime = times[i];
  }

  return peaks;
}

function recenterForward(samples, timeSec) {
  const start = Math.max(0, Math.floor(timeSec * TARGET_SR));
  const end = Math.min(samples.length, start + Math.floor(RECENTER_SEARCH_MS * TARGET_SR / 1000));
  let peakIdx = start;
  let peak = 0;

  for (let i = start; i < end; i++) {
    const value = Math.abs(samples[i]);
    if (value > peak) {
      peak = value;
      peakIdx = i;
    }
  }

  return { time: peakIdx / TARGET_SR, peak };
}

async function collectSourceRows(sourceDir) {
  const out = [];
  for (const entry of await readdir(sourceDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const folder = entry.name;
    const folderPath = join(sourceDir, folder);
    const files = await readdir(folderPath);
    const audioFiles = files.filter(file => extname(file).toLowerCase() === '.m4a');
    if (audioFiles.length !== 1) {
      throw new Error(`${folder}: expected exactly one .m4a, found ${audioFiles.length}`);
    }
    out.push({
      folder,
      audioFile: audioFiles[0],
      sourcePath: join(folderPath, audioFiles[0]),
      sourceFolderPath: folderPath,
      targetFolderPath: join(ROOT, folder),
    });
  }

  return out.sort((a, b) => a.folder.localeCompare(b.folder, undefined, { numeric: true }));
}

function keyFor(row, labels) {
  const preferred = row.audioFile;
  const existing = labels[preferred];
  if (!existing) return preferred;

  const rel = `${row.folder}/${row.audioFile}`;
  if (existing.path?.replace(/^samples\//, '') === rel) return preferred;

  throw new Error(`${preferred}: labels.json already has a different entry for this basename`);
}

const sourceDir = parseSourceDir();
const labelsDoc = JSON.parse(await readFile(LABELS_PATH, 'utf8'));
const rows = await collectSourceRows(sourceDir);

if (!rows.length) {
  console.log(`No importable folders found in ${sourceDir}`);
  process.exit(0);
}

const now = new Date().toISOString();
const imported = [];

for (const row of rows) {
  try {
    await readdir(row.targetFolderPath);
    throw new Error(`${basename(row.targetFolderPath)}: target folder already exists at repo root`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const samples = await decode(row.sourcePath);
  const { flux, times } = computeFlux(samples);
  const peaks = pickPeaks(flux, times).sort((a, b) => b.flux - a.flux);
  if (!peaks.length) {
    throw new Error(`${row.folder}: no onset peak above ${ONSET_THRESHOLD}`);
  }

  const candidates = peaks.slice(0, 5).map(peak => {
    const recentered = recenterForward(samples, peak.time);
    return {
      time: +recentered.time.toFixed(6),
      flux: +peak.flux.toFixed(6),
      peakAbs: +recentered.peak.toFixed(6),
    };
  });
  const selected = candidates[0];
  const labelKey = keyFor(row, labelsDoc.labels);

  imported.push({
    row,
    labelKey,
    label: {
      path: `samples/${row.folder}/${row.audioFile}`,
      shotTimes: [selected.time],
      folderLabel: row.folder,
      duration: +(samples.length / TARGET_SR).toFixed(6),
      labeledAt: now,
      labeledBy: 'codex-auto-import',
      labelingMethod: 'spectral_flux_recentered_peak_v1',
      importSource: `new_data/${row.folder}`,
      detector: {
        sampleRate: TARGET_SR,
        fftSize: FFT_SIZE,
        hopSize: HOP_SIZE,
        onsetThreshold: ONSET_THRESHOLD,
        minGapSec: MIN_GAP_SEC,
        recenterSearchMs: RECENTER_SEARCH_MS,
        selected,
        candidates,
      },
    },
  });
}

if (!DRY_RUN) {
  for (const item of imported) {
    labelsDoc.labels[item.labelKey] = item.label;
    await rename(item.row.sourceFolderPath, item.row.targetFolderPath);
  }
  await writeFile(LABELS_PATH, `${JSON.stringify(labelsDoc, null, 2)}\n`);
}

for (const item of imported) {
  const action = DRY_RUN ? 'Would import' : 'Imported';
  console.log(`${action} ${item.row.folder}/${item.row.audioFile} at ${item.label.shotTimes[0].toFixed(3)}s`);
}

console.log(`${DRY_RUN ? 'Dry-run complete' : 'Updated labels and moved'} ${imported.length} sample folders.`);
