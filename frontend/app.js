import { magSpectrum, hann } from './fft.js';
import { encodeWav, resample } from './wav.js';
import { loadStage1bModel, verifyShot } from './stage1b.js';
import {
  clearDetectionsStore,
  deleteDetection as deleteStoredDetection,
  getAllDetections,
  putDetection,
} from './shot_store.js';

// ------------------------------------------------------------
// Tunables (also reflected in UI controls)
// ------------------------------------------------------------
const TARGET_SR = 16000;          // sample rate we save clips at
const CLIP_PRE_MS = 100;
const CLIP_POST_MS = 400;
const CLIP_LEN_MS = CLIP_PRE_MS + CLIP_POST_MS;
const CONTEXT_PRE_MS = 1000;
const CONTEXT_POST_MS = 1000;
const CONTEXT_LEN_MS = CONTEXT_PRE_MS + CONTEXT_POST_MS;

// Spectral flux parameters for file-mode (offline) analysis
const FILE_FFT_SIZE = 1024;
const FILE_HOP_SIZE = 256;        // 75% overlap
const FILE_CALIBRATED_THRESHOLD = 0.65; // validated against all labeled sample files

// Live-mode uses the AnalyserNode at 60 Hz poll rate.
const LIVE_CALIBRATION_ARMED_THRESHOLD = 0.25;
const LIVE_CALIBRATION_THRESHOLD_FACTOR = 0.65;

const params = {
  threshold: 0.8,     // spectral flux threshold — tuned against the 28 m4a samples
  minGapMs: 200,      // min time between detected onsets
  stage1bThreshold: 0.7,
  showRejected: false,
  saveRejected: true,
  calibrationArmed: false,
  calibrationFlux: null,
};

const tester = new URLSearchParams(location.search).get('tester') || 'anon';
document.getElementById('tester-label').textContent = tester;

// ------------------------------------------------------------
// UI helpers
// ------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const el = (tag, props = {}, ...children) => {
  const n = document.createElement(tag);
  Object.assign(n, props);
  for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(c));
  return n;
};

function setStatus(msg, kind = 'info') {
  const s = $('#status');
  s.textContent = msg;
  s.className = `status status-${kind}`;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function makeId(prefix = 'det') {
  if (crypto.randomUUID) return `${prefix}_${crypto.randomUUID()}`;
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function nowStamp() {
  return new Date().toISOString();
}

function fileStamp() {
  return nowStamp().replace(/[:.]/g, '-');
}

function safeName(s) {
  return String(s || 'clip')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 96) || 'clip';
}

function samplesToWav(samples, sampleRate) {
  return encodeWav(samples, sampleRate);
}

async function blobToArrayBuffer(blob) {
  return blob.arrayBuffer();
}

function arrayBufferToWavBlob(buffer) {
  return new Blob([buffer], { type: 'audio/wav' });
}

function cropPadded(samples, centerSample, preSamples, postSamples) {
  const length = preSamples + postSamples;
  const out = new Float32Array(length);
  const srcStartWanted = centerSample - preSamples;
  const srcStart = Math.max(0, srcStartWanted);
  const dstStart = Math.max(0, -srcStartWanted);
  const copyLen = Math.min(samples.length - srcStart, length - dstStart);
  if (copyLen > 0) out.set(samples.subarray(srcStart, srcStart + copyLen), dstStart);
  return out;
}

function liveFluxThreshold() {
  return params.calibrationArmed
    ? Math.min(params.threshold, LIVE_CALIBRATION_ARMED_THRESHOLD)
    : params.threshold;
}

function updateThresholdUi() {
  $('#threshold').value = params.threshold;
  $('#threshold-val').textContent = params.threshold.toFixed(2);
}

function updateCalibrationUi() {
  const state = $('#calibration-state');
  const detail = $('#calibration-detail');
  const btn = $('#calibrate-next-shot');
  if (!state || !detail || !btn) return;

  if (params.calibrationArmed) {
    state.textContent = `armed @ ${LIVE_CALIBRATION_ARMED_THRESHOLD.toFixed(2)}`;
    detail.textContent = 'hit one real shot now';
    btn.textContent = 'Cancel calibration';
    return;
  }

  btn.textContent = 'Calibrate next shot';
  if (params.calibrationFlux == null) {
    state.textContent = 'not calibrated';
    detail.textContent = `default threshold ${params.threshold.toFixed(2)}`;
  } else {
    state.textContent = `shot strength ${params.calibrationFlux.toFixed(2)}`;
    detail.textContent = `threshold ${(params.calibrationFlux * LIVE_CALIBRATION_THRESHOLD_FACTOR).toFixed(2)}`;
  }
}

// ------------------------------------------------------------
// Waveform + flux strip rendering (live mode)
// ------------------------------------------------------------
class Strip {
  constructor(canvas, color, scale) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.color = color;
    this.scale = scale;               // pixels per unit
    this.history = [];                // ring of values
    this.max = canvas.width;
  }
  push(v) {
    this.history.push(v);
    if (this.history.length > this.max) this.history.shift();
  }
  draw(overlays = []) {
    const { ctx, canvas, color, scale } = this;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.strokeStyle = color;
    ctx.beginPath();
    const off = w - this.history.length;
    for (let i = 0; i < this.history.length; i++) {
      const y = h / 2 - this.history[i] * scale;
      if (i === 0) ctx.moveTo(off + i, y); else ctx.lineTo(off + i, y);
    }
    ctx.stroke();
    // threshold line (for flux strip)
    if (overlays.includes('threshold')) {
      ctx.strokeStyle = '#f55';
      ctx.setLineDash([4, 4]);
      const y = h / 2 - liveFluxThreshold() * scale;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

// ------------------------------------------------------------
// Live-mic pipeline
// ------------------------------------------------------------
const live = {
  ctx: null,
  stream: null,
  source: null,
  analyser: null,
  worklet: null,
  running: false,
  prevMag: null,
  lastOnsetSample: -Infinity,
  waveStrip: null,
  fluxStrip: null,
  workletTotalSamples: 0,
  pendingExtracts: new Map(),
  extractSeq: 0,
};

async function startLive() {
  if (live.running) return;
  setStatus('Requesting microphone…');
  try {
    live.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      }
    });
  } catch (e) {
    setStatus(`Mic denied: ${e.message}`, 'error');
    return;
  }
  live.ctx = new (window.AudioContext || window.webkitAudioContext)();
  await live.ctx.audioWorklet.addModule('./onset-worklet.js');
  live.source = live.ctx.createMediaStreamSource(live.stream);
  live.analyser = live.ctx.createAnalyser();
  live.analyser.fftSize = 2048;
  live.analyser.smoothingTimeConstant = 0;        // raw FFT — critical for transient flux
  live.worklet = new AudioWorkletNode(live.ctx, 'ring-buffer', {
    processorOptions: { seconds: 5 },
    numberOfInputs: 1,
    numberOfOutputs: 0,
  });
  live.worklet.port.onmessage = onWorkletMessage;
  live.source.connect(live.analyser);
  live.source.connect(live.worklet);
  // Note: do NOT connect anything to ctx.destination or we'll feed mic back.

  live.waveStrip = new Strip($('#live-wave'), '#6cf', 80);
  live.fluxStrip = new Strip($('#live-flux'), '#6f6', 30);
  live.prevMag = null;
  live.running = true;
  live.workletTotalSamples = 0;
  live.lastOnsetSample = -Infinity;
  live.peakFlux = 0;

  setStatus(`Listening (ctx ${live.ctx.sampleRate} Hz)`, 'ok');
  $('#live-start').disabled = true;
  $('#live-stop').disabled = false;
  $('#debug-ctx-sr').textContent = live.ctx.sampleRate;
  requestAnimationFrame(livePollTick);
}

function stopLive() {
  if (!live.running) return;
  live.running = false;
  try { live.worklet.disconnect(); } catch {}
  try { live.source.disconnect(); } catch {}
  try { live.stream.getTracks().forEach(t => t.stop()); } catch {}
  try { live.ctx.close(); } catch {}
  setStatus('Stopped');
  $('#live-start').disabled = false;
  $('#live-stop').disabled = true;
}

function onWorkletMessage(e) {
  const m = e.data;
  if (m.type === 'progress') {
    live.workletTotalSamples = m.totalSamples;
    return;
  }
  if (m.requestId !== undefined) {
    const p = live.pendingExtracts.get(m.requestId);
    if (!p) return;
    live.pendingExtracts.delete(m.requestId);
    if (m.error) p.reject(new Error(m.error));
    else p.resolve({ samples: m.samples, startSample: m.startSample });
  }
}

function requestExtract(sampleIndex, length) {
  return new Promise((resolve, reject) => {
    const requestId = ++live.extractSeq;
    live.pendingExtracts.set(requestId, { resolve, reject });
    live.worklet.port.postMessage({ cmd: 'extract', requestId, sampleIndex, length });
  });
}

async function waitForLiveSamples(endSample, timeoutMs = 1800) {
  const started = performance.now();
  while (live.running && live.workletTotalSamples < endSample) {
    if (performance.now() - started > timeoutMs) return false;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  return live.workletTotalSamples >= endSample;
}

async function requestExtractPadded(sampleIndex, length) {
  const out = new Float32Array(length);
  const oldest = Math.max(0, live.workletTotalSamples - Math.ceil(live.ctx.sampleRate * 5));
  const wantedEnd = sampleIndex + length;
  const requestStart = Math.max(oldest, sampleIndex);
  const requestEnd = Math.min(live.workletTotalSamples, wantedEnd);
  const requestLen = requestEnd - requestStart;
  if (requestLen <= 0) return { samples: out, startSample: sampleIndex };
  const { samples } = await requestExtract(requestStart, requestLen);
  out.set(samples, requestStart - sampleIndex);
  return { samples: out, startSample: sampleIndex };
}

// Poll AnalyserNode, compute spectral flux, peak-pick.
function livePollTick() {
  if (!live.running) return;
  requestAnimationFrame(livePollTick);

  const a = live.analyser;
  const nBins = a.frequencyBinCount;
  const mag = new Float32Array(nBins);
  a.getFloatFrequencyData(mag);      // dB scale
  // Convert to linear magnitudes (approximate)
  for (let i = 0; i < nBins; i++) {
    mag[i] = Math.pow(10, mag[i] / 20);
  }

  // Waveform sample
  const td = new Float32Array(a.fftSize);
  a.getFloatTimeDomainData(td);
  let peak = 0;
  for (let i = 0; i < td.length; i++) {
    const v = Math.abs(td[i]);
    if (v > peak) peak = v;
  }
  live.waveStrip.push(peak);

  // Spectral flux
  let flux = 0;
  if (live.prevMag) {
    for (let i = 0; i < nBins; i++) {
      const d = mag[i] - live.prevMag[i];
      if (d > 0) flux += d;
    }
  }
  live.prevMag = mag;

  live.fluxStrip.push(flux);
  $('#live-flux-now').textContent = flux.toFixed(2);
  if (flux > live.peakFlux) live.peakFlux = flux;
  $('#live-flux-peak').textContent = live.peakFlux.toFixed(2);

  // Peak pick: flux crosses threshold and enough time has passed
  const ctxTimeSample = Math.floor(live.ctx.currentTime * live.ctx.sampleRate);
  if (flux > liveFluxThreshold() &&
      (ctxTimeSample - live.lastOnsetSample) > (params.minGapMs / 1000) * live.ctx.sampleRate) {
    live.lastOnsetSample = ctxTimeSample;
    onLiveDetected(ctxTimeSample, flux);
  }

  // Redraw
  live.waveStrip.draw();
  live.fluxStrip.draw(['threshold']);
}

async function onLiveDetected(ctxTimeSample, flux) {
  // Extract a window wide enough to re-center on the peak-amplitude sample.
  // Plan: grab [flux_peak - PRE, flux_peak + POST + SEARCH], then within the post
  // region find the peak |amplitude| sample (empirically ~28ms after flux peak)
  // and re-crop to a canonical PRE/POST window centered on that sample.
  const sr = live.ctx.sampleRate;
  const preSamples = Math.floor((CLIP_PRE_MS / 1000) * sr);
  const postSamples = Math.floor((CLIP_POST_MS / 1000) * sr);
  const searchSamples = Math.floor(0.12 * sr);   // 120ms forward search
  const extraStart = preSamples;
  const extraLen = preSamples + postSamples + searchSamples;
  let startSample = ctxTimeSample - extraStart;
  if (startSample + extraLen > live.workletTotalSamples) {
    startSample = live.workletTotalSamples - extraLen;
  }
  try {
    const shortExtract = await requestExtractPadded(startSample, extraLen);
    const { samples } = shortExtract;
    // Find peak |amplitude| in [preSamples, preSamples + searchSamples)
    let peakIdx = preSamples, peakVal = 0;
    for (let i = preSamples; i < preSamples + searchSamples && i < samples.length; i++) {
      const v = Math.abs(samples[i]);
      if (v > peakVal) { peakVal = v; peakIdx = i; }
    }
    const impactSample = shortExtract.startSample + peakIdx;
    // Re-crop to [peakIdx - preSamples, peakIdx + postSamples]
    const cropStart = Math.max(0, peakIdx - preSamples);
    const cropEnd = Math.min(samples.length, peakIdx + postSamples);
    const cropped = samples.slice(cropStart, cropEnd);
    if (params.calibrationArmed) {
      applyLiveCalibration(flux);
    }
    const resampled = await resample(cropped, sr, TARGET_SR);
    const verification = verifyShot(resampled, { threshold: params.stage1bThreshold });
    const contextPreSamples = Math.floor((CONTEXT_PRE_MS / 1000) * sr);
    const contextPostSamples = Math.floor((CONTEXT_POST_MS / 1000) * sr);
    const contextStart = impactSample - contextPreSamples;
    const contextLen = contextPreSamples + contextPostSamples;
    await waitForLiveSamples(impactSample + contextPostSamples);
    const contextExtract = await requestExtractPadded(contextStart, contextLen);
    const contextResampled = await resample(contextExtract.samples, sr, TARGET_SR);
    const ts = fileStamp();
    void addDetection({
      source: 'live',
      timestamp: ts,
      flux: flux.toFixed(2),
      samples: verification.samples,
      contextSamples: contextResampled,
      sampleRate: TARGET_SR,
      contextSampleRate: TARGET_SR,
      verification,
      calibration: {
        onsetThreshold: params.threshold,
        calibrationFlux: params.calibrationFlux,
        stage1bThreshold: params.stage1bThreshold,
      },
    });
  } catch (e) {
    console.warn('extract failed', e);
  }
}

function applyLiveCalibration(flux) {
  params.calibrationArmed = false;
  params.calibrationFlux = flux;
  params.threshold = clamp(flux * LIVE_CALIBRATION_THRESHOLD_FACTOR, 0.1, 3);
  updateThresholdUi();
  updateCalibrationUi();
  setStatus(`Calibrated onset threshold to ${params.threshold.toFixed(2)} from shot strength ${flux.toFixed(2)}`, 'ok');
}

// ------------------------------------------------------------
// Detection list UI (shared live + file)
// ------------------------------------------------------------
const detections = [];

function defaultDetectionLabel(det) {
  return det.verification?.label === 'not_shot' ? 'no_shot' : 'unsure';
}

function detectionFilename(det, kind = 'context') {
  const label = det.label || 'unlabeled';
  const source = safeName(det.source);
  const stamp = safeName(det.timestamp || det.createdAt || 'time');
  return `${tester}_${String(det.displayId || det.id).padStart(3, '0')}_${source}_${stamp}_${label}_${kind}.wav`;
}

async function materializeDetection(det) {
  if (!det.wavBuffer && det.samples) {
    det.wavBuffer = await blobToArrayBuffer(samplesToWav(det.samples, det.sampleRate));
  }
  if (!det.contextWavBuffer) {
    const contextSamples = det.contextSamples || det.samples;
    const contextRate = det.contextSampleRate || det.sampleRate;
    det.contextWavBuffer = await blobToArrayBuffer(samplesToWav(contextSamples, contextRate));
  }
  delete det.samples;
  delete det.contextSamples;
  return det;
}

function ensureDetectionUrls(det) {
  if (!det.wavUrl && det.wavBuffer) {
    det.wavUrl = URL.createObjectURL(arrayBufferToWavBlob(det.wavBuffer));
  }
  if (!det.contextUrl && det.contextWavBuffer) {
    det.contextUrl = URL.createObjectURL(arrayBufferToWavBlob(det.contextWavBuffer));
  }
}

function revokeDetectionUrls(det) {
  if (det.wavUrl) URL.revokeObjectURL(det.wavUrl);
  if (det.contextUrl) URL.revokeObjectURL(det.contextUrl);
  det.wavUrl = null;
  det.contextUrl = null;
}

async function addDetection(det) {
  det.id = det.id || makeId('det');
  det.createdAt = det.createdAt || nowStamp();
  det.verification ||= { available: false, label: 'shot', pShot: 1, confidence: 1 };
  if (det.verification.samples) delete det.verification.samples;
  det.label = det.label || defaultDetectionLabel(det);
  det.tester = tester;
  det.contextMs = CONTEXT_LEN_MS;
  det.modelClipMs = CLIP_LEN_MS;
  await materializeDetection(det);
  ensureDetectionUrls(det);
  detections.push(det);
  sortDetections();
  renderDetections();
  try {
    await putDetection(stripRuntimeDetectionFields(det));
  } catch (e) {
    console.warn('failed to store detection', e);
    setStatus(`Detection captured but not stored: ${e.message}`, 'error');
  }
}

function stripRuntimeDetectionFields(det) {
  const { wavUrl, contextUrl, displayId, ...record } = det;
  return record;
}

function sortDetections() {
  detections.sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''));
  detections.forEach((d, i) => { d.displayId = i + 1; });
}

async function loadStoredDetections() {
  try {
    const stored = await getAllDetections();
    detections.length = 0;
    for (const det of stored) {
      ensureDetectionUrls(det);
      detections.push(det);
    }
    sortDetections();
    renderDetections();
  } catch (e) {
    console.warn('failed to load stored detections', e);
    setStatus(`Stored detection load failed: ${e.message}`, 'error');
  }
}

function renderDetections() {
  $('#detection-rows').innerHTML = '';
  for (const det of detections) renderDetectionRow(det);
  const accepted = detections.filter(d => d.verification.label === 'shot').length;
  const labeled = detections.filter(d => d.label && d.label !== 'unsure').length;
  $('#detection-count').textContent = `${accepted}/${detections.length}`;
  $('#stored-count').textContent = `${detections.length} stored · ${labeled} labeled`;
}

function renderDetectionRow(det) {
  const rejected = det.verification.label === 'not_shot';

  const audio = el('audio', { controls: true, src: det.contextUrl || det.wavUrl, preload: 'none' });
  audio.style.height = '28px';

  const labelBtn = (lbl) => {
    const b = el('button', { className: `lbl ${det.label === lbl ? 'active' : ''}` }, lbl);
    b.onclick = async () => {
      det.label = lbl;
      det.labeledAt = nowStamp();
      det.labeledBy = tester;
      await putDetection(stripRuntimeDetectionFields(det));
      renderDetections();
    };
    return b;
  };

  const contextDl = el('a', {
    href: det.contextUrl || det.wavUrl,
    download: detectionFilename(det, 'context_2s'),
    className: 'dl',
    textContent: 'ctx',
  });
  const modelDl = el('a', {
    href: det.wavUrl,
    download: detectionFilename(det, 'model_500ms'),
    className: 'dl',
    textContent: '500ms',
  });
  const del = el('button', { className: 'mini danger', textContent: 'del' });
  del.onclick = async () => {
    if (!confirm(`Delete detection ${det.displayId}?`)) return;
    revokeDetectionUrls(det);
    await deleteStoredDetection(det.id);
    const idx = detections.findIndex(d => d.id === det.id);
    if (idx >= 0) detections.splice(idx, 1);
    sortDetections();
    renderDetections();
  };

  const verifierText = det.verification.available
    ? `${det.verification.label} ${(det.verification.pShot * 100).toFixed(0)}%`
    : 'onset-only';
  const verifierClass = det.verification.label === 'shot' ? 'verifier-ok' : 'verifier-reject';

  const row = el('tr', { className: rejected ? 'rejected' : '' },
    el('td', {}, String(det.displayId)),
    el('td', {}, det.source),
    el('td', {}, det.timestamp),
    el('td', {}, det.flux),
    el('td', { className: verifierClass }, verifierText),
    el('td', {}, audio),
    el('td', {},
      labelBtn('pure'),
      labelBtn('fat'),
      labelBtn('topped'),
      labelBtn('no_shot'),
      labelBtn('unsure'),
    ),
    el('td', {}, contextDl, ' ', modelDl, ' ', del),
  );
  if (rejected && !params.showRejected) row.classList.add('hidden-rejected');
  $('#detection-rows').appendChild(row);
}

async function exportAll() {
  if (detections.length === 0) {
    alert('No detections yet.');
    return;
  }
  if (!window.JSZip) {
    setStatus('ZIP library unavailable; exporting manifest only.', 'error');
    exportManifestOnly();
    return;
  }

  setStatus(`Building ZIP for ${detections.length} detections…`, 'info');
  const zip = new window.JSZip();
  const manifest = detections.map(d => manifestRecord(d));
  zip.file('manifest.json', JSON.stringify({
    version: 1,
    exportedAt: nowStamp(),
    tester,
    app: 'golf-shot-detector',
    detections: manifest,
  }, null, 2));

  for (const det of detections) {
    const contextName = detectionFilename(det, 'context_2s');
    const modelName = detectionFilename(det, 'model_500ms');
    zip.file(`clips/context/${contextName}`, det.contextWavBuffer || det.wavBuffer);
    zip.file(`clips/model_500ms/${modelName}`, det.wavBuffer);
  }

  const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${tester}_detections_${fileStamp()}.zip`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  setStatus(`Exported ${detections.length} detections.`, 'ok');
}

function manifestRecord(d) {
  return {
    id: d.id,
    displayId: d.displayId,
    createdAt: d.createdAt,
    source: d.source,
    timestamp: d.timestamp,
    flux: d.flux,
    label: d.label,
    labeledAt: d.labeledAt || null,
    labeledBy: d.labeledBy || null,
    tester: d.tester || tester,
    verifier: d.verification,
    calibration: d.calibration || null,
    sampleRate: d.sampleRate,
    contextSampleRate: d.contextSampleRate || d.sampleRate,
    modelClipMs: d.modelClipMs || CLIP_LEN_MS,
    contextMs: d.contextMs || CONTEXT_LEN_MS,
    contextFile: `clips/context/${detectionFilename(d, 'context_2s')}`,
    modelClipFile: `clips/model_500ms/${detectionFilename(d, 'model_500ms')}`,
  };
}

function exportManifestOnly() {
  const blob = new Blob([JSON.stringify(detections.map(d => manifestRecord(d)), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${tester}_labels_${fileStamp()}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function clearDetections() {
  if (!confirm(`Clear all ${detections.length} detections?`)) return;
  for (const det of detections) revokeDetectionUrls(det);
  detections.length = 0;
  await clearDetectionsStore();
  $('#detection-rows').innerHTML = '';
  $('#detection-count').textContent = '0';
  $('#stored-count').textContent = '0 stored · 0 labeled';
}

// ------------------------------------------------------------
// File-mode analysis
// ------------------------------------------------------------
async function analyzeFile(file) {
  setStatus(`Decoding ${file.name}…`);
  const arrayBuf = await file.arrayBuffer();
  const tmpCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(
    1, 1, 16000  // dummy, just for decode
  );
  // Actually use a regular AudioContext for decode — OfflineAudioContext decode is quirky
  const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  let audioBuf;
  try {
    audioBuf = await decodeCtx.decodeAudioData(arrayBuf.slice(0));
  } catch (e) {
    setStatus(`Decode failed: ${e.message}`, 'error');
    return;
  } finally {
    decodeCtx.close();
  }
  const srcRate = audioBuf.sampleRate;
  // Mix to mono
  let mono;
  if (audioBuf.numberOfChannels === 1) {
    mono = audioBuf.getChannelData(0).slice();
  } else {
    const L = audioBuf.getChannelData(0), R = audioBuf.getChannelData(1);
    mono = new Float32Array(audioBuf.length);
    for (let i = 0; i < mono.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
  }
  // Resample to target rate for detection (makes flux threshold comparable to live mode)
  const sig = await resample(mono, srcRate, TARGET_SR);
  setStatus(`Analyzing ${file.name} (${(sig.length/TARGET_SR).toFixed(2)}s, ${TARGET_SR} Hz)…`);

  const { fluxCurve, frameTimes } = computeFlux(sig, TARGET_SR, FILE_FFT_SIZE, FILE_HOP_SIZE);
  const onsetsRaw = pickPeaks(fluxCurve, frameTimes, FILE_CALIBRATED_THRESHOLD, params.minGapMs / 1000);
  // Re-center each onset on peak amplitude — aligns with label convention.
  const onsets = onsetsRaw.map(o => ({ time: recenterOnsetTime(sig, o.time), flux: o.flux }));

  renderFileAnalysis(file.name, sig, TARGET_SR, fluxCurve, frameTimes, onsets, FILE_CALIBRATED_THRESHOLD);
  const maxFlux = fluxCurve.reduce((a, b) => Math.max(a, b), 0);
  setStatus(`${file.name}: ${onsets.length} onset(s), max flux ${maxFlux.toFixed(2)} (file threshold ${FILE_CALIBRATED_THRESHOLD})`, 'ok');

  // Extract clips
  for (const o of onsets) {
    // o.time is already peak-amplitude re-centered above.
    const centerSample = Math.floor(o.time * TARGET_SR);
    const preSamples = Math.floor((CLIP_PRE_MS / 1000) * TARGET_SR);
    const postSamples = Math.floor((CLIP_POST_MS / 1000) * TARGET_SR);
    let start = centerSample - preSamples;
    let end = centerSample + postSamples;
    if (start < 0) { end += -start; start = 0; }
    if (end > sig.length) { start -= end - sig.length; end = sig.length; start = Math.max(0, start); }
    const clip = sig.slice(start, end);
    const contextClip = cropPadded(
      sig,
      centerSample,
      Math.floor((CONTEXT_PRE_MS / 1000) * TARGET_SR),
      Math.floor((CONTEXT_POST_MS / 1000) * TARGET_SR)
    );
    const verification = verifyShot(clip, { threshold: params.stage1bThreshold });
    void addDetection({
      source: `file:${file.name}`,
      timestamp: `t=${o.time.toFixed(3)}s`,
      flux: o.flux.toFixed(2),
      samples: verification.samples,
      contextSamples: contextClip,
      sampleRate: TARGET_SR,
      contextSampleRate: TARGET_SR,
      verification,
    });
  }
}

function computeFlux(sig, sr, fftSize, hop) {
  const win = hann(fftSize);
  const frameTimes = [];
  const fluxCurve = [];
  let prevMag = null;
  // Normalize FFT magnitudes so that a full-scale signal yields values in [0, 1]
  // range — matches the linear-magnitude scale used in live mode (which converts
  // AnalyserNode dB output to linear).
  const magNorm = 2 / fftSize;
  for (let start = 0; start + fftSize <= sig.length; start += hop) {
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = sig[start + i] * win[i];
    const mag = magSpectrum(frame);
    for (let k = 0; k < mag.length; k++) mag[k] *= magNorm;
    let f = 0;
    if (prevMag) {
      for (let k = 0; k < mag.length; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) f += d;
      }
    }
    fluxCurve.push(f);
    frameTimes.push((start + fftSize / 2) / sr);
    prevMag = mag;
  }
  return { fluxCurve, frameTimes };
}

// Re-center an onset on the sample with max |amplitude| in a forward search window.
// Aligns detection times with perceptual impact rather than the spectral onset rising edge.
// Caller passes the signal the flux came from (16 kHz mono).
function recenterOnsetTime(sig, tSec, searchMs = 120, sr = TARGET_SR) {
  const startSample = Math.max(0, Math.floor(tSec * sr));
  const endSample = Math.min(sig.length, startSample + Math.floor(searchMs * sr / 1000));
  let peakIdx = startSample, peakVal = 0;
  for (let i = startSample; i < endSample; i++) {
    const v = Math.abs(sig[i]);
    if (v > peakVal) { peakVal = v; peakIdx = i; }
  }
  return peakIdx / sr;
}

function pickPeaks(flux, times, threshold, minGapSec) {
  const onsets = [];
  let lastTime = -Infinity;
  for (let i = 1; i < flux.length - 1; i++) {
    if (flux[i] < threshold) continue;
    if (flux[i] <= flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    if (times[i] - lastTime < minGapSec) continue;
    onsets.push({ time: times[i], flux: flux[i] });
    lastTime = times[i];
  }
  return onsets;
}

function renderFileAnalysis(name, sig, sr, flux, times, onsets, thresholdUsed = FILE_CALIBRATED_THRESHOLD) {
  const card = $('#file-result');
  card.innerHTML = '';
  card.append(el('h3', {}, `${name} — ${(sig.length/sr).toFixed(2)}s — ${onsets.length} detections — threshold ${thresholdUsed}`));

  const wCanvas = el('canvas', { width: 900, height: 120, className: 'wave-canvas' });
  const fCanvas = el('canvas', { width: 900, height: 80, className: 'wave-canvas' });
  card.append(wCanvas, fCanvas);

  // Waveform with onset lines
  const wctx = wCanvas.getContext('2d');
  wctx.fillStyle = '#111';
  wctx.fillRect(0, 0, wCanvas.width, wCanvas.height);
  wctx.strokeStyle = '#6cf';
  wctx.beginPath();
  const samplesPerPx = Math.max(1, Math.floor(sig.length / wCanvas.width));
  for (let x = 0; x < wCanvas.width; x++) {
    let mn = 0, mx = 0;
    const s0 = x * samplesPerPx;
    const s1 = Math.min(sig.length, s0 + samplesPerPx);
    for (let i = s0; i < s1; i++) {
      if (sig[i] < mn) mn = sig[i];
      if (sig[i] > mx) mx = sig[i];
    }
    const midY = wCanvas.height / 2;
    wctx.moveTo(x, midY - mx * (wCanvas.height / 2));
    wctx.lineTo(x, midY - mn * (wCanvas.height / 2));
  }
  wctx.stroke();
  // Onset overlays
  wctx.strokeStyle = '#f5a';
  for (const o of onsets) {
    const x = Math.floor((o.time * sr) / samplesPerPx);
    wctx.beginPath(); wctx.moveTo(x, 0); wctx.lineTo(x, wCanvas.height); wctx.stroke();
  }

  // Flux curve with threshold
  const fctx = fCanvas.getContext('2d');
  fctx.fillStyle = '#111';
  fctx.fillRect(0, 0, fCanvas.width, fCanvas.height);
  const fluxMax = Math.max(thresholdUsed * 2, ...flux);
  fctx.strokeStyle = '#6f6';
  fctx.beginPath();
  for (let i = 0; i < flux.length; i++) {
    const x = (i / flux.length) * fCanvas.width;
    const y = fCanvas.height - (flux[i] / fluxMax) * fCanvas.height;
    if (i === 0) fctx.moveTo(x, y); else fctx.lineTo(x, y);
  }
  fctx.stroke();
  // threshold
  fctx.strokeStyle = '#f55';
  fctx.setLineDash([4, 4]);
  const thY = fCanvas.height - (thresholdUsed / fluxMax) * fCanvas.height;
  fctx.beginPath(); fctx.moveTo(0, thY); fctx.lineTo(fCanvas.width, thY); fctx.stroke();
  fctx.setLineDash([]);
}

// ------------------------------------------------------------
// Wire up UI
// ------------------------------------------------------------
$('#live-start').onclick = startLive;
$('#live-stop').onclick = stopLive;
$('#export-all').onclick = exportAll;
$('#clear-detections').onclick = clearDetections;
$('#calibrate-next-shot').onclick = () => {
  params.calibrationArmed = !params.calibrationArmed;
  updateCalibrationUi();
  if (params.calibrationArmed) {
    setStatus('Calibration armed: hit one real shot. Only onset strength will be calibrated.', 'info');
  } else {
    setStatus('Calibration cancelled.', 'info');
  }
};
$('#threshold').oninput = (e) => {
  params.threshold = parseFloat(e.target.value);
  params.calibrationArmed = false;
  updateThresholdUi();
  updateCalibrationUi();
};
$('#min-gap').oninput = (e) => {
  params.minGapMs = parseInt(e.target.value);
  $('#min-gap-val').textContent = params.minGapMs;
};
$('#stage1b-threshold').oninput = (e) => {
  params.stage1bThreshold = parseFloat(e.target.value);
  $('#stage1b-threshold-val').textContent = params.stage1bThreshold.toFixed(2);
};
$('#show-rejected').onchange = (e) => {
  params.showRejected = e.target.checked;
  renderDetections();
};
$('#file-input').onchange = async (e) => {
  const files = Array.from(e.target.files || []);
  for (const f of files) await analyzeFile(f);
  e.target.value = '';   // allow re-select same file
};

// Show initial param values
updateThresholdUi();
$('#min-gap').value = params.minGapMs;
$('#min-gap-val').textContent = params.minGapMs;
$('#stage1b-threshold').value = params.stage1bThreshold;
$('#stage1b-threshold-val').textContent = params.stage1bThreshold.toFixed(2);
$('#show-rejected').checked = params.showRejected;
updateCalibrationUi();

async function initStage1b() {
  try {
    const model = await loadStage1bModel();
    params.stage1bThreshold = model.threshold ?? params.stage1bThreshold;
    $('#stage1b-threshold').value = params.stage1bThreshold;
    $('#stage1b-threshold-val').textContent = params.stage1bThreshold.toFixed(2);
    $('#stage1b-status').textContent = `${model.training?.positives ?? '?'}+/${model.training?.negatives ?? '?'}-`;
  } catch (e) {
    $('#stage1b-status').textContent = 'off';
    console.warn(e);
  }
}

initStage1b();
loadStoredDetections();

// Debug info
$('#debug-ua').textContent = navigator.userAgent;

// ------------------------------------------------------------
// Labeling — ground-truth shot-onset times per raw recording
// ------------------------------------------------------------
const LABELS_KEY = 'golf-shot-labels-v1';
const labelingState = {
  files: [],
  activeIndex: -1,
  audio: null,
  decodedBuffer: null,
  duration: 0,
  animId: 0,
};

function loadLabels() {
  try {
    const raw = localStorage.getItem(LABELS_KEY);
    if (!raw) return { version: 1, labels: {} };
    const parsed = JSON.parse(raw);
    if (!parsed.labels) return { version: 1, labels: {} };
    return parsed;
  } catch { return { version: 1, labels: {} }; }
}
function saveLabels() { localStorage.setItem(LABELS_KEY, JSON.stringify(labelsStore)); }
let labelsStore = loadLabels();

function handleFolderSelection(fileList) {
  const audio = Array.from(fileList).filter(f => /\.(m4a|wav|mp3|aac)$/i.test(f.name));
  audio.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  labelingState.files = audio;
  $('#files-count').textContent = audio.length;
  renderFileList();
  updateLabelsCount();
  if (audio.length > 0) selectFile(0);
}

function renderFileList() {
  const list = $('#file-list');
  list.innerHTML = '';
  if (labelingState.files.length === 0) {
    list.innerHTML = '<div class="file-list-empty">No folder selected.</div>';
    return;
  }
  labelingState.files.forEach((f, i) => {
    const folder = (f.webkitRelativePath || f.name).split('/').slice(-2, -1)[0] || '';
    const lbl = labelsStore.labels[f.name];
    const labeled = lbl && lbl.shotTimes && lbl.shotTimes.length > 0;
    const item = el('div', { className: 'file-item' + (labeled ? ' labeled' : '') + (i === labelingState.activeIndex ? ' active' : '') },
      el('div', { style: 'min-width:0; flex:1' },
        el('div', { style: 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }, f.name),
        el('div', { className: 'folder-tag' }, folder),
      ),
      el('span', { className: 'dot' }),
    );
    item.onclick = () => selectFile(i);
    list.appendChild(item);
  });
}

function updateLabelsCount() {
  const n = Object.values(labelsStore.labels).filter(l => l.shotTimes && l.shotTimes.length > 0).length;
  $('#labels-count').textContent = n;
  $('#calibrate-run').disabled = n === 0;
}

async function selectFile(index) {
  if (index < 0 || index >= labelingState.files.length) return;
  labelingState.activeIndex = index;
  renderFileList();
  const file = labelingState.files[index];
  $('#labeling-filename').textContent = file.name;
  const folderPath = (file.webkitRelativePath || '').split('/').slice(0, -1).join('/');
  $('#labeling-folder-label').textContent = folderPath;

  if (labelingState.audio) {
    labelingState.audio.pause();
    if (labelingState.audio.src) URL.revokeObjectURL(labelingState.audio.src);
  }
  const url = URL.createObjectURL(file);
  const audio = new Audio(url);
  audio.preload = 'auto';
  labelingState.audio = audio;
  audio.addEventListener('loadedmetadata', () => {
    labelingState.duration = audio.duration;
    $('#labeling-duration').textContent = audio.duration.toFixed(3);
    $('#labeling-play').disabled = false;
    $('#labeling-mark').disabled = false;
    $('#labeling-clear-one').disabled = false;
  });

  labelingState.decodedBuffer = null;
  try {
    const arrayBuf = await file.arrayBuffer();
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buffer = await decodeCtx.decodeAudioData(arrayBuf);
    decodeCtx.close();
    labelingState.decodedBuffer = buffer;
  } catch (e) {
    setStatus(`Decode failed for ${file.name}: ${e.message}`, 'error');
  }

  if (labelingState.animId) cancelAnimationFrame(labelingState.animId);
  const animate = () => {
    drawLabelingWaveform();
    labelingState.animId = requestAnimationFrame(animate);
  };
  animate();
}

function getChannelMono(buffer) {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice();
  const L = buffer.getChannelData(0), R = buffer.getChannelData(1);
  const mono = new Float32Array(buffer.length);
  for (let i = 0; i < mono.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
  return mono;
}

function drawLabelingWaveform() {
  const canvas = $('#labeling-wave');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#0a0a0c';
  ctx.fillRect(0, 0, w, h);

  if (labelingState.decodedBuffer) {
    const sig = getChannelMono(labelingState.decodedBuffer);
    const spp = Math.max(1, Math.floor(sig.length / w));
    ctx.strokeStyle = '#6cf';
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      let mn = 0, mx = 0;
      const s0 = x * spp, s1 = Math.min(sig.length, s0 + spp);
      for (let i = s0; i < s1; i++) {
        if (sig[i] < mn) mn = sig[i];
        if (sig[i] > mx) mx = sig[i];
      }
      const midY = h / 2;
      ctx.moveTo(x, midY - mx * (h / 2));
      ctx.lineTo(x, midY - mn * (h / 2));
    }
    ctx.stroke();
  }

  const file = labelingState.files[labelingState.activeIndex];
  const lbl = file && labelsStore.labels[file.name];
  const shotTimes = (lbl && lbl.shotTimes) || [];
  for (const t of shotTimes) {
    const x = (t / (labelingState.duration || 1)) * w;
    ctx.strokeStyle = '#f5a';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = '#f5a';
    ctx.fillRect(x - 6, 0, 12, 5);
    ctx.lineWidth = 1;
  }
  $('#labeling-shot-time').textContent = shotTimes.length === 0
    ? '—' : shotTimes.map(t => t.toFixed(3)).join(', ');

  if (labelingState.audio) {
    const t = labelingState.audio.currentTime;
    const x = (t / (labelingState.duration || 1)) * w;
    ctx.strokeStyle = '#fff';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    $('#labeling-time').textContent = t.toFixed(3);
  }
}

function markShotAt(t) {
  const file = labelingState.files[labelingState.activeIndex];
  if (!file) return;
  const folderLabel = (file.webkitRelativePath || '').split('/').slice(-2, -1)[0] || '';
  labelsStore.labels[file.name] = {
    path: file.webkitRelativePath || file.name,
    shotTimes: [t],
    folderLabel,
    duration: labelingState.duration,
    labeledAt: new Date().toISOString(),
    labeledBy: tester,
  };
  saveLabels();
  renderFileList();
  updateLabelsCount();
}

function markShotAtPlayhead() {
  if (labelingState.audio) markShotAt(labelingState.audio.currentTime);
}

function clearCurrentLabel() {
  const file = labelingState.files[labelingState.activeIndex];
  if (!file) return;
  delete labelsStore.labels[file.name];
  saveLabels();
  renderFileList();
  updateLabelsCount();
}

function togglePlay() {
  if (!labelingState.audio) return;
  if (labelingState.audio.paused) labelingState.audio.play();
  else labelingState.audio.pause();
}

function advanceFile(delta) {
  const next = labelingState.activeIndex + delta;
  if (next >= 0 && next < labelingState.files.length) selectFile(next);
}

function exportLabels() {
  const blob = new Blob([JSON.stringify(labelsStore, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `labels_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function importLabels(file) {
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (!parsed.labels) throw new Error('missing "labels" field');
    Object.assign(labelsStore.labels, parsed.labels);
    saveLabels();
    renderFileList();
    updateLabelsCount();
    setStatus(`Imported ${Object.keys(parsed.labels).length} labels`, 'ok');
  } catch (e) {
    setStatus(`Import failed: ${e.message}`, 'error');
  }
}

function clearAllLabels() {
  const n = Object.keys(labelsStore.labels).length;
  if (n === 0 || !confirm(`Clear ALL ${n} labels?`)) return;
  labelsStore = { version: 1, labels: {} };
  saveLabels();
  renderFileList();
  updateLabelsCount();
}

$('#labeling-folder').addEventListener('change', (e) => handleFolderSelection(e.target.files));
$('#labeling-play').addEventListener('click', togglePlay);
$('#labeling-mark').addEventListener('click', markShotAtPlayhead);
$('#labeling-clear-one').addEventListener('click', clearCurrentLabel);
$('#labels-export').addEventListener('click', exportLabels);
$('#labels-import').addEventListener('click', () => $('#labels-import-input').click());
$('#labels-import-input').addEventListener('change', (e) => {
  if (e.target.files[0]) importLabels(e.target.files[0]);
  e.target.value = '';
});
$('#labels-clear').addEventListener('click', clearAllLabels);

$('#labeling-wave').addEventListener('click', (ev) => {
  if (!labelingState.audio || !labelingState.duration) return;
  const canvas = ev.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const t = x * labelingState.duration;
  if (ev.shiftKey) markShotAt(t);
  else labelingState.audio.currentTime = t;
});

window.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) return;
  if (labelingState.files.length === 0) return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'l' || e.key === 'L') { e.preventDefault(); markShotAtPlayhead(); }
  else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); advanceFile(+1); }
  else if (e.key === 'p' || e.key === 'P') { e.preventDefault(); advanceFile(-1); }
});

updateLabelsCount();

// ------------------------------------------------------------
// Calibration — sweep thresholds over labeled files to find best detector params
// ------------------------------------------------------------
let lastCalibration = null;

async function runCalibration() {
  const toleranceMs = parseInt($('#calibrate-tolerance').value);
  const toleranceSec = toleranceMs / 1000;
  const resultsDiv = $('#calibration-results');
  resultsDiv.innerHTML = '<div class="hint">Running… this takes a moment per file.</div>';

  const files = labelingState.files.filter(f => {
    const l = labelsStore.labels[f.name];
    return l && l.shotTimes && l.shotTimes.length > 0;
  });
  if (files.length === 0) {
    resultsDiv.innerHTML = '<div class="hint status-error">No labeled files. Label some first.</div>';
    return;
  }

  const perFile = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    resultsDiv.innerHTML = `<div class="hint">Analyzing ${i + 1}/${files.length}: ${file.name}</div>`;
    try {
      const arrayBuf = await file.arrayBuffer();
      const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
      const buffer = await decodeCtx.decodeAudioData(arrayBuf);
      decodeCtx.close();
      const mono = getChannelMono(buffer);
      const sig = await resample(mono, buffer.sampleRate, TARGET_SR);
      const { fluxCurve, frameTimes } = computeFlux(sig, TARGET_SR, FILE_FFT_SIZE, FILE_HOP_SIZE);
      perFile.push({
        name: file.name,
        duration: buffer.duration,
        groundTruth: labelsStore.labels[file.name].shotTimes.slice(),
        fluxCurve, frameTimes, sig,
      });
    } catch (e) {
      console.warn(`skipping ${file.name}: ${e.message}`);
    }
  }

  const thresholds = [];
  for (let t = 0.05; t <= 3.0; t += 0.05) thresholds.push(+t.toFixed(2));

  const sweep = thresholds.map(threshold => {
    let tp = 0, fp = 0, fn = 0;
    const perFileDetections = [];
    for (const f of perFile) {
      const onsetsRaw = pickPeaks(f.fluxCurve, f.frameTimes, threshold, params.minGapMs / 1000);
      const onsets = onsetsRaw.map(o => ({ time: recenterOnsetTime(f.sig, o.time), flux: o.flux }));
      const gtMatched = new Array(f.groundTruth.length).fill(false);
      const detMatched = new Array(onsets.length).fill(false);
      for (let gi = 0; gi < f.groundTruth.length; gi++) {
        let bestIdx = -1, bestDt = Infinity;
        for (let di = 0; di < onsets.length; di++) {
          if (detMatched[di]) continue;
          const dt = Math.abs(onsets[di].time - f.groundTruth[gi]);
          if (dt < bestDt) { bestDt = dt; bestIdx = di; }
        }
        if (bestIdx >= 0 && bestDt <= toleranceSec) {
          gtMatched[gi] = true;
          detMatched[bestIdx] = true;
        }
      }
      const fileTP = gtMatched.filter(Boolean).length;
      const fileFN = gtMatched.filter(m => !m).length;
      const fileFP = detMatched.filter(m => !m).length;
      tp += fileTP; fp += fileFP; fn += fileFN;
      perFileDetections.push({ name: f.name, gt: f.groundTruth, onsets, gtMatched, detMatched,
                               fileTP, fileFP, fileFN });
    }
    const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);
    const recall    = (tp + fn) === 0 ? 0 : tp / (tp + fn);
    const f1        = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);
    return { threshold, tp, fp, fn, precision, recall, f1, perFileDetections };
  });

  const bestF1 = sweep.reduce((a, b) => b.f1 > a.f1 ? b : a);
  const perfect = sweep.filter(s => s.recall >= 0.999);
  const bestPerfectRecall = perfect.length
    ? perfect.reduce((a, b) => b.threshold > a.threshold ? b : a) : null;

  lastCalibration = { sweep, bestF1, bestPerfectRecall, toleranceMs };
  renderCalibration();
  $('#calibrate-apply').disabled = false;
}

function metricCard(label, value, sub, best) {
  const c = el('div', { className: 'cal-metric' + (best ? ' best' : '') },
    el('div', { className: 'label' }, label),
    el('div', { className: 'value' }, value),
  );
  if (sub) c.append(el('div', { className: 'label' }, sub));
  return c;
}

function renderCalibration() {
  const r = lastCalibration;
  if (!r) return;
  const div = $('#calibration-results');
  div.innerHTML = '';

  const summary = el('div', { className: 'calibration-summary' },
    metricCard('best F1', r.bestF1.f1.toFixed(3),
      `thr ${r.bestF1.threshold}  P=${r.bestF1.precision.toFixed(2)}  R=${r.bestF1.recall.toFixed(2)}`,
      true),
    r.bestPerfectRecall
      ? metricCard('highest thr @ 100% recall', r.bestPerfectRecall.threshold.toFixed(2),
          `P=${r.bestPerfectRecall.precision.toFixed(2)}  FP=${r.bestPerfectRecall.fp}`)
      : metricCard('100% recall', 'none',
          'no threshold catches every labeled shot at this tolerance'),
    metricCard('tolerance', `±${r.toleranceMs} ms`),
    metricCard('labeled files', String(r.bestF1.perFileDetections.length)),
  );
  div.append(summary);

  const sweepTable = el('table', { className: 'sweep-table' });
  sweepTable.innerHTML = `<thead><tr>
    <th>threshold</th><th>TP</th><th>FP</th><th>FN</th>
    <th>precision</th><th>recall</th><th>F1</th>
  </tr></thead><tbody></tbody>`;
  const tbody = sweepTable.querySelector('tbody');
  r.sweep.forEach((s, i) => {
    const show = i % 4 === 0 || s === r.bestF1 || s === r.bestPerfectRecall;
    if (!show) return;
    const tr = el('tr', {},
      el('td', { className: 'num' }, s.threshold.toFixed(2)),
      el('td', { className: 'num' }, String(s.tp)),
      el('td', { className: 'num' }, String(s.fp)),
      el('td', { className: 'num' }, String(s.fn)),
      el('td', { className: 'num' }, s.precision.toFixed(2)),
      el('td', { className: 'num' }, s.recall.toFixed(2)),
      el('td', { className: 'num' }, s.f1.toFixed(3)),
    );
    if (s === r.bestF1) tr.classList.add('best');
    else if (s === r.bestPerfectRecall) tr.classList.add('recall100');
    tbody.appendChild(tr);
  });
  div.append(el('h3', {}, 'Threshold sweep (green = best F1, blue = highest threshold @ 100% recall)'));
  div.append(sweepTable);

  div.append(el('h3', {}, `Per-file detail at threshold ${r.bestF1.threshold} (red = missed shot)`));
  const pfTable = el('table', { className: 'sweep-table' });
  pfTable.innerHTML = `<thead><tr>
    <th>file</th><th>ground truth (s)</th><th>detections (s)</th><th>TP</th><th>FP</th><th>FN</th>
  </tr></thead><tbody></tbody>`;
  const pfTbody = pfTable.querySelector('tbody');
  for (const pf of r.bestF1.perFileDetections) {
    const tr = el('tr', {},
      el('td', {}, pf.name),
      el('td', { className: 'num' }, pf.gt.map(t => t.toFixed(3)).join(', ')),
      el('td', { className: 'num' }, pf.onsets.map(o => o.time.toFixed(3)).join(', ') || '—'),
      el('td', { className: 'num' }, String(pf.fileTP)),
      el('td', { className: 'num' }, String(pf.fileFP)),
      el('td', { className: 'num' }, String(pf.fileFN)),
    );
    if (pf.fileFN > 0) tr.style.color = '#f88';
    pfTbody.appendChild(tr);
  }
  div.append(pfTable);
}

function applySuggestedThreshold() {
  if (!lastCalibration) return;
  const thr = lastCalibration.bestF1.threshold;
  params.threshold = thr;
  $('#threshold').value = thr;
  $('#threshold-val').textContent = thr.toFixed(2);
  setStatus(`Applied threshold ${thr} (best F1 ${lastCalibration.bestF1.f1.toFixed(3)})`, 'ok');
}

$('#calibrate-run').addEventListener('click', runCalibration);
$('#calibrate-apply').addEventListener('click', applySuggestedThreshold);
$('#calibrate-tolerance').addEventListener('input', (e) => {
  $('#calibrate-tolerance-val').textContent = e.target.value;
});
