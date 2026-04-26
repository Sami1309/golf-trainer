const REVIEWER = 'sam-visual';
const NUDGE_MS_FINE = 1;
const NUDGE_MS_COARSE = 10;
const PEAK_SEARCH_MS = 100;
const MODEL_PRE_MS = 100;
const MODEL_POST_MS = 400;
const PLAYBACK_PRE_MS = 200;
const PLAYBACK_POST_MS = 600;
const FULL_CANVAS_HEIGHT = 180;
const ZOOM_CANVAS_HEIGHT = 160;

const state = {
  doc: null,
  edits: new Map(),
  selectedKey: null,
  audioCtx: null,
  buffers: new Map(),
  inflightDecodes: new Map(),
  playback: null,
  filter: '',
};

const root = document.getElementById('review-app');
const boot = document.getElementById('boot-status');

init();

async function init() {
  let doc;
  try {
    const res = await fetch('/api/labels', { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    doc = await res.json();
  } catch (err) {
    showLocalOnly(err);
    return;
  }
  state.doc = doc;
  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: undefined });
  renderApp();
  const firstKey = sortedKeys()[0];
  if (firstKey) selectEntry(firstKey);
  window.addEventListener('keydown', onGlobalKeyDown);
}

function showLocalOnly(err) {
  boot.outerHTML = `
    <div class="local-only">
      <h1>Labels review (local only)</h1>
      <p>This page needs the dev server. Audio files (<code>*.m4a</code>) and <code>data/labels.json</code> are not deployed to GitHub Pages, so this tool only runs from a local checkout.</p>
      <p>From the repo root:</p>
      <pre>npm run review</pre>
      <p>Then open <code>http://127.0.0.1:5173/frontend/labels_review.html</code>.</p>
      <p style="color:#6e7681; font-size: 11px; margin-top: 18px;">${err && err.message ? `Reason: ${escapeHtml(err.message)}` : ''}</p>
    </div>
  `;
}

function renderApp() {
  root.innerHTML = `
    <header class="toolbar">
      <span class="title">Labels review</span>
      <span class="stats" id="stats"></span>
      <span class="spacer"></span>
      <button id="btn-snap-peak" title="Snap impact to peak in ±${PEAK_SEARCH_MS} ms (R)">Snap to peak</button>
      <button id="btn-confirm" title="Mark reviewed without changing time (C)">Confirm reviewed</button>
      <button id="btn-revert" class="danger" title="Discard local changes for this entry">Revert</button>
      <button id="btn-save" class="primary" disabled title="PATCH all edited entries to disk">Save changes</button>
      <button id="btn-prepare" title="Run npm run prepare:stage1b">Re-prepare clips</button>
      <button id="btn-train-stage1b" title="Run npm run train:stage1b">Retrain Stage 1b</button>
      <button id="btn-train-stage2" title="Run npm run train:stage2:pure-fat">Retrain Stage 2</button>
    </header>
    <div class="layout">
      <aside class="sidebar">
        <div class="sidebar-header">
          <input id="filter" type="search" placeholder="Filter by number or label…" autocomplete="off" />
        </div>
        <ul class="sidebar-list" id="entries"></ul>
      </aside>
      <main class="editor empty" id="editor">Select an entry to review.</main>
    </div>
    <div id="toast-host"></div>
  `;
  document.getElementById('filter').addEventListener('input', e => {
    state.filter = e.target.value.trim().toLowerCase();
    renderSidebar();
  });
  document.getElementById('btn-save').addEventListener('click', saveAll);
  document.getElementById('btn-snap-peak').addEventListener('click', snapToPeak);
  document.getElementById('btn-confirm').addEventListener('click', confirmReviewed);
  document.getElementById('btn-revert').addEventListener('click', revertSelection);
  document.getElementById('btn-prepare').addEventListener('click', () => runScript('prepare:stage1b'));
  document.getElementById('btn-train-stage1b').addEventListener('click', () => runScript('train:stage1b'));
  document.getElementById('btn-train-stage2').addEventListener('click', () => runScript('train:stage2:pure-fat'));
  renderSidebar();
  renderStats();
}

function sortedKeys() {
  return Object.keys(state.doc.labels).sort((a, b) => {
    const na = shotNumber(state.doc.labels[a]);
    const nb = shotNumber(state.doc.labels[b]);
    if (na != null && nb != null) return na - nb;
    return a.localeCompare(b);
  });
}

function shotNumber(entry) {
  const m = (entry.folderLabel || '').trim().match(/^(\d+)/);
  return m ? Number(m[1]) : null;
}

function effectiveEntry(key) {
  const base = state.doc.labels[key];
  const edit = state.edits.get(key);
  if (!edit) return base;
  return { ...base, ...edit };
}

function renderSidebar() {
  const list = document.getElementById('entries');
  if (!list) return;
  const keys = sortedKeys().filter(key => {
    if (!state.filter) return true;
    const e = state.doc.labels[key];
    return key.toLowerCase().includes(state.filter)
      || (e.folderLabel || '').toLowerCase().includes(state.filter);
  });
  list.innerHTML = '';
  for (const key of keys) {
    const entry = state.doc.labels[key];
    const eff = effectiveEntry(key);
    const dirty = state.edits.has(key);
    const auto = entry.labeledBy === 'codex-auto-import';
    const reviewed = !!eff.reviewedAt;
    const li = document.createElement('li');
    if (key === state.selectedKey) li.classList.add('active');
    li.innerHTML = `
      <div class="row-top">
        <span class="shot-num">${shotNumber(entry) ?? '–'}</span>
        <span class="folder">${escapeHtml(entry.folderLabel || key)}</span>
      </div>
      <div class="row-bottom">
        <span>${(eff.shotTimes?.[0] ?? 0).toFixed(3)} s</span>
        ${auto ? '<span class="badge auto">auto</span>' : ''}
        ${reviewed ? '<span class="badge reviewed">reviewed</span>' : ''}
        ${dirty ? '<span class="badge dirty">unsaved</span>' : ''}
      </div>
    `;
    li.addEventListener('click', () => selectEntry(key));
    list.appendChild(li);
  }
}

function renderStats() {
  const el = document.getElementById('stats');
  if (!el) return;
  const all = sortedKeys();
  const reviewed = all.filter(k => effectiveEntry(k).reviewedAt).length;
  const auto = all.filter(k => state.doc.labels[k].labeledBy === 'codex-auto-import').length;
  const dirty = state.edits.size;
  el.innerHTML = `<strong>${reviewed}</strong> / ${all.length} reviewed · <strong>${auto}</strong> auto-imported · <strong>${dirty}</strong> unsaved`;
  document.getElementById('btn-save').disabled = dirty === 0;
}

async function selectEntry(key) {
  state.selectedKey = key;
  stopPlayback();
  renderSidebar();
  renderEditor();
  await ensureBuffer(key);
  renderEditor();
}

function renderEditor() {
  const editor = document.getElementById('editor');
  if (!editor) return;
  const key = state.selectedKey;
  if (!key) {
    editor.className = 'editor empty';
    editor.textContent = 'Select an entry to review.';
    return;
  }
  editor.className = 'editor';
  const entry = state.doc.labels[key];
  const eff = effectiveEntry(key);
  const buffer = state.buffers.get(key);
  const baseTime = entry.shotTimes?.[0] ?? 0;
  const currTime = eff.shotTimes?.[0] ?? baseTime;
  const delta = currTime - baseTime;
  const auto = entry.labeledBy === 'codex-auto-import';
  const reviewed = !!eff.reviewedAt;

  editor.innerHTML = `
    <div class="editor-header">
      <h2>${escapeHtml(entry.folderLabel || key)}</h2>
      <span class="meta">
        ${escapeHtml(key)} · duration ${(entry.duration ?? 0).toFixed(2)} s
        ${auto ? '· labeled by auto-import' : `· labeled by ${escapeHtml(entry.labeledBy || 'unknown')}`}
        ${reviewed ? `· reviewed ${escapeHtml(eff.reviewedAt)}` : ''}
      </span>
    </div>

    <div class="canvas-block">
      <div class="label-row">
        <span>Full waveform</span>
        <span id="full-cursor"></span>
      </div>
      <canvas id="full-canvas"></canvas>
    </div>

    <div class="canvas-block">
      <div class="label-row">
        <span>Zoom ±${PEAK_SEARCH_MS} ms around impact (orange = 500 ms model clip)</span>
        <span id="zoom-cursor"></span>
      </div>
      <canvas id="zoom-canvas"></canvas>
    </div>

    <div class="controls">
      <span class="impact-time">impact: <strong id="impact-readout">${currTime.toFixed(4)}</strong> s</span>
      <span class="delta" id="delta-readout">${delta !== 0 ? `(Δ ${(delta * 1000).toFixed(1)} ms)` : ''}</span>
      <button id="btn-play">▶ Play (Space)</button>
      <button id="btn-nudge-back">← 10 ms</button>
      <button id="btn-nudge-fwd">→ 10 ms</button>
      <button id="btn-nudge-back-fine">← 1 ms (Shift+←)</button>
      <button id="btn-nudge-fwd-fine">→ 1 ms (Shift+→)</button>
    </div>

    <div id="run-host"></div>

    <div class="shortcut-help">
      Click waveform to set impact ·
      <kbd>←</kbd>/<kbd>→</kbd> nudge ${NUDGE_MS_COARSE} ms ·
      <kbd>Shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> nudge ${NUDGE_MS_FINE} ms ·
      <kbd>R</kbd> snap to peak ·
      <kbd>C</kbd> confirm reviewed ·
      <kbd>Space</kbd> play 600 ms ·
      <kbd>↑</kbd>/<kbd>↓</kbd> previous/next entry
    </div>
  `;

  document.getElementById('btn-play').addEventListener('click', playWindow);
  document.getElementById('btn-nudge-back').addEventListener('click', () => nudge(-NUDGE_MS_COARSE));
  document.getElementById('btn-nudge-fwd').addEventListener('click', () => nudge(+NUDGE_MS_COARSE));
  document.getElementById('btn-nudge-back-fine').addEventListener('click', () => nudge(-NUDGE_MS_FINE));
  document.getElementById('btn-nudge-fwd-fine').addEventListener('click', () => nudge(+NUDGE_MS_FINE));

  if (buffer) {
    drawCanvases();
    bindCanvasInteractions();
  } else {
    drawLoadingPlaceholder();
  }
}

function drawLoadingPlaceholder() {
  for (const [id, cssH] of [['full-canvas', FULL_CANVAS_HEIGHT], ['zoom-canvas', ZOOM_CANVAS_HEIGHT]]) {
    const c = document.getElementById(id);
    if (!c) continue;
    const dim = sizeCanvas(c, cssH);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, dim.width, dim.height);
    ctx.fillStyle = '#6e7681';
    ctx.font = '12px system-ui';
    ctx.fillText('decoding audio…', 12, 24);
  }
}

function sizeCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.parentElement.clientWidth;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  return { width: cssWidth, height: cssHeight };
}

function drawCanvases() {
  const buffer = state.buffers.get(state.selectedKey);
  if (!buffer) return;
  const eff = effectiveEntry(state.selectedKey);
  const t = eff.shotTimes?.[0] ?? 0;
  const samples = buffer.getChannelData(0);
  const sr = buffer.sampleRate;
  const fullCanvas = document.getElementById('full-canvas');
  const zoomCanvas = document.getElementById('zoom-canvas');
  if (fullCanvas) {
    const dim = sizeCanvas(fullCanvas, FULL_CANVAS_HEIGHT);
    drawWaveform(fullCanvas, samples, sr, {
      startTime: 0,
      endTime: samples.length / sr,
      markerTime: t,
      markerColor: '#ff5555',
      modelClip: { preMs: MODEL_PRE_MS, postMs: MODEL_POST_MS },
      width: dim.width,
      height: dim.height,
    });
  }
  if (zoomCanvas) {
    const dim = sizeCanvas(zoomCanvas, ZOOM_CANVAS_HEIGHT);
    const startTime = Math.max(0, t - PEAK_SEARCH_MS / 1000);
    const endTime = Math.min(samples.length / sr, t + PEAK_SEARCH_MS / 1000);
    drawWaveform(zoomCanvas, samples, sr, {
      startTime,
      endTime,
      markerTime: t,
      markerColor: '#ff5555',
      modelClip: { preMs: MODEL_PRE_MS, postMs: MODEL_POST_MS },
      grid: true,
      width: dim.width,
      height: dim.height,
    });
  }
}

function drawWaveform(canvas, samples, sr, opts) {
  const ctx = canvas.getContext('2d');
  const W = opts.width;
  const H = opts.height;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  const span = opts.endTime - opts.startTime;
  if (span <= 0) return;
  const startSample = Math.max(0, Math.floor(opts.startTime * sr));
  const endSample = Math.min(samples.length, Math.ceil(opts.endTime * sr));

  if (opts.modelClip && opts.markerTime != null) {
    const xClipStart = ((opts.markerTime - opts.modelClip.preMs / 1000) - opts.startTime) / span * W;
    const xClipEnd = ((opts.markerTime + opts.modelClip.postMs / 1000) - opts.startTime) / span * W;
    ctx.fillStyle = 'rgba(255, 165, 0, 0.10)';
    ctx.fillRect(xClipStart, 0, xClipEnd - xClipStart, H);
  }

  if (opts.grid) {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let ms = -PEAK_SEARCH_MS; ms <= PEAK_SEARCH_MS; ms += 10) {
      const t = (opts.markerTime ?? 0) + ms / 1000;
      const x = (t - opts.startTime) / span * W;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = '#3aa8ff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const mid = H / 2;
  const sampleSpan = endSample - startSample;
  const samplesPerPixel = sampleSpan / W;
  for (let x = 0; x < W; x++) {
    const i0 = startSample + Math.floor(x * samplesPerPixel);
    const i1 = Math.min(samples.length, startSample + Math.floor((x + 1) * samplesPerPixel) + 1);
    let mn = 0;
    let mx = 0;
    for (let i = i0; i < i1; i++) {
      const v = samples[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const yMin = mid - mn * mid * 0.92;
    const yMax = mid - mx * mid * 0.92;
    ctx.moveTo(x + 0.5, yMin);
    ctx.lineTo(x + 0.5, yMax);
  }
  ctx.stroke();

  if (opts.markerTime != null && opts.markerTime >= opts.startTime && opts.markerTime <= opts.endTime) {
    const x = (opts.markerTime - opts.startTime) / span * W;
    ctx.strokeStyle = opts.markerColor || '#ff5555';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
}

function bindCanvasInteractions() {
  for (const id of ['full-canvas', 'zoom-canvas']) {
    const c = document.getElementById(id);
    if (!c) continue;
    c.addEventListener('click', e => onCanvasClick(c, e));
  }
}

function onCanvasClick(canvas, event) {
  const buffer = state.buffers.get(state.selectedKey);
  if (!buffer) return;
  const rect = canvas.getBoundingClientRect();
  const fraction = (event.clientX - rect.left) / rect.width;
  const eff = effectiveEntry(state.selectedKey);
  const t = eff.shotTimes?.[0] ?? 0;
  const isZoom = canvas.id === 'zoom-canvas';
  const startTime = isZoom ? Math.max(0, t - PEAK_SEARCH_MS / 1000) : 0;
  const endTime = isZoom ? Math.min(buffer.duration, t + PEAK_SEARCH_MS / 1000) : buffer.duration;
  const newTime = startTime + fraction * (endTime - startTime);
  setImpactTime(newTime);
}

function setImpactTime(t) {
  const key = state.selectedKey;
  if (!key) return;
  const buffer = state.buffers.get(key);
  const max = buffer ? buffer.duration : (state.doc.labels[key].duration || Infinity);
  const clamped = Math.min(Math.max(0, t), max);
  const base = state.doc.labels[key];
  const baseTime = base.shotTimes?.[0] ?? 0;
  const eff = effectiveEntry(key);
  const newShotTimes = [+clamped.toFixed(6), ...((eff.shotTimes || []).slice(1))];
  const dirty = newShotTimes[0] !== baseTime;
  if (!dirty && state.edits.has(key)) {
    state.edits.delete(key);
  } else if (dirty) {
    const cur = state.edits.get(key) || {};
    state.edits.set(key, { ...cur, shotTimes: newShotTimes });
  }
  drawCanvases();
  updateImpactReadout();
  renderSidebar();
  renderStats();
}

function updateImpactReadout() {
  const key = state.selectedKey;
  if (!key) return;
  const eff = effectiveEntry(key);
  const base = state.doc.labels[key];
  const t = eff.shotTimes?.[0] ?? 0;
  const baseTime = base.shotTimes?.[0] ?? 0;
  const delta = t - baseTime;
  const elT = document.getElementById('impact-readout');
  const elD = document.getElementById('delta-readout');
  if (elT) elT.textContent = t.toFixed(4);
  if (elD) elD.textContent = delta !== 0 ? `(Δ ${(delta * 1000).toFixed(1)} ms)` : '';
}

function nudge(deltaMs) {
  const key = state.selectedKey;
  if (!key) return;
  const eff = effectiveEntry(key);
  setImpactTime((eff.shotTimes?.[0] ?? 0) + deltaMs / 1000);
}

function snapToPeak() {
  const key = state.selectedKey;
  if (!key) return;
  const buffer = state.buffers.get(key);
  if (!buffer) return;
  const samples = buffer.getChannelData(0);
  const sr = buffer.sampleRate;
  const eff = effectiveEntry(key);
  const center = Math.round((eff.shotTimes?.[0] ?? 0) * sr);
  const radius = Math.round((PEAK_SEARCH_MS / 1000) * sr);
  const lo = Math.max(0, center - radius);
  const hi = Math.min(samples.length, center + radius);
  let peak = 0;
  let peakIdx = center;
  for (let i = lo; i < hi; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) { peak = v; peakIdx = i; }
  }
  setImpactTime(peakIdx / sr);
  toast(`Snapped to peak (|x| ${peak.toFixed(3)})`, 'success');
}

async function confirmReviewed() {
  const key = state.selectedKey;
  if (!key) return;
  const eff = effectiveEntry(key);
  try {
    const res = await patchLabel(key, { shotTimes: eff.shotTimes });
    state.doc.labels[key] = res.entry;
    state.edits.delete(key);
    advanceToNextUnreviewed();
    toast('Confirmed reviewed', 'success');
  } catch (err) {
    toast(`Save failed: ${err.message}`, 'error');
  }
  renderSidebar();
  renderStats();
}

async function saveAll() {
  const keys = [...state.edits.keys()];
  if (!keys.length) return;
  const failures = [];
  for (const key of keys) {
    const edit = state.edits.get(key);
    try {
      const res = await patchLabel(key, { shotTimes: edit.shotTimes });
      state.doc.labels[key] = res.entry;
      state.edits.delete(key);
    } catch (err) {
      failures.push({ key, message: err.message });
    }
  }
  if (failures.length) {
    toast(`Saved ${keys.length - failures.length}/${keys.length}; failed: ${failures.map(f => f.key).join(', ')}`, 'error');
  } else {
    toast(`Saved ${keys.length} edit${keys.length === 1 ? '' : 's'}`, 'success');
  }
  renderSidebar();
  renderStats();
  if (state.selectedKey) renderEditor();
}

function revertSelection() {
  const key = state.selectedKey;
  if (!key) return;
  if (!state.edits.has(key)) return;
  state.edits.delete(key);
  drawCanvases();
  updateImpactReadout();
  renderSidebar();
  renderStats();
  toast('Reverted', 'success');
}

async function patchLabel(key, body) {
  const res = await fetch('/api/labels', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, ...body, reviewedBy: REVIEWER }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

async function ensureBuffer(key) {
  if (state.buffers.has(key)) return state.buffers.get(key);
  if (state.inflightDecodes.has(key)) return state.inflightDecodes.get(key);
  const entry = state.doc.labels[key];
  const relPath = entry.path.replace(/^samples\//, '');
  const url = '/' + relPath.split('/').map(encodeURIComponent).join('/');
  const promise = (async () => {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`audio fetch ${res.status}`);
      const ab = await res.arrayBuffer();
      const buffer = await state.audioCtx.decodeAudioData(ab);
      state.buffers.set(key, buffer);
      return buffer;
    } finally {
      state.inflightDecodes.delete(key);
    }
  })();
  state.inflightDecodes.set(key, promise);
  return promise;
}

async function playWindow() {
  const key = state.selectedKey;
  if (!key) return;
  const buffer = state.buffers.get(key);
  if (!buffer) return;
  stopPlayback();
  if (state.audioCtx.state === 'suspended') {
    try { await state.audioCtx.resume(); } catch {}
  }
  const eff = effectiveEntry(key);
  const t = eff.shotTimes?.[0] ?? 0;
  const startTime = Math.max(0, t - PLAYBACK_PRE_MS / 1000);
  const duration = Math.min(buffer.duration - startTime, (PLAYBACK_PRE_MS + PLAYBACK_POST_MS) / 1000);
  if (duration <= 0) return;
  const src = state.audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(state.audioCtx.destination);
  const when = state.audioCtx.currentTime;
  src.start(when, startTime, duration);
  src.stop(when + duration);
  state.playback = { src };
  src.onended = () => { if (state.playback?.src === src) state.playback = null; };
}

function stopPlayback() {
  if (!state.playback) return;
  const { src } = state.playback;
  state.playback = null;
  try {
    src.onended = null;
    src.stop();
    src.disconnect();
  } catch {}
}

function advanceToNextUnreviewed() {
  const keys = sortedKeys();
  const idx = keys.indexOf(state.selectedKey);
  for (let i = idx + 1; i < keys.length; i++) {
    if (!effectiveEntry(keys[i]).reviewedAt) { selectEntry(keys[i]); return; }
  }
  for (let i = 0; i < idx; i++) {
    if (!effectiveEntry(keys[i]).reviewedAt) { selectEntry(keys[i]); return; }
  }
}

function moveSelection(direction) {
  const keys = sortedKeys();
  const idx = keys.indexOf(state.selectedKey);
  if (idx < 0) return;
  const next = keys[Math.min(keys.length - 1, Math.max(0, idx + direction))];
  if (next && next !== state.selectedKey) selectEntry(next);
}

function onGlobalKeyDown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.metaKey || e.ctrlKey) return;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    nudge(e.shiftKey ? -NUDGE_MS_FINE : -NUDGE_MS_COARSE);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    nudge(e.shiftKey ? +NUDGE_MS_FINE : +NUDGE_MS_COARSE);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    moveSelection(-1);
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    moveSelection(+1);
  } else if (e.key === ' ') {
    e.preventDefault();
    if (state.playback) stopPlayback(); else playWindow();
  } else if (e.key === 'r' || e.key === 'R') {
    snapToPeak();
  } else if (e.key === 'c' || e.key === 'C') {
    confirmReviewed();
  } else if (e.key === 's' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    saveAll();
  }
}

function runScript(script) {
  const host = document.getElementById('run-host');
  if (!host) return;
  host.innerHTML = `<div class="run-panel" id="run-panel"><strong>$ npm run ${script}</strong>\n</div>`;
  const panel = document.getElementById('run-panel');
  const append = (klass, text) => {
    const span = document.createElement('span');
    span.className = klass;
    span.textContent = text;
    panel.appendChild(span);
    panel.scrollTop = panel.scrollHeight;
  };
  const es = new EventSource(`/api/run?script=${encodeURIComponent(script)}`);
  es.addEventListener('start', () => append('stdout', `[started ${new Date().toLocaleTimeString()}]\n`));
  es.addEventListener('stdout', e => append('stdout', JSON.parse(e.data).line));
  es.addEventListener('stderr', e => append('stderr', JSON.parse(e.data).line));
  es.addEventListener('exit', e => {
    const code = JSON.parse(e.data).code;
    append(code === 0 ? 'exit-ok' : 'exit-fail', `\n[exit ${code}]\n`);
    if (code !== 0) panel.classList.add('error');
    es.close();
    if (code === 0 && script === 'prepare:stage1b') {
      toast('Re-prepare done. Now retrain Stage 1b and Stage 2.', 'success');
    } else if (code === 0) {
      toast(`${script} done`, 'success');
    } else {
      toast(`${script} failed (${code})`, 'error');
    }
  });
  es.addEventListener('error', () => {
    append('stderr', '\n[stream error]\n');
    es.close();
    toast(`${script} stream error`, 'error');
  });
}

function toast(message, kind) {
  const host = document.getElementById('toast-host');
  if (!host) return;
  const el = document.createElement('div');
  el.className = `toast ${kind || ''}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
