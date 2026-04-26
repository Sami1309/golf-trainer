import { createServer } from 'node:http';
import { readFile, writeFile, stat, rename } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, join, extname, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.REVIEW_SERVER_PORT || 5173);
const HOST = process.env.REVIEW_SERVER_HOST || '127.0.0.1';
const LABELS_PATH = join(ROOT, 'data', 'labels.json');

const ALLOWED_SCRIPTS = new Set([
  'prepare:stage1b',
  'train:stage1b',
  'train:stage1b:logmel',
  'train:stage1b:handcrafted',
  'train:stage2:pure-fat',
  'validate:stage2:pure-fat',
]);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.m4a': 'audio/mp4',
  '.mp4': 'audio/mp4',
  '.wav': 'audio/wav',
  '.mov': 'video/quicktime',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
};

function safeJoin(rootDir, requestPath) {
  let decoded;
  try { decoded = decodeURIComponent(requestPath); } catch { return null; }
  const trimmed = decoded.replace(/^\/+/, '');
  const target = normalize(join(rootDir, trimmed));
  if (target !== rootDir && !target.startsWith(rootDir + sep)) return null;
  return target;
}

let writeChain = Promise.resolve();
function withWriteLock(fn) {
  const next = writeChain.then(fn, fn);
  writeChain = next.catch(() => {});
  return next;
}

async function readBody(req, maxBytes = 2 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let received = 0;
    const chunks = [];
    req.on('data', chunk => {
      received += chunk.length;
      if (received > maxBytes) {
        reject(new Error('payload too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function jsonResponse(res, status, body) {
  const buf = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': buf.length,
    'Cache-Control': 'no-store',
  });
  res.end(buf);
}

async function handleGetLabels(req, res) {
  const raw = await readFile(LABELS_PATH, 'utf8');
  res.writeHead(200, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(raw),
    'Cache-Control': 'no-store',
  });
  res.end(raw);
}

async function handlePatchLabel(req, res) {
  const raw = await readBody(req);
  let payload;
  try { payload = JSON.parse(raw.toString('utf8')); }
  catch (e) { return jsonResponse(res, 400, { error: 'invalid_json', message: e.message }); }

  const { key, shotTimes, reviewedBy, reviewedAt, note, clearReview } = payload || {};
  if (typeof key !== 'string' || !key) {
    return jsonResponse(res, 400, { error: 'missing_key' });
  }
  if (shotTimes !== undefined) {
    if (!Array.isArray(shotTimes) || !shotTimes.length || !shotTimes.every(t => Number.isFinite(t) && t >= 0)) {
      return jsonResponse(res, 400, { error: 'invalid_shot_times' });
    }
  }

  try {
    const updated = await withWriteLock(async () => {
      const doc = JSON.parse(await readFile(LABELS_PATH, 'utf8'));
      if (!doc.labels[key]) {
        const err = new Error(`unknown_key:${key}`);
        err.status = 404;
        throw err;
      }
      const entry = doc.labels[key];
      const before = JSON.stringify(entry.shotTimes);
      if (Array.isArray(shotTimes)) {
        entry.shotTimes = shotTimes.map(t => +Number(t).toFixed(6));
      }
      const movedTime = JSON.stringify(entry.shotTimes) !== before;
      if (clearReview) {
        delete entry.reviewedAt;
        delete entry.reviewedBy;
        delete entry.reviewNote;
      } else {
        entry.reviewedAt = reviewedAt || new Date().toISOString();
        if (reviewedBy) entry.reviewedBy = reviewedBy;
        if (note) entry.reviewNote = note;
        if (movedTime) {
          entry.lastEditedAt = entry.reviewedAt;
          entry.lastEditedBy = entry.reviewedBy || 'unknown';
        }
      }
      const tmpPath = `${LABELS_PATH}.${randomUUID()}.tmp`;
      await writeFile(tmpPath, `${JSON.stringify(doc, null, 2)}\n`);
      await rename(tmpPath, LABELS_PATH);
      return entry;
    });
    jsonResponse(res, 200, { ok: true, key, entry: updated });
  } catch (e) {
    if (e.status === 404) return jsonResponse(res, 404, { error: e.message });
    throw e;
  }
}

function handleRunScript(req, res, url) {
  const script = url.searchParams.get('script');
  if (!script || !ALLOWED_SCRIPTS.has(script)) {
    return jsonResponse(res, 400, { error: 'unknown_script', allowed: [...ALLOWED_SCRIPTS] });
  }
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  send('start', { script, at: new Date().toISOString() });

  const child = spawn('npm', ['run', script], { cwd: ROOT, env: process.env });
  child.stdout.on('data', chunk => send('stdout', { line: chunk.toString() }));
  child.stderr.on('data', chunk => send('stderr', { line: chunk.toString() }));
  child.on('close', code => {
    send('exit', { code });
    res.end();
  });
  child.on('error', err => {
    send('error', { message: err.message });
    res.end();
  });
  req.on('close', () => {
    if (child.exitCode == null) child.kill('SIGTERM');
  });
}

async function handleStatic(req, res, url) {
  let path = url.pathname;
  if (path === '/' || path === '') path = '/frontend/labels_review.html';
  const target = safeJoin(ROOT, path);
  if (!target) {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    res.end('forbidden');
    return;
  }
  let info;
  try { info = await stat(target); }
  catch { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('not found'); return; }

  if (info.isDirectory()) {
    const indexPath = join(target, 'index.html');
    try {
      const data = await readFile(indexPath);
      res.writeHead(200, { 'Content-Type': MIME['.html'], 'Content-Length': data.length, 'Cache-Control': 'no-store' });
      res.end(data);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('not found');
    }
    return;
  }

  const data = await readFile(target);
  const mime = MIME[extname(target).toLowerCase()] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': mime,
    'Content-Length': data.length,
    'Cache-Control': 'no-store',
    'Accept-Ranges': 'bytes',
  });
  res.end(data);
}

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);
    if (url.pathname === '/api/labels') {
      if (req.method === 'GET') return await handleGetLabels(req, res);
      if (req.method === 'PATCH') return await handlePatchLabel(req, res);
      res.writeHead(405); res.end('method not allowed'); return;
    }
    if (url.pathname === '/api/run') {
      if (req.method === 'GET') return handleRunScript(req, res, url);
      res.writeHead(405); res.end('method not allowed'); return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405); res.end('method not allowed'); return;
    }
    return await handleStatic(req, res, url);
  } catch (e) {
    console.error('[review-server]', e);
    if (!res.headersSent) jsonResponse(res, 500, { error: e.message || 'server_error' });
    else res.end();
  }
}).listen(PORT, HOST, () => {
  console.log(`Review server listening on http://${HOST}:${PORT}`);
  console.log(`Open http://${HOST}:${PORT}/frontend/labels_review.html`);
  console.log('Press Ctrl+C to stop.');
});
