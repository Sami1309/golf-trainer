// IndexedDB store for in-progress session recordings.
//
// Two object stores in a separate database from `shot_store`:
//   - sessions: { id, status: 'recording' | 'complete' | 'failed',
//                 mimeType, startedAt, endedAt?, fileExt, sampleRate? }
//   - chunks:   { key: `${sessionId}#${index}`, sessionId, index, blob, createdAt }
//
// `recording` rows whose chunks remain on next page load are treated as
// orphan/crash-recovered sessions and surfaced to the user.

const DB_NAME = 'golf-shot-session-recordings';
const DB_VERSION = 1;
const SESSIONS = 'sessions';
const CHUNKS = 'chunks';

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(SESSIONS)) {
        const s = db.createObjectStore(SESSIONS, { keyPath: 'id' });
        s.createIndex('status', 'status');
        s.createIndex('startedAt', 'startedAt');
      }
      if (!db.objectStoreNames.contains(CHUNKS)) {
        const c = db.createObjectStore(CHUNKS, { keyPath: 'key' });
        c.createIndex('sessionId', 'sessionId');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function reqToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export async function createSession(meta) {
  const db = await openDb();
  const tx = db.transaction(SESSIONS, 'readwrite');
  tx.objectStore(SESSIONS).put({
    id: meta.id,
    status: 'recording',
    mimeType: meta.mimeType || 'audio/webm',
    fileExt: meta.fileExt || 'webm',
    startedAt: meta.startedAt || new Date().toISOString(),
    note: meta.note || null,
  });
  await txDone(tx);
}

export async function appendChunk(sessionId, index, blob) {
  const db = await openDb();
  const tx = db.transaction(CHUNKS, 'readwrite');
  tx.objectStore(CHUNKS).put({
    key: `${sessionId}#${String(index).padStart(6, '0')}`,
    sessionId,
    index,
    blob,
    createdAt: new Date().toISOString(),
  });
  await txDone(tx);
}

export async function listSessions() {
  const db = await openDb();
  const tx = db.transaction(SESSIONS, 'readonly');
  const all = await reqToPromise(tx.objectStore(SESSIONS).getAll());
  return all.sort((a, b) => (a.startedAt || '').localeCompare(b.startedAt || ''));
}

export async function listOrphanSessions() {
  const all = await listSessions();
  return all.filter(s => s.status === 'recording');
}

export async function getSessionChunks(sessionId) {
  const db = await openDb();
  const tx = db.transaction(CHUNKS, 'readonly');
  const idx = tx.objectStore(CHUNKS).index('sessionId');
  const rows = await reqToPromise(idx.getAll(IDBKeyRange.only(sessionId)));
  rows.sort((a, b) => a.index - b.index);
  return rows;
}

export async function assembleSessionBlob(sessionId) {
  const session = await getSession(sessionId);
  const chunks = await getSessionChunks(sessionId);
  if (!chunks.length) return null;
  const mimeType = session?.mimeType || chunks[0].blob.type || 'audio/webm';
  return new Blob(chunks.map(c => c.blob), { type: mimeType });
}

export async function getSession(sessionId) {
  const db = await openDb();
  const tx = db.transaction(SESSIONS, 'readonly');
  return reqToPromise(tx.objectStore(SESSIONS).get(sessionId));
}

export async function markSessionComplete(sessionId) {
  const db = await openDb();
  const tx = db.transaction(SESSIONS, 'readwrite');
  const store = tx.objectStore(SESSIONS);
  const existing = await reqToPromise(store.get(sessionId));
  if (existing) {
    existing.status = 'complete';
    existing.endedAt = new Date().toISOString();
    store.put(existing);
  }
  await txDone(tx);
}

export async function deleteSession(sessionId) {
  const db = await openDb();
  const tx = db.transaction([SESSIONS, CHUNKS], 'readwrite');
  tx.objectStore(SESSIONS).delete(sessionId);
  const idx = tx.objectStore(CHUNKS).index('sessionId');
  const cursorReq = idx.openCursor(IDBKeyRange.only(sessionId));
  await new Promise((resolve, reject) => {
    cursorReq.onsuccess = () => {
      const cursor = cursorReq.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      } else {
        resolve();
      }
    };
    cursorReq.onerror = () => reject(cursorReq.error);
  });
  await txDone(tx);
}

export async function totalSessionBytes(sessionId) {
  const chunks = await getSessionChunks(sessionId);
  return chunks.reduce((sum, c) => sum + (c.blob?.size || 0), 0);
}
