// SessionRecorder: long-form mic capture for a recording session.
//
// Wraps MediaRecorder so that:
//   - the entire mic stream is captured to a single file at the end of the
//     session (Voice-Memos-style)
//   - chunks are persisted to IndexedDB every `timeslice` ms while recording
//     so that an app crash, reload, or tab close does not lose the audio
//   - on a normal stop, chunks are concatenated in memory, returned as a
//     single Blob, and the IDB rows are deleted
//   - on a crash, the IDB session row is left in `recording` state and can be
//     recovered + downloaded on the next page load
//
// Public API:
//   const rec = new SessionRecorder({ stream, timesliceMs })
//   await rec.start()
//   ...later...
//   const { blob, filename, mimeType } = await rec.stop()
//   await rec.discard()  // give up without keeping anything
//
// MediaRecorder availability/format:
//   - Chrome/Android: audio/webm;codecs=opus (preferred)
//   - iOS Safari 14.3+: audio/mp4 (some builds also accept audio/aac)
//   - Falls back to whatever the browser advertises if neither preferred
//     mimeType is supported.

import {
  appendChunk,
  assembleSessionBlob,
  createSession,
  deleteSession,
  getSession,
  listOrphanSessions,
  markSessionComplete,
  totalSessionBytes,
} from './recording_store.js';

const PREFERRED_MIME_TYPES = [
  { mimeType: 'audio/webm;codecs=opus', fileExt: 'webm' },
  { mimeType: 'audio/webm', fileExt: 'webm' },
  { mimeType: 'audio/mp4', fileExt: 'm4a' },
  { mimeType: 'audio/aac', fileExt: 'm4a' },
  { mimeType: 'audio/ogg;codecs=opus', fileExt: 'ogg' },
];

export function pickSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const candidate of PREFERRED_MIME_TYPES) {
    try {
      if (MediaRecorder.isTypeSupported(candidate.mimeType)) return candidate;
    } catch {}
  }
  return { mimeType: '', fileExt: 'webm' };
}

export function isMediaRecorderSupported() {
  return pickSupportedMimeType() != null;
}

function makeId(prefix = 'session') {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export class SessionRecorder {
  constructor({ stream, timesliceMs = 30000, fileBaseName = 'session' } = {}) {
    if (!stream) throw new Error('SessionRecorder requires a MediaStream');
    this.stream = stream;
    this.timesliceMs = timesliceMs;
    this.fileBaseName = fileBaseName;
    this.id = makeId('session');
    this.startedAt = null;
    this.chunks = [];
    this.chunkIndex = 0;
    this.recorder = null;
    this.mimeType = null;
    this.fileExt = 'webm';
    this.persistFailed = false;
    this._stopPromise = null;
    this._error = null;
  }

  async start() {
    const picked = pickSupportedMimeType();
    if (!picked) {
      throw new Error('MediaRecorder is not available in this browser.');
    }
    this.mimeType = picked.mimeType;
    this.fileExt = picked.fileExt;
    this.startedAt = new Date().toISOString();

    try {
      this.recorder = new MediaRecorder(
        this.stream,
        this.mimeType ? { mimeType: this.mimeType } : undefined,
      );
    } catch (e) {
      // some Safari builds reject the explicit mimeType but accept the default
      this.recorder = new MediaRecorder(this.stream);
      this.mimeType = this.recorder.mimeType || '';
    }

    this.recorder.ondataavailable = (ev) => {
      if (!ev.data || ev.data.size === 0) return;
      this.chunks.push(ev.data);
      const idx = this.chunkIndex++;
      // Persist asynchronously; never block the recorder.
      appendChunk(this.id, idx, ev.data).catch(err => {
        this.persistFailed = true;
        console.warn('chunk persist failed', err);
      });
    };
    this.recorder.onerror = (ev) => {
      this._error = ev.error || new Error('MediaRecorder error');
      console.warn('MediaRecorder error', this._error);
    };

    await createSession({
      id: this.id,
      mimeType: this.mimeType,
      fileExt: this.fileExt,
      startedAt: this.startedAt,
    });

    // Some Safari/iOS builds reject `start(timeslice)` and need a discrete
    // start() then a manual `requestData()` cadence. We try the timeslice
    // path first (Chrome / modern Safari) and fall back if it throws.
    try {
      this.recorder.start(this.timesliceMs);
    } catch {
      this.recorder.start();
      this._fallbackChunkTimer = setInterval(() => {
        if (this.recorder && this.recorder.state === 'recording') {
          try { this.recorder.requestData(); } catch {}
        }
      }, this.timesliceMs);
    }
  }

  async stop() {
    if (this._stopPromise) return this._stopPromise;
    this._stopPromise = (async () => {
      if (this._fallbackChunkTimer) clearInterval(this._fallbackChunkTimer);
      const recorder = this.recorder;
      if (recorder && recorder.state !== 'inactive') {
        await new Promise((resolve) => {
          recorder.onstop = () => resolve();
          try { recorder.stop(); } catch { resolve(); }
        });
      }

      const mimeType = this.mimeType || (this.chunks[0]?.type) || 'audio/webm';
      const blob = new Blob(this.chunks, { type: mimeType });
      const filename = `${this.fileBaseName}_${this.startedAt.replace(/[:.]/g, '-')}.${this.fileExt}`;

      try {
        await markSessionComplete(this.id);
      } catch (e) {
        console.warn('mark session complete failed', e);
      }
      try {
        await deleteSession(this.id);
      } catch (e) {
        console.warn('delete session failed', e);
      }

      return { blob, filename, mimeType, sessionId: this.id, sizeBytes: blob.size };
    })();
    return this._stopPromise;
  }

  async discard() {
    if (this._fallbackChunkTimer) clearInterval(this._fallbackChunkTimer);
    const recorder = this.recorder;
    if (recorder && recorder.state !== 'inactive') {
      try { recorder.stop(); } catch {}
    }
    try { await deleteSession(this.id); } catch {}
  }

  get state() {
    return this.recorder ? this.recorder.state : 'inactive';
  }
}

export async function listRecoverableSessions() {
  const sessions = await listOrphanSessions();
  const out = [];
  for (const s of sessions) {
    const bytes = await totalSessionBytes(s.id);
    if (bytes > 0) out.push({ ...s, sizeBytes: bytes });
  }
  return out;
}

export async function recoverSessionToBlob(sessionId) {
  const blob = await assembleSessionBlob(sessionId);
  const session = await getSession(sessionId);
  if (!blob || !session) return null;
  const filename = `${session.note || 'session'}_recovered_${(session.startedAt || '').replace(/[:.]/g, '-')}.${session.fileExt || 'webm'}`;
  return { blob, filename, mimeType: session.mimeType || blob.type, session };
}

export async function discardRecoverableSession(sessionId) {
  await deleteSession(sessionId);
}
