const DB_NAME = 'golf-shot-live-store';
const DB_VERSION = 1;
const STORE = 'detections';

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('createdAt', 'createdAt');
        store.createIndex('label', 'label');
        store.createIndex('stage1bLabel', 'verification.label');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function requestToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function transaction(mode, fn) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const store = tx.objectStore(STORE);
    let result;
    try {
      result = fn(store);
    } catch (e) {
      reject(e);
      return;
    }
    tx.oncomplete = () => resolve(result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export async function putDetection(record) {
  await transaction('readwrite', store => {
    store.put(record);
  });
  return record;
}

export async function getAllDetections() {
  const db = await openDb();
  const tx = db.transaction(STORE, 'readonly');
  const store = tx.objectStore(STORE);
  const req = store.index('createdAt').getAll();
  const records = await requestToPromise(req);
  return records.sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''));
}

export async function deleteDetection(id) {
  await transaction('readwrite', store => {
    store.delete(id);
  });
}

export async function clearDetectionsStore() {
  await transaction('readwrite', store => {
    store.clear();
  });
}
