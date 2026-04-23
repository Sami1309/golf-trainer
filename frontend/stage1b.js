import {
  LOG_MEL_FEATURE_NAMES,
  STAGE1_FEATURE_NAMES,
  extractLogMelFeatures,
  extractStage1Features,
  prepareModelClip,
} from './audio_features.js';

const DEFAULT_MODEL_URL = './models/stage1b_detector.json';
const FEATURE_EXTRACTORS = {
  stage1_handcrafted: {
    names: STAGE1_FEATURE_NAMES,
    extract: extractStage1Features,
  },
  logmel_summary: {
    names: LOG_MEL_FEATURE_NAMES,
    extract: extractLogMelFeatures,
  },
};

let model = null;
let loadError = null;

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export async function loadStage1bModel(url = DEFAULT_MODEL_URL) {
  loadError = null;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    loadError = `HTTP ${res.status}`;
    throw new Error(`Stage 1b model load failed: ${loadError}`);
  }
  const loaded = await res.json();
  const featureExtractor = loaded.featureExtractor || 'stage1_handcrafted';
  const extractor = FEATURE_EXTRACTORS[featureExtractor];
  if (!extractor) {
    throw new Error(`Unsupported Stage 1b feature extractor: ${featureExtractor}`);
  }
  if (!Array.isArray(loaded.features) || loaded.features.join('|') !== extractor.names.join('|')) {
    throw new Error('Stage 1b model feature list does not match frontend feature extractor');
  }
  loaded.featureExtractor = featureExtractor;
  model = loaded;
  return model;
}

export function getStage1bModel() {
  return model;
}

export function getStage1bLoadError() {
  return loadError;
}

export function verifyShot(samples, opts = {}) {
  const prepared = prepareModelClip(samples);
  if (!model) {
    return {
      available: false,
      label: 'shot',
      pShot: 1,
      confidence: 1,
      threshold: null,
      samples: prepared,
    };
  }

  const extractor = FEATURE_EXTRACTORS[model.featureExtractor || 'stage1_handcrafted'];
  const features = extractor.extract(prepared);
  let logit = model.bias;
  for (let i = 0; i < features.length; i++) {
    const z = (features[i] - model.mean[i]) / model.std[i];
    logit += model.weights[i] * z;
  }

  const pShot = sigmoid(logit);
  const threshold = opts.threshold ?? model.threshold ?? 0.7;
  const accepted = pShot >= threshold;
  return {
    available: true,
    label: accepted ? 'shot' : 'not_shot',
    pShot,
    confidence: accepted ? pShot : 1 - pShot,
    threshold,
    samples: prepared,
  };
}
