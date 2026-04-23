import {
  LOG_MEL_FEATURE_NAMES,
  extractLogMelFeatures,
  prepareModelClip,
} from './audio_features.js';

const DEFAULT_MODEL_URL = './models/stage2_pure_fat.json';

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

export async function loadStage2Model(url = DEFAULT_MODEL_URL) {
  loadError = null;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    loadError = `HTTP ${res.status}`;
    throw new Error(`Stage 2 model load failed: ${loadError}`);
  }
  const loaded = await res.json();
  if (loaded.featureExtractor !== 'logmel_summary') {
    throw new Error(`Unsupported Stage 2 feature extractor: ${loaded.featureExtractor}`);
  }
  if (!Array.isArray(loaded.features) || loaded.features.join('|') !== LOG_MEL_FEATURE_NAMES.join('|')) {
    throw new Error('Stage 2 model feature list does not match frontend feature extractor');
  }
  model = loaded;
  return model;
}

export function getStage2Model() {
  return model;
}

export function getStage2LoadError() {
  return loadError;
}

export function classifyShotQuality(samples, opts = {}) {
  const prepared = prepareModelClip(samples);
  if (!model) {
    return {
      available: false,
      label: 'unsure',
      predictedLabel: 'unsure',
      confidence: 0,
      threshold: null,
      probabilities: { pure: null, fat: null },
    };
  }

  const features = extractLogMelFeatures(prepared);
  let logit = model.bias;
  for (let i = 0; i < features.length; i++) {
    const z = (features[i] - model.mean[i]) / model.std[i];
    logit += model.weights[i] * z;
  }

  const pPure = sigmoid(logit);
  const pFat = 1 - pPure;
  const predictedLabel = pPure >= 0.5 ? 'pure' : 'fat';
  const confidence = Math.max(pPure, pFat);
  const threshold = opts.confidenceThreshold ?? model.confidenceThreshold ?? 0.6;
  return {
    available: true,
    label: confidence >= threshold ? predictedLabel : 'unsure',
    predictedLabel,
    confidence,
    threshold,
    probabilities: { pure: pPure, fat: pFat },
  };
}
