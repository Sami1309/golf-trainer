import { magSpectrum, hann } from './fft.js';

export const MODEL_SAMPLE_RATE = 16000;
export const MODEL_CLIP_SAMPLES = 8000; // 500 ms at 16 kHz
export const MODEL_TARGET_DBFS = -3;
export const LOG_MEL_BANDS = 40;

export const STAGE1_FEATURE_NAMES = [
  'log_rms_all',
  'crest_factor',
  'zero_cross_rate',
  'peak_pos',
  'impact_vs_pre',
  'mid_vs_impact',
  'tail_vs_impact',
  'centroid_peak',
  'bandwidth_peak',
  'hf_ratio_peak',
  'low_ratio_peak',
  'flatness_peak',
  'centroid_mean',
  'hf_ratio_mean',
  'flux_mean',
  'flux_max',
  'peak_band_0_250',
  'peak_band_250_500',
  'peak_band_500_1000',
  'peak_band_1000_2000',
  'peak_band_2000_4000',
  'peak_band_4000_8000',
  'mean_band_0_250',
  'mean_band_250_500',
  'mean_band_500_1000',
  'mean_band_1000_2000',
  'mean_band_2000_4000',
  'mean_band_4000_8000',
];

function buildLogMelFeatureNames() {
  const names = [
    'logmel_total_mean',
    'logmel_total_std',
    'logmel_total_max',
    'logmel_total_peak_pos',
    'logmel_total_impact_delta',
    'logmel_total_tail_delta',
  ];
  for (let b = 0; b < LOG_MEL_BANDS; b++) {
    const prefix = `mel_${String(b).padStart(2, '0')}`;
    names.push(
      `${prefix}_mean`,
      `${prefix}_std`,
      `${prefix}_max`,
      `${prefix}_impact_delta`,
      `${prefix}_tail_delta`
    );
  }
  return names;
}

export const LOG_MEL_FEATURE_NAMES = buildLogMelFeatureNames();

const EPS = 1e-8;
const FFT_SIZE = 512;
const HOP_SIZE = 128;
const BAND_EDGES = [0, 250, 500, 1000, 2000, 4000, 8000];
let cachedWindow = null;
const melFilterCache = new Map();

function getWindow() {
  if (!cachedWindow) cachedWindow = hann(FFT_SIZE);
  return cachedWindow;
}

export function fitClipLength(samples, targetLength = MODEL_CLIP_SAMPLES) {
  if (samples.length === targetLength) return new Float32Array(samples);
  const out = new Float32Array(targetLength);
  if (samples.length > targetLength) {
    out.set(samples.subarray(0, targetLength));
  } else {
    out.set(samples);
  }
  return out;
}

export function peakNormalize(samples, targetDbfs = MODEL_TARGET_DBFS) {
  const out = new Float32Array(samples.length);
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  if (peak < EPS) return out;

  const targetPeak = Math.pow(10, targetDbfs / 20);
  const gain = targetPeak / peak;
  for (let i = 0; i < samples.length; i++) {
    out[i] = Math.max(-1, Math.min(1, samples[i] * gain));
  }
  return out;
}

export function prepareModelClip(samples) {
  return peakNormalize(fitClipLength(samples));
}

function rmsRange(samples, start, end) {
  const lo = Math.max(0, Math.min(samples.length, start));
  const hi = Math.max(lo + 1, Math.min(samples.length, end));
  let sum = 0;
  for (let i = lo; i < hi; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (hi - lo));
}

function zeroCrossRate(samples) {
  let crossings = 0;
  let prev = samples[0] >= 0;
  for (let i = 1; i < samples.length; i++) {
    const cur = samples[i] >= 0;
    if (cur !== prev) crossings++;
    prev = cur;
  }
  return crossings / Math.max(1, samples.length - 1);
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function getMelFilters(sampleRate, fftSize = FFT_SIZE, bands = LOG_MEL_BANDS) {
  const key = `${sampleRate}:${fftSize}:${bands}`;
  const cached = melFilterCache.get(key);
  if (cached) return cached;

  const minMel = hzToMel(40);
  const maxMel = hzToMel(sampleRate / 2);
  const melPoints = [];
  for (let i = 0; i < bands + 2; i++) {
    melPoints.push(minMel + (i / (bands + 1)) * (maxMel - minMel));
  }
  const hzPoints = melPoints.map(melToHz);
  const filters = [];

  for (let b = 0; b < bands; b++) {
    const lower = hzPoints[b];
    const center = hzPoints[b + 1];
    const upper = hzPoints[b + 2];
    const weights = new Float32Array(fftSize >> 1);
    let weightSum = 0;
    for (let k = 1; k < weights.length; k++) {
      const freq = (k * sampleRate) / fftSize;
      let weight = 0;
      if (freq >= lower && freq <= center) {
        weight = (freq - lower) / Math.max(EPS, center - lower);
      } else if (freq > center && freq <= upper) {
        weight = (upper - freq) / Math.max(EPS, upper - center);
      }
      weights[k] = Math.max(0, weight);
      weightSum += weights[k];
    }
    if (weightSum > EPS) {
      for (let k = 0; k < weights.length; k++) weights[k] /= weightSum;
    }
    filters.push(weights);
  }

  melFilterCache.set(key, filters);
  return filters;
}

function mean(values, indices = null) {
  const n = indices ? indices.length : values.length;
  if (!n) return 0;
  let sum = 0;
  if (indices) {
    for (const idx of indices) sum += values[idx];
  } else {
    for (const v of values) sum += v;
  }
  return sum / n;
}

function std(values, avg) {
  if (values.length < 2) return 0;
  let sum = 0;
  for (const v of values) {
    const d = v - avg;
    sum += d * d;
  }
  return Math.sqrt(sum / (values.length - 1));
}

function maxValue(values) {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  return max === -Infinity ? 0 : max;
}

export function extractLogMelFeatures(inputSamples, sampleRate = MODEL_SAMPLE_RATE) {
  const samples = prepareModelClip(inputSamples);
  const win = getWindow();
  const filters = getMelFilters(sampleRate);
  const magNorm = 2 / FFT_SIZE;
  const perBand = Array.from({ length: LOG_MEL_BANDS }, () => []);
  const totalLogEnergy = [];
  const frameTimes = [];

  for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) frame[i] = samples[start + i] * win[i];
    const mag = magSpectrum(frame);

    const power = new Float32Array(mag.length);
    let total = EPS;
    for (let k = 1; k < mag.length; k++) {
      const m = mag[k] * magNorm;
      const e = m * m;
      power[k] = e;
      total += e;
    }

    for (let b = 0; b < LOG_MEL_BANDS; b++) {
      const weights = filters[b];
      let energy = EPS;
      for (let k = 1; k < power.length; k++) energy += power[k] * weights[k];
      perBand[b].push(Math.log(energy));
    }
    totalLogEnergy.push(Math.log(total));
    frameTimes.push((start + FFT_SIZE / 2) / sampleRate);
  }

  const preFrames = [];
  const impactFrames = [];
  const tailFrames = [];
  let peakFrame = 0;
  let peakEnergy = -Infinity;
  for (let i = 0; i < frameTimes.length; i++) {
    const t = frameTimes[i];
    if (t < 0.08) preFrames.push(i);
    if (t >= 0.08 && t < 0.18) impactFrames.push(i);
    if (t >= 0.28) tailFrames.push(i);
    if (totalLogEnergy[i] > peakEnergy) {
      peakEnergy = totalLogEnergy[i];
      peakFrame = i;
    }
  }

  const totalMean = mean(totalLogEnergy);
  const totalImpact = mean(totalLogEnergy, impactFrames);
  const totalTail = mean(totalLogEnergy, tailFrames);
  const features = [
    totalMean,
    std(totalLogEnergy, totalMean),
    maxValue(totalLogEnergy),
    frameTimes.length > 1 ? peakFrame / (frameTimes.length - 1) : 0,
    totalImpact - mean(totalLogEnergy, preFrames),
    totalTail - totalImpact,
  ];

  for (let b = 0; b < LOG_MEL_BANDS; b++) {
    const values = perBand[b];
    const avg = mean(values);
    const impact = mean(values, impactFrames);
    const tail = mean(values, tailFrames);
    features.push(
      avg,
      std(values, avg),
      maxValue(values),
      impact - mean(values, preFrames),
      tail - impact
    );
  }

  return features;
}

export function extractStage1Features(inputSamples, sampleRate = MODEL_SAMPLE_RATE) {
  const samples = prepareModelClip(inputSamples);
  let peak = 0;
  let peakIndex = 0;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) {
      peak = abs;
      peakIndex = i;
    }
    sumSq += samples[i] * samples[i];
  }

  const rmsAll = Math.sqrt(sumSq / samples.length);
  const preRms = rmsRange(samples, 0, Math.round(0.08 * sampleRate));
  const impactRms = rmsRange(samples, Math.round(0.09 * sampleRate), Math.round(0.16 * sampleRate));
  const midRms = rmsRange(samples, Math.round(0.18 * sampleRate), Math.round(0.32 * sampleRate));
  const tailRms = rmsRange(samples, Math.round(0.35 * sampleRate), samples.length);

  const win = getWindow();
  const magNorm = 2 / FFT_SIZE;
  let prevMag = null;
  let peakFrameEnergy = -Infinity;
  let peakStats = null;
  let centroidSum = 0;
  let hfRatioSum = 0;
  const bandSums = new Array(BAND_EDGES.length - 1).fill(0);
  let frameCount = 0;
  let fluxSum = 0;
  let fluxMax = 0;

  for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) frame[i] = samples[start + i] * win[i];
    const mag = magSpectrum(frame);
    for (let i = 0; i < mag.length; i++) mag[i] *= magNorm;

    let magSum = EPS;
    let energy = EPS;
    let weightedFreq = 0;
    let logMagSum = 0;
    let lowEnergy = EPS;
    let highEnergy = EPS;
    const bands = new Array(BAND_EDGES.length - 1).fill(EPS);
    for (let k = 1; k < mag.length; k++) {
      const freq = (k * sampleRate) / FFT_SIZE;
      const m = mag[k] + EPS;
      const e = m * m;
      magSum += m;
      energy += e;
      weightedFreq += freq * m;
      logMagSum += Math.log(m);
      if (freq < 800) lowEnergy += e;
      if (freq >= 2500) highEnergy += e;
      for (let b = 0; b < BAND_EDGES.length - 1; b++) {
        if (freq >= BAND_EDGES[b] && freq < BAND_EDGES[b + 1]) {
          bands[b] += e;
          break;
        }
      }
    }

    const centroid = weightedFreq / magSum;
    let bandwidthNum = 0;
    for (let k = 1; k < mag.length; k++) {
      const freq = (k * sampleRate) / FFT_SIZE;
      const m = mag[k] + EPS;
      bandwidthNum += Math.pow(freq - centroid, 2) * m;
    }
    const bandwidth = Math.sqrt(bandwidthNum / magSum);
    const hfRatio = highEnergy / energy;
    const lowRatio = lowEnergy / energy;
    const bandRatios = bands.map(v => v / energy);
    const flatness = Math.exp(logMagSum / Math.max(1, mag.length - 1)) / (magSum / Math.max(1, mag.length - 1));

    let flux = 0;
    if (prevMag) {
      for (let k = 1; k < mag.length; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) flux += d;
      }
      fluxSum += flux;
      if (flux > fluxMax) fluxMax = flux;
    }
    prevMag = mag;

    centroidSum += centroid;
    hfRatioSum += hfRatio;
    for (let b = 0; b < bandRatios.length; b++) bandSums[b] += bandRatios[b];
    frameCount++;

    if (energy > peakFrameEnergy) {
      peakFrameEnergy = energy;
      peakStats = { centroid, bandwidth, hfRatio, lowRatio, flatness, bandRatios };
    }
  }

  peakStats ||= {
    centroid: 0,
    bandwidth: 0,
    hfRatio: 0,
    lowRatio: 0,
    flatness: 0,
    bandRatios: new Array(BAND_EDGES.length - 1).fill(0),
  };
  const denom = Math.max(1, frameCount);

  return [
    Math.log(rmsAll + EPS),
    peak / (rmsAll + EPS),
    zeroCrossRate(samples),
    peakIndex / samples.length,
    Math.log((impactRms + EPS) / (preRms + EPS)),
    Math.log((midRms + EPS) / (impactRms + EPS)),
    Math.log((tailRms + EPS) / (impactRms + EPS)),
    peakStats.centroid / 8000,
    peakStats.bandwidth / 8000,
    peakStats.hfRatio,
    peakStats.lowRatio,
    peakStats.flatness,
    (centroidSum / denom) / 8000,
    hfRatioSum / denom,
    fluxSum / denom,
    fluxMax,
    ...peakStats.bandRatios,
    ...bandSums.map(v => v / denom),
  ];
}
