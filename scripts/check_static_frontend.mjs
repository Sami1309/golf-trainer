import { access, readFile } from 'node:fs/promises';
import { dirname, join, normalize, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  LOG_MEL_FEATURE_NAMES,
  STAGE1_FEATURE_NAMES,
} from '../frontend/audio_features.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const FRONTEND = join(ROOT, 'frontend');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function mustExist(relPath) {
  await access(join(ROOT, relPath));
}

function isFiniteNumberArray(value, expectedLength) {
  return Array.isArray(value) &&
    value.length === expectedLength &&
    value.every(n => typeof n === 'number' && Number.isFinite(n));
}

async function readJson(relPath) {
  return JSON.parse(await readFile(join(ROOT, relPath), 'utf8'));
}

async function checkModel(relPath, expectedExtractor, expectedFeatures, thresholdKey) {
  const model = await readJson(relPath);
  const count = expectedFeatures.length;

  assert(model.featureExtractor === expectedExtractor,
    `${relPath}: expected featureExtractor ${expectedExtractor}, got ${model.featureExtractor}`);
  assert(Array.isArray(model.features) && model.features.join('|') === expectedFeatures.join('|'),
    `${relPath}: feature list does not match frontend extractor`);
  assert(typeof model.bias === 'number' && Number.isFinite(model.bias),
    `${relPath}: bias must be a finite number`);
  assert(isFiniteNumberArray(model.weights, count),
    `${relPath}: weights must contain ${count} finite numbers`);
  assert(isFiniteNumberArray(model.mean, count),
    `${relPath}: mean must contain ${count} finite numbers`);
  assert(isFiniteNumberArray(model.std, count),
    `${relPath}: std must contain ${count} finite numbers`);
  assert(model.std.every(n => n > 0),
    `${relPath}: std values must be greater than zero`);
  assert(typeof model[thresholdKey] === 'number' && Number.isFinite(model[thresholdKey]),
    `${relPath}: ${thresholdKey} must be a finite number`);

  return { relPath, featureExtractor: model.featureExtractor, featureCount: count };
}

async function checkRelativeImports(relPath) {
  const absPath = join(ROOT, relPath);
  const source = await readFile(absPath, 'utf8');
  const importerDir = dirname(absPath);
  const importPattern = /from\s+['"](\.\/[^'"]+)['"]/g;
  const imports = [...source.matchAll(importPattern)].map(match => match[1]);

  for (const imported of imports) {
    const target = normalize(join(importerDir, imported));
    const targetRel = relative(ROOT, target);
    assert(!targetRel.startsWith('..') && !targetRel.startsWith('/'),
      `${relPath}: import escapes repo root: ${imported}`);
    await access(target);
  }
}

const requiredFiles = [
  'frontend/index.html',
  'frontend/style.css',
  'frontend/app.js',
  'frontend/audio_features.js',
  'frontend/fft.js',
  'frontend/wav.js',
  'frontend/stage1b.js',
  'frontend/stage2.js',
  'frontend/shot_store.js',
  'frontend/onset-worklet.js',
  'frontend/models/stage1b_detector.json',
  'frontend/models/stage2_pure_fat.json',
];

await Promise.all(requiredFiles.map(mustExist));

const html = await readFile(join(FRONTEND, 'index.html'), 'utf8');
assert(html.includes('src="app.js"'), 'frontend/index.html must load app.js');
assert(html.includes('href="style.css"'), 'frontend/index.html must load style.css');
assert(html.includes('jszip'), 'frontend/index.html must load JSZip for ZIP export');

await Promise.all([
  checkRelativeImports('frontend/app.js'),
  checkRelativeImports('frontend/stage1b.js'),
  checkRelativeImports('frontend/stage2.js'),
  checkRelativeImports('frontend/audio_features.js'),
]);

const checkedModels = await Promise.all([
  checkModel('frontend/models/stage1b_detector.json', 'logmel_summary', LOG_MEL_FEATURE_NAMES, 'threshold'),
  checkModel('frontend/models/stage1b_logmel.json', 'logmel_summary', LOG_MEL_FEATURE_NAMES, 'threshold'),
  checkModel('frontend/models/stage1b_handcrafted.json', 'stage1_handcrafted', STAGE1_FEATURE_NAMES, 'threshold'),
  checkModel('frontend/models/stage2_pure_fat.json', 'logmel_summary', LOG_MEL_FEATURE_NAMES, 'confidenceThreshold'),
]);

for (const model of checkedModels) {
  console.log(`ok ${model.relPath}: ${model.featureExtractor}, ${model.featureCount} features`);
}
console.log('Static frontend checks passed.');
