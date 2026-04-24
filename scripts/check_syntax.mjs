import { readdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CHECK_DIRS = ['frontend', 'scripts'];
const CHECK_EXTS = new Set(['.js', '.mjs']);

async function collectJsFiles(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules') continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectJsFiles(path));
    } else if (CHECK_EXTS.has(extname(entry.name))) {
      files.push(path);
    }
  }
  return files;
}

function nodeCheck(path) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--check', path], { stdio: 'inherit' });
    child.on('error', reject);
    child.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error(`node --check failed for ${relative(ROOT, path)}`));
    });
  });
}

const files = (await Promise.all(CHECK_DIRS.map(dir => collectJsFiles(join(ROOT, dir)))))
  .flat()
  .sort();

for (const file of files) {
  await nodeCheck(file);
}

console.log(`Syntax checks passed for ${files.length} JS/MJS files.`);
