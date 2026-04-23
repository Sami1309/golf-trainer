import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
export const STAGE2_PURE_FAT_EXCLUSION_POLICY_REL = 'data/stage2_pure_fat_exclusions.json';
const POLICY_PATH = join(ROOT, STAGE2_PURE_FAT_EXCLUSION_POLICY_REL);

let cachedPolicy = null;

export async function loadStage2PureFatPolicy() {
  if (cachedPolicy) return cachedPolicy;
  const raw = JSON.parse(await readFile(POLICY_PATH, 'utf8'));
  const manualByShotNumber = new Map();
  for (const item of raw.manualExclusions ?? []) {
    manualByShotNumber.set(String(item.shotNumber), item);
  }
  cachedPolicy = { ...raw, manualByShotNumber };
  return cachedPolicy;
}

export function shotNumberFromFolder(folderLabel) {
  if (!folderLabel) return null;
  const match = folderLabel.trim().match(/^(\d+)/);
  return match ? match[1] : null;
}

export function exclusionForFolder(folderLabel, policy) {
  if (!folderLabel) return { reason: 'missing_label_entry' };

  const shotNumber = shotNumberFromFolder(folderLabel);
  const manual = shotNumber ? policy.manualByShotNumber.get(shotNumber) : null;
  if (manual) {
    return {
      reason: manual.reason,
      shotNumber,
      policyPath: STAGE2_PURE_FAT_EXCLUSION_POLICY_REL,
      note: manual.note,
    };
  }

  const label = folderLabel.toLowerCase();
  if (label.includes('topped')) return { reason: 'topped_not_in_pure_vs_fat_v0', shotNumber };
  if (label.includes('1mm')) return { reason: 'borderline_fat_excluded', shotNumber };
  if (!label.includes('pure') && !label.includes('fat')) return { reason: 'unknown_folder_label', shotNumber };
  return null;
}

export function classFromFolder(folderLabel, policy) {
  if (exclusionForFolder(folderLabel, policy)) return null;
  const label = folderLabel.toLowerCase();
  if (label.includes('pure')) return 'pure';
  if (label.includes('fat')) return 'fat';
  return null;
}

export function summarizeStage2PureFatPolicy(policy) {
  return {
    path: STAGE2_PURE_FAT_EXCLUSION_POLICY_REL,
    version: policy.version,
    updatedAt: policy.updatedAt,
    manualExclusions: (policy.manualExclusions ?? []).map(item => ({
      shotNumber: String(item.shotNumber),
      reason: item.reason,
    })),
    builtInReasons: (policy.builtInExclusionRules ?? []).map(item => item.reason),
  };
}
