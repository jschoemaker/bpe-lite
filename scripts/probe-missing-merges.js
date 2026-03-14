#!/usr/bin/env node
'use strict';

/**
 * Targeted merge-gap probe: finds which BPE merges Claude has that our Xenova
 * vocab lacks, by probing adjacent token pairs against the count_tokens API.
 *
 * Strategy:
 *   1. Tokenize the sample text using our tokenizer.
 *   2. For each adjacent pair of tokens, probe (left + right) against the API.
 *   3. If api_count < our_count for that pair, Claude can merge them further.
 *   4. Recursively probe the merged result to find deeper merges.
 *   5. Output the discovered missing tokens as base64 keys + suggested ranks.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... node scripts/probe-missing-merges.js
 */

const { encode, countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 120;

let overhead = 0;

async function apiCount(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: text }] }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return (await res.json()).input_tokens - overhead;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Decode token IDs back to their UTF-8 strings using bpe-lite internals
function tokensToStrings(ids) {
  const { anthropic } = require('../src/index');
  const tok = anthropic();
  return ids.map(id => tok.decode([id]));
}

async function calibrateOverhead() {
  const raw = await apiCount('Hi');
  // 'Hi' = 1 token in our tokenizer. overhead already subtracted via apiCount wrapping
  // Re-calibrate: raw returned already has overhead subtracted (apiCount subtracts overhead=0 here)
  // Actually overhead starts at 0, so raw = api_raw. We want overhead = api_raw - our_count
  const ourCount = countTokens('Hi', 'anthropic');
  // apiCount already subtracts overhead(=0 here), so raw = api_raw - 0 = api_raw
  overhead = raw - ourCount;
  console.log(`Overhead calibrated: ${overhead} tokens (api_raw=${ raw + overhead }, our=Hi=${ourCount})`);
  await sleep(DELAY_MS);
}

// Probe a text fragment. Returns api token count for just that fragment.
const probeCache = new Map();
async function probe(text) {
  if (probeCache.has(text)) return probeCache.get(text);
  if (/^\s+$/.test(text)) { probeCache.set(text, 0); return 0; }
  const count = await apiCount(text);
  probeCache.set(text, count);
  await sleep(DELAY_MS);
  return count;
}

// Given a string, find all pairs of adjacent bytes that Claude merges
// differently from our tokenizer. Returns list of {text, ourTokens, apiTokens}.
async function findMissingMerges(text, label) {
  console.log(`\n=== ${label} ===`);
  console.log(`Text: ${JSON.stringify(text)}`);

  const ourIds = encode(text, 'anthropic');
  const { anthropic } = require('../src/index');
  const tok = anthropic();
  const ourStrings = ourIds.map(id => tok.decode([id]));

  const apiTotal = await probe(text);
  const ourTotal = ourIds.length;

  console.log(`Our tokens (${ourTotal}): ${ourStrings.map(s => JSON.stringify(s)).join(' ')}`);
  console.log(`API total: ${apiTotal}, Our total: ${ourTotal}, Gap: ${ourTotal - apiTotal}`);

  if (ourTotal <= apiTotal) {
    console.log('  No gap — skipping.');
    return [];
  }

  // Probe adjacent pairs from our tokenization to find where Claude merges differently
  const missing = [];
  const toProbe = [...ourStrings]; // start with our token boundaries

  // Sliding window: probe all adjacent pairs
  console.log('\nProbing adjacent token pairs:');
  for (let i = 0; i < toProbe.length - 1; i++) {
    const pair = toProbe[i] + toProbe[i + 1];
    const pairApi = await probe(pair);
    const pairOur = 2; // always 2 in our tokenization (they're separate tokens)

    if (pairApi < pairOur) {
      console.log(`  MERGE: ${JSON.stringify(toProbe[i])} + ${JSON.stringify(toProbe[i+1])} → api=${pairApi} (we give 2)`);
      const b64 = Buffer.from(pair, 'utf8').toString('base64');
      missing.push({ text: pair, b64, ourCount: pairOur, apiCount: pairApi });

      // Also probe triple (if the merged pair can merge further with neighbor)
      if (pairApi === 1 && i + 2 < toProbe.length) {
        const triple = pair + toProbe[i + 2];
        const tripleApi = await probe(triple);
        if (tripleApi < pairApi + 1) {
          console.log(`    TRIPLE MERGE: ${JSON.stringify(triple)} → api=${tripleApi}`);
          const b64t = Buffer.from(triple, 'utf8').toString('base64');
          missing.push({ text: triple, b64: b64t, ourCount: 3, apiCount: tripleApi });
        }
      }
    } else {
      console.log(`  ok:    ${JSON.stringify(toProbe[i])} + ${JSON.stringify(toProbe[i+1])} → api=${pairApi}`);
    }
  }

  return missing;
}

// Also probe individual characters that might be 1 token in Claude but not us
async function probeIndividualChars(text, label) {
  const chars = [...new Set([...text].filter(c => c.trim()))];
  const missing = [];
  console.log(`\n--- Individual chars in ${label} ---`);
  for (const ch of chars) {
    const cp = ch.codePointAt(0);
    if (cp < 128) continue; // ASCII — skip
    const ourCount = countTokens(ch, 'anthropic');
    const apiCnt = await probe(ch);
    const flag = ourCount !== apiCnt ? ` ← MISMATCH our=${ourCount}` : ' ✓';
    console.log(`  ${JSON.stringify(ch)} U+${cp.toString(16).toUpperCase()} our=${ourCount} api=${apiCnt}${flag}`);
    if (apiCnt < ourCount) {
      missing.push({ text: ch, b64: Buffer.from(ch, 'utf8').toString('base64'), ourCount, apiCount: apiCnt });
    }
  }
  return missing;
}

const SAMPLES = [
  {
    label: 'Arabic',
    text: 'يغير الذكاء الاصطناعي العالم. يمكن لنماذج التعلم الآلي فهم اللغة الطبيعية وتوليدها.',
  },
  {
    label: 'Japanese',
    text: '人工知能は世界を変えています。機械学習モデルは自然言語を理解し生成することができます。',
  },
  {
    label: 'Chinese',
    text: '人工智能正在改变世界。机器学习模型可以理解和生成自然语言。深度学习是人工智能的重要分支。',
  },
];

async function main() {
  console.log(`probe-missing-merges — model: ${MODEL}`);

  // Calibrate overhead
  const rawHi = await (async () => {
    const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
      method: 'POST',
      headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: 'Hi' }] }),
    });
    return (await res.json()).input_tokens;
  })();
  overhead = rawHi - countTokens('Hi', 'anthropic');
  console.log(`Overhead: ${overhead} (rawHi=${rawHi})`);
  await sleep(DELAY_MS);

  const allMissing = [];

  for (const { label, text } of SAMPLES) {
    const charMissing = await probeIndividualChars(text, label);
    const pairMissing = await findMissingMerges(text, label);
    allMissing.push(...charMissing, ...pairMissing);
  }

  // Deduplicate
  const seen = new Set();
  const unique = allMissing.filter(m => {
    if (seen.has(m.b64)) return false;
    seen.add(m.b64);
    return true;
  });

  if (unique.length === 0) {
    console.log('\n\nNo missing merges found — vocab is already aligned.');
    return;
  }

  console.log('\n\n=== MISSING MERGES SUMMARY ===');
  console.log('Add these to build-vocabs.js byteOnlyChars (if api=byte-level) or as injected merges:\n');
  for (const m of unique) {
    const bytes = Buffer.from(m.b64, 'base64');
    const hex = [...bytes].map(b => b.toString(16).padStart(2, '0').toUpperCase()).join(' ');
    console.log(`text=${JSON.stringify(m.text)}  api=${m.apiCount}  our=${m.ourCount}  bytes=[${hex}]  b64=${m.b64}`);
  }

  console.log('\n\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
