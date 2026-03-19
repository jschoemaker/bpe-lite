#!/usr/bin/env node
'use strict';

/**
 * Probe whitespace and repeated-char patterns against Claude API.
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/probe-whitespace-repeat.js
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 700; // conservative — 100 req/min limit

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

function our(text) { return countTokens(text, 'anthropic'); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function section(title) { console.log('\n' + '─'.repeat(60)); console.log(title); console.log('─'.repeat(60)); }

async function probe(label, text) {
  if (/^\s+$/.test(text)) { console.log(label.padEnd(28) + '  (skipped — whitespace only)'); return null; }
  const o = our(text);
  const a = await apiCount(text);
  const d = o - a;
  const flag = d !== 0 ? (d > 0 ? ' ← OVER +' + d : ' ← UNDER ' + d) : ' ✓';
  console.log(label.padEnd(28) + 'our=' + String(o).padStart(3) + '  api=' + String(a).padStart(3) + flag);
  await sleep(DELAY_MS);
  return { label, text, our: o, api: a, delta: d };
}

async function main() {
  console.log('probe-whitespace-repeat — model:', MODEL);

  // Calibrate overhead
  const raw0 = await (async () => {
    const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
      method: 'POST',
      headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: 'Hi' }] }),
    });
    return (await res.json()).input_tokens;
  })();
  overhead = raw0 - our('Hi');
  console.log('overhead:', overhead, '  (our Hi=1, apiRaw Hi=' + raw0 + ')');
  await sleep(DELAY_MS);

  // ─── Repeated spaces ───────────────────────────────────────────────────────
  section('Repeated spaces (in prose context — prefix+suffix to avoid whitespace-only rejection)');
  const PREFIX = 'x';
  const SUFFIX = 'x';
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]) {
    // Embed spaces in a context so API doesn't reject
    const inner = ' '.repeat(n);
    const full = PREFIX + inner + SUFFIX;
    const ctx = countTokens(PREFIX, 'anthropic') + countTokens(SUFFIX, 'anthropic');
    const o = our(full) - ctx;
    const a = await apiCount(full) - ctx;
    const d = o - a;
    const flag = d !== 0 ? (d > 0 ? ' ← OVER +' + d : ' ← UNDER ' + d) : ' ✓';
    console.log(('space×' + n).padEnd(28) + 'our=' + String(o).padStart(3) + '  api=' + String(a).padStart(3) + flag);
    await sleep(DELAY_MS);
  }

  section('Repeated tabs (in context)');
  for (const n of [1, 2, 3, 4, 6, 8]) {
    const full = PREFIX + '\t'.repeat(n) + SUFFIX;
    const ctx = countTokens(PREFIX, 'anthropic') + countTokens(SUFFIX, 'anthropic');
    const o = our(full) - ctx;
    const a = await apiCount(full) - ctx;
    const d = o - a;
    const flag = d !== 0 ? (d > 0 ? ' ← OVER +' + d : ' ← UNDER ' + d) : ' ✓';
    console.log(('tab×' + n).padEnd(28) + 'our=' + String(o).padStart(3) + '  api=' + String(a).padStart(3) + flag);
    await sleep(DELAY_MS);
  }

  section('Repeated newlines (in context)');
  for (const n of [1, 2, 3, 4, 6, 8]) {
    const full = PREFIX + '\n'.repeat(n) + SUFFIX;
    const ctx = countTokens(PREFIX, 'anthropic') + countTokens(SUFFIX, 'anthropic');
    const o = our(full) - ctx;
    const a = await apiCount(full) - ctx;
    const d = o - a;
    const flag = d !== 0 ? (d > 0 ? ' ← OVER +' + d : ' ← UNDER ' + d) : ' ✓';
    console.log(('nl×' + n).padEnd(28) + 'our=' + String(o).padStart(3) + '  api=' + String(a).padStart(3) + flag);
    await sleep(DELAY_MS);
  }

  // ─── Repeated letters ──────────────────────────────────────────────────────
  section('Repeated chars — verify a×N (bare)');
  for (const n of [2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 24, 32, 48, 64]) {
    await probe('a×' + n, 'a'.repeat(n));
  }

  section('Repeated chars — x (has "xxx"?)');
  for (const n of [2, 3, 4, 6, 8, 9, 12, 16]) {
    await probe('x×' + n, 'x'.repeat(n));
  }

  section('Repeated chars — uppercase');
  for (const ch of ['A', 'B', 'C', 'Z']) {
    for (const n of [2, 3, 4, 8]) {
      await probe(ch + '×' + n, ch.repeat(n));
    }
  }

  section('Repeated punctuation');
  for (const ch of ['-', '_', '.', '=', '#', '*']) {
    for (const n of [2, 3, 4, 8, 16]) {
      await probe(ch + '×' + n, ch.repeat(n));
    }
  }

  console.log('\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
