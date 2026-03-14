#!/usr/bin/env node
'use strict';
/**
 * Targeted probe for accuracy outliers.
 * Uses differential probe: tokens(X) = apiRaw(prefix + X + suffix) - apiRaw(prefix + suffix)
 * to cleanly isolate token counts for any fragment.
 *
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/probe-outliers.js
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 160;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function apiRaw(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: text }] }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return (await res.json()).input_tokens;
}

// Differential probe: tokens(text) = apiRaw(prefix + text + suffix) - base
// where base = apiRaw(prefix + suffix), computed once per prefix/suffix pair.
// Prefix/suffix should produce enough tokens that overhead is stable.
const PREFIX = 'The answer is';   // consistent left context (3 tokens approx)
const SUFFIX = 'and nothing else.'; // consistent right context (4 tokens approx)
let base = -1;

async function init() {
  base = await apiRaw(PREFIX + SUFFIX);
  console.log(`Differential base: ${base} raw tokens for "${PREFIX}${SUFFIX}"`);
  await sleep(DELAY_MS);
}

const cache = new Map();
async function probe(text) {
  if (cache.has(text)) return cache.get(text);
  const raw = await apiRaw(PREFIX + text + SUFFIX);
  const result = raw - base;
  cache.set(text, result);
  await sleep(DELAY_MS);
  return result;
}

function our(text) { return countTokens(text, 'anthropic'); }

async function section(title) {
  console.log('\n' + '='.repeat(55));
  console.log(title);
  console.log('='.repeat(55));
}

function fmt(label, o, a) {
  const d = a - o;
  const flag = d > 0 ? ' UNDER' : d < 0 ? ' OVER' : '';
  return `${label.padEnd(22)} our=${o}  api=${a}  delta=${d >= 0 ? '+' : ''}${d}${flag}`;
}

// ─── Emoji ───────────────────────────────────────────────────────────────────
async function probeEmoji() {
  await section('EMOJI  (our=73, api=61)');
  const emojis = [
    '😀', ' 😂', ' 🤔', ' 🎉', ' 🚀', ' 🌍', ' 🔥', ' 💡',
    ' ❤️', ' 🐍', ' 🦊', ' 🌈', ' ⭐', ' 🎵', ' 🏆',
  ];
  let ourTotal = 0, apiTotal = 0;
  for (const e of emojis) {
    const o = our(e);
    const a = await probe(e);
    ourTotal += o; apiTotal += a;
    console.log(fmt(JSON.stringify(e), o, a));
  }
  console.log(`${'TOTAL'.padEnd(22)} our=${ourTotal}  api=${apiTotal}  delta=${apiTotal - ourTotal}`);
}

// ─── Whitespace ───────────────────────────────────────────────────────────────
async function probeWhitespace() {
  await section('WHITESPACE  (our=19, api=16)');
  // Test the exact sample chunks
  const sampleChunks = ['word1', '     ', 'word2', '\t\t', 'word3', '\n\n\n', 'word4', '   \t  ', 'word5'];
  let ourS = 0, apiS = 0;
  for (const c of sampleChunks) {
    const o = our(c);
    const a = await probe(c);
    ourS += o; apiS += a;
    console.log(fmt(JSON.stringify(c), o, a));
  }
  console.log(`${'sample total'.padEnd(22)} our=${ourS}  api=${apiS}`);

  // Extra whitespace patterns for diagnosis
  console.log('\n-- extra patterns --');
  for (const c of ['  ', '   ', '    ', '     ', '      ', '\t', '\t\t\t', '\n', '\n\n']) {
    const o = our(c);
    const a = await probe(c);
    console.log(fmt(JSON.stringify(c), o, a));
  }
}

// ─── Currency ────────────────────────────────────────────────────────────────
async function probeCurrency() {
  await section('CURRENCY  (our=44, api=50)');
  // Full sample chunks
  const chunks = ['$100.00', ' €85.50', ' £72.30', ' ¥12,000', ' ₿0.00234', ' ₹8,250',
                  ' CHF', ' 95.00', ' CAD', ' 135.00'];
  let ourT = 0, apiT = 0;
  for (const c of chunks) {
    const o = our(c);
    const a = await probe(c);
    ourT += o; apiT += a;
    console.log(fmt(JSON.stringify(c), o, a));
  }
  console.log(`${'total'.padEnd(22)} our=${ourT}  api=${apiT}  delta=${apiT - ourT}`);

  console.log('\n-- individual symbols --');
  for (const s of ['$', ' €', ' £', ' ¥', ' ₿', ' ₹', '€', '£', '¥', '₿', '₹']) {
    const o = our(s);
    const a = await probe(s);
    console.log(fmt(JSON.stringify(s), o, a));
  }
}

// ─── Repeated chars ──────────────────────────────────────────────────────────
async function probeRepeated() {
  await section('REPEATED CHARS  (our=32, api=23)');
  const full = 'a'.repeat(64);
  {
    const o = our(full);
    const a = await probe(full);
    console.log(fmt('a×64', o, a));
  }

  console.log('\n-- length sweep (no space prefix) --');
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32]) {
    const text = 'a'.repeat(n);
    const o = our(text);
    const a = await probe(text);
    const d = a - o;
    const flag = d !== 0 ? (d > 0 ? ' UNDER' : ' OVER') : '';
    console.log(`  a×${String(n).padEnd(4)} our=${o}  api=${a}  bytes/tok=${(n/a).toFixed(2)}${flag}`);
  }
}

// ─── Arabic ──────────────────────────────────────────────────────────────────
async function probeArabic() {
  await section('ARABIC  (our=67, api=56)');
  const text = 'يغير الذكاء الاصطناعي العالم. يمكن لنماذج التعلم الآلي فهم اللغة الطبيعية وتوليدها.';
  {
    const o = our(text);
    const a = await probe(text);
    console.log(fmt('full text', o, a));
  }
  // Probe individual words
  console.log('\n-- per word --');
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const o = our(w);
    const a = await probe(w);
    console.log(fmt(w.padEnd(16), o, a));
  }
}

async function main() {
  console.log(`probe-outliers — model: ${MODEL}`);
  await init();
  await probeEmoji();
  await probeWhitespace();
  await probeCurrency();
  await probeRepeated();
  await probeArabic();
  console.log('\n\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
