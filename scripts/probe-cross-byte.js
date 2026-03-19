#!/usr/bin/env node
'use strict';

/**
 * Probe cross-symbol byte merges in Claude's tokenizer.
 *
 * Claude uses byte-level BPE without regex pre-tokenization. When two symbols
 * are adjacent, the last byte of S1 and first byte of S2 can merge. This script
 * systematically measures which cross-boundary byte pairs Claude has in its vocab
 * by comparing api(S1S2) vs api(S1) + api(S2) - overhead.
 *
 * For our [E2,XX,YY] symbols: first byte of S2 is always 0xE2, so we're mapping
 * which [YY, E2] byte pairs Claude merges. For [C2,XX] symbols the last byte XX
 * can merge with the following 0xE2 or 0xC2.
 *
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/probe-cross-byte.js
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 700;

let overhead = 0;

async function rawApi(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: text }] }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return (await res.json()).input_tokens;
}

async function api(text) { return (await rawApi(text)) - overhead; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function hex(b) { return b.toString(16).toUpperCase().padStart(2, '0'); }
function section(t) { console.log('\n' + '─'.repeat(70)); console.log(t); console.log('─'.repeat(70)); }

// All symbols with their byte sequences
const SYMBOLS = [
  // 2-byte [C2,xx]
  ['©', 'copyright',   [0xC2, 0xA9]],
  ['®', 'registered',  [0xC2, 0xAE]],
  ['±', 'plus-minus',  [0xC2, 0xB1]],
  ['×', 'times',       [0xC3, 0x97]], // C3!
  ['£', 'pound',       [0xC2, 0xA3]],
  ['¥', 'yen',         [0xC2, 0xA5]],
  ['¢', 'cent',        [0xC2, 0xA2]],
  // 3-byte [E2,80,xx] — E2-80 block
  ['•', 'bullet',      [0xE2, 0x80, 0xA2]],
  ['…', 'ellipsis',    [0xE2, 0x80, 0xA6]], // NFKC→... but original bytes
  ['†', 'dagger',      [0xE2, 0x80, 0xA0]],
  ['‡', 'dagger2',     [0xE2, 0x80, 0xA1]],
  ['−', 'minus',       [0xE2, 0x88, 0x92]], // E2-88 block
  // 3-byte [E2,82,xx] — currency
  ['€', 'euro',        [0xE2, 0x82, 0xAC]],
  ['₿', 'bitcoin',     [0xE2, 0x82, 0xBF]],
  ['₹', 'rupee',       [0xE2, 0x82, 0xB9]],
  ['₩', 'won',         [0xE2, 0x82, 0xA9]],
  ['₽', 'ruble',       [0xE2, 0x82, 0xBD]],
  // 3-byte [E2,86,xx] — arrows
  ['←', 'left-arrow',  [0xE2, 0x86, 0x90]],
  ['→', 'right-arrow', [0xE2, 0x86, 0x92]],
  ['↑', 'up-arrow',    [0xE2, 0x86, 0x91]],
  ['↓', 'down-arrow',  [0xE2, 0x86, 0x93]],
  ['↔', 'lr-arrow',    [0xE2, 0x86, 0x94]],
  // 3-byte [E2,88,xx] — math
  ['√', 'sqrt',        [0xE2, 0x88, 0x9A]],
  ['∞', 'infinity',    [0xE2, 0x88, 0x9E]],
  ['∑', 'sum',         [0xE2, 0x88, 0x91]],
  ['∫', 'integral',    [0xE2, 0x88, 0xAB]],
  // 3-byte [E2,89,xx] — relations
  ['≠', 'neq',         [0xE2, 0x89, 0xA0]],
  ['≤', 'leq',         [0xE2, 0x89, 0xA4]],
  ['≥', 'geq',         [0xE2, 0x89, 0xA5]],
  ['≈', 'approx',      [0xE2, 0x89, 0x88]],
];

// Measure individual costs once, cache them
const individualCost = new Map();

async function getIndividualCost(sym, bytes) {
  if (individualCost.has(sym)) return individualCost.get(sym);
  const cost = await api(sym);
  await sleep(DELAY_MS);
  individualCost.set(sym, cost);
  return cost;
}

// Probe a pair: returns { full, c1, c2, delta, crossByte }
async function probePair(s1, bytes1, s2, bytes2) {
  const full = await api(s1 + s2);
  await sleep(DELAY_MS);
  const c1 = await getIndividualCost(s1, bytes1);
  const c2 = await getIndividualCost(s2, bytes2);
  const delta = full - c1 - c2; // negative = cross-byte merges save tokens
  const crossByte = `[${hex(bytes1[bytes1.length - 1])},${hex(bytes2[0])}]`;
  return { full, c1, c2, delta, crossByte };
}

async function main() {
  console.log('probe-cross-byte — model:', MODEL);

  // Calibrate overhead
  const raw0 = await rawApi('Hi');
  overhead = raw0 - countTokens('Hi', 'anthropic');
  console.log(`overhead: ${overhead}`);
  await sleep(DELAY_MS);

  // Pre-fetch all individual costs
  section('Individual symbol costs (api, bare)');
  for (const [sym, name, bytes] of SYMBOLS) {
    const cost = await getIndividualCost(sym, bytes);
    const bpe = countTokens(sym, 'anthropic');
    const match = cost === bpe ? '✓' : `✗ bpe=${bpe}`;
    const byteStr = bytes.map(hex).join(',');
    console.log(`${sym} ${name.padEnd(14)} [${byteStr}]  api=${cost} ${match}`);
  }

  // ─── Part 1: All adjacent pairs in the accuracy-test "symbols" string ─────────
  section('Part 1: Adjacent pairs in the "symbols" accuracy string');
  console.log('String: © ® ™ § ¶ † ‡ • … ← → ↑ ↓ ↔ ≠ ≤ ≥ ± × ÷ √ ∞ ∑ ∏ ∫');
  console.log('Probing each adjacent pair (with space stripped) to find cross-byte merges.\n');

  // The symbols in order from the accuracy test (™→TM/2 and …→.../1 via NFKC, skip for raw bytes)
  const accuracySymbols = [
    ['©', [0xC2, 0xA9]], ['®', [0xC2, 0xAE]], ['™', [0xE2, 0x84, 0xA2]],
    ['§', [0xC2, 0xA7]], ['¶', [0xC2, 0xB6]], ['†', [0xE2, 0x80, 0xA0]],
    ['‡', [0xE2, 0x80, 0xA1]], ['•', [0xE2, 0x80, 0xA2]], ['…', [0xE2, 0x80, 0xA6]],
    ['←', [0xE2, 0x86, 0x90]], ['→', [0xE2, 0x86, 0x92]], ['↑', [0xE2, 0x86, 0x91]],
    ['↓', [0xE2, 0x86, 0x93]], ['↔', [0xE2, 0x86, 0x94]], ['≠', [0xE2, 0x89, 0xA0]],
    ['≤', [0xE2, 0x89, 0xA4]], ['≥', [0xE2, 0x89, 0xA5]], ['±', [0xC2, 0xB1]],
    ['×', [0xC3, 0x97]], ['÷', [0xC3, 0xB7]], ['√', [0xE2, 0x88, 0x9A]],
    ['∞', [0xE2, 0x88, 0x9E]], ['∑', [0xE2, 0x88, 0x91]], ['∏', [0xE2, 0x88, 0x8F]],
    ['∫', [0xE2, 0x88, 0xAB]],
  ];

  // Pre-fetch individual costs for accuracy symbols
  for (const [sym, bytes] of accuracySymbols) {
    if (!individualCost.has(sym)) {
      individualCost.set(sym, await api(sym));
      await sleep(DELAY_MS);
    }
  }

  let totalPairSavings = 0;
  const mergesFound = new Map(); // crossByte → count

  console.log('pair            cross-byte   full  sum  delta');
  console.log('-'.repeat(48));
  for (let i = 0; i < accuracySymbols.length - 1; i++) {
    const [s1, b1] = accuracySymbols[i];
    const [s2, b2] = accuracySymbols[i + 1];
    const full = await api(s1 + s2); await sleep(DELAY_MS);
    const c1 = individualCost.get(s1);
    const c2 = individualCost.get(s2);
    const delta = full - c1 - c2;
    const crossByte = `[${hex(b1[b1.length - 1])},${hex(b2[0])}]`;
    if (delta !== 0) {
      totalPairSavings += delta;
      mergesFound.set(crossByte, (mergesFound.get(crossByte) || 0) + 1);
    }
    const flag = delta < 0 ? ` ← MERGE saves ${-delta}` : delta > 0 ? ` ← OVER +${delta}` : ' ✓ no merge';
    console.log(`${s1}+${s2}  ${crossByte.padEnd(12)}  ${full}     ${c1+c2}   ${delta}${flag}`);
  }
  console.log(`\nTotal cross-boundary savings: ${-totalPairSavings} tokens`);

  // ─── Part 2: Systematic cross-byte mapping ────────────────────────────────────
  section('Part 2: Cross-byte pair mapping [last_byte_of_S1, first_byte_of_S2]');
  console.log('Testing which [XX, first_byte] pairs Claude has in vocab.');
  console.log('Format: S1+S2 → cross=[XX,YY] delta (negative=merge)\n');

  // Group symbols by first byte of their UTF-8 encoding
  const byFirst = {};
  for (const [sym, , bytes] of SYMBOLS) {
    const fb = hex(bytes[0]);
    if (!byFirst[fb]) byFirst[fb] = [];
    byFirst[fb].push([sym, bytes]);
  }

  // For each symbol as S1, test against one representative of each first-byte group
  const crossResults = []; // { s1, s2, crossByte, delta }

  // Test specifically: last_byte_of_E2_symbols → E2 (the most important cross)
  console.log('Last byte of [E2,XX,YY] symbols → first byte 0xE2:');
  const e2Symbols = SYMBOLS.filter(([, , b]) => b[0] === 0xE2);
  // Use → as a fixed S2 (1 token, well-behaved) to probe last-byte merges
  const anchorS2 = '→';
  const anchorB2 = [0xE2, 0x86, 0x92];
  const anchorCost = individualCost.get('→') ?? (await api('→'));
  for (const [sym, name, bytes] of e2Symbols) {
    if (sym === anchorS2) continue;
    const full = await api(sym + anchorS2); await sleep(DELAY_MS);
    const c1 = individualCost.get(sym) ?? (await api(sym));
    const delta = full - c1 - anchorCost;
    const lastByte = hex(bytes[bytes.length - 1]);
    const flag = delta < 0 ? ` ← [${lastByte},E2] MERGES, saves ${-delta}` : ' no merge';
    console.log(`${sym}+→  [${lastByte},E2]  full=${full} c1=${c1} c2=${anchorCost} delta=${delta}${flag}`);
    if (delta !== 0) crossResults.push({ s1: sym, s2: anchorS2, crossByte: `[${lastByte},E2]`, delta });
  }

  // Test: last byte of C2/C3 symbols → E2 first byte
  console.log('\nLast byte of [C2,xx] and [C3,xx] symbols → first byte 0xE2:');
  const c2c3Symbols = SYMBOLS.filter(([, , b]) => b[0] === 0xC2 || b[0] === 0xC3);
  for (const [sym, name, bytes] of c2c3Symbols) {
    const full = await api(sym + anchorS2); await sleep(DELAY_MS);
    const c1 = individualCost.get(sym) ?? (await api(sym));
    const delta = full - c1 - anchorCost;
    const lastByte = hex(bytes[bytes.length - 1]);
    const flag = delta < 0 ? ` ← [${lastByte},E2] MERGES, saves ${-delta}` : ' no merge';
    console.log(`${sym}+→  [${lastByte},E2]  full=${full} c1=${c1} c2=${anchorCost} delta=${delta}${flag}`);
    if (delta !== 0) crossResults.push({ s1: sym, s2: anchorS2, crossByte: `[${lastByte},E2]`, delta });
  }

  // Test: last byte of E2 symbols → first byte 0xC2
  console.log('\nLast byte of [E2,xx,yy] symbols → first byte 0xC2 (using © as anchor):');
  const anchorC2 = '©';
  const anchorC2Cost = individualCost.get('©') ?? (await api('©'));
  for (const [sym, name, bytes] of e2Symbols) {
    if (sym === anchorC2) continue;
    const full = await api(sym + anchorC2); await sleep(DELAY_MS);
    const c1 = individualCost.get(sym) ?? (await api(sym));
    const delta = full - c1 - anchorC2Cost;
    const lastByte = hex(bytes[bytes.length - 1]);
    const flag = delta < 0 ? ` ← [${lastByte},C2] MERGES, saves ${-delta}` : ' no merge';
    console.log(`${sym}+©  [${lastByte},C2]  full=${full} c1=${c1} c2=${anchorC2Cost} delta=${delta}${flag}`);
    if (delta !== 0) crossResults.push({ s1: sym, s2: anchorC2, crossByte: `[${lastByte},C2]`, delta });
  }

  // ─── Summary ─────────────────────────────────────────────────────────────────
  section('Summary: cross-byte pairs in Claude vocab');
  if (crossResults.length === 0) {
    console.log('No cross-boundary merges found in tested pairs.');
  } else {
    console.log('Byte pairs that appear to be in Claude BPE vocab:');
    const seen = new Set();
    for (const { s1, s2, crossByte, delta } of crossResults) {
      if (!seen.has(crossByte)) {
        seen.add(crossByte);
        console.log(`  ${crossByte}  (found via ${s1}+${s2}, delta=${delta})`);
      }
    }
    console.log(`\nTotal: ${seen.size} cross-byte pairs found.`);
    console.log('These represent byte-level merges in Claude BPE that span symbol boundaries.');
    console.log('Difficult to replicate in bpe-lite (regex pre-tokenization chunks symbols separately).');
  }

  console.log('\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
