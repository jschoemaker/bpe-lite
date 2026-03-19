#!/usr/bin/env node
'use strict';

/**
 * Context-aware symbol/currency probe for Claude tokenizer reverse-engineering.
 *
 * Instead of probing symbols in isolation (misleading — Claude likely does byte-level
 * BPE without regex pre-tokenization), this script uses three approaches:
 *
 *   1. Differential context probing — measures the marginal token cost of each symbol
 *      embedded in 4 controlled contexts (letter, digit, space, word surroundings).
 *
 *   2. Pre-tokenization boundary test — checks whether Claude respects symbol/letter
 *      boundaries (like OpenAI's regex) or does pure byte-level BPE across them.
 *
 *   3. Concatenated symbol probes — checks whether adjacent symbols produce cross-symbol
 *      byte merges (lower total than sum of individuals).
 *
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/probe-symbol-context.js
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 700; // conservative — 100 req/min limit

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

async function api(text) {
  return (await rawApi(text)) - overhead;
}

function our(text) { return countTokens(text, 'anthropic'); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function section(title) { console.log('\n' + '─'.repeat(70)); console.log(title); console.log('─'.repeat(70)); }
function pad(s, w) { s = String(s); return s.length >= w ? s : s + ' '.repeat(w - s.length); }
function padL(s, w) { s = String(s); return s.length >= w ? s : ' '.repeat(w - s.length) + s; }

const SYMBOLS = [
  // Currency
  ['€', 'euro'],
  ['£', 'pound'],
  ['¥', 'yen'],
  ['₿', 'bitcoin'],
  ['₹', 'rupee'],
  ['¢', 'cent'],
  ['₩', 'won'],
  ['₽', 'ruble'],
  // IP / legal
  ['©', 'copyright'],
  ['®', 'registered'],
  ['™', 'trademark'],
  // Arrows
  ['→', 'right-arrow'],
  ['←', 'left-arrow'],
  ['↑', 'up-arrow'],
  ['↓', 'down-arrow'],
  ['↔', 'lr-arrow'],
  // Math
  ['≠', 'neq'],
  ['≤', 'leq'],
  ['≥', 'geq'],
  ['≈', 'approx'],
  ['±', 'plus-minus'],
  ['×', 'times'],
  ['÷', 'divide'],
  ['√', 'sqrt'],
  ['∞', 'infinity'],
  ['∑', 'sum'],
  ['∫', 'integral'],
  // Punctuation
  ['•', 'bullet'],
  ['…', 'ellipsis'],
  ['†', 'dagger'],
  ['‡', 'dagger2'],
  ['§', 'section'],
  ['¶', 'pilcrow'],
  ['−', 'minus-sign'],
];

// Concatenation groups for approach 3
const CONCAT_GROUPS = [
  { label: 'currency "€£¥₿"', chars: '€£¥₿' },
  { label: 'currency "₹¢₩₽"', chars: '₹¢₩₽' },
  { label: 'arrows "→←↑↓"', chars: '→←↑↓' },
  { label: 'legal "©®™"', chars: '©®™' },
  { label: 'math "≠≤≥≈"', chars: '≠≤≥≈' },
  { label: 'math "±×÷√"', chars: '±×÷√' },
  { label: 'math "∞∑∫"', chars: '∞∑∫' },
  { label: 'punct "•…†‡"', chars: '•…†‡' },
];

// Boundary test pairs
const BOUNDARY_TESTS = [
  { label: '"word€word"', full: 'word€word', parts: ['word', '€', 'word'] },
  { label: '"word£word"', full: 'word£word', parts: ['word', '£', 'word'] },
  { label: '"word→word"', full: 'word→word', parts: ['word', '→', 'word'] },
  { label: '"word©word"', full: 'word©word', parts: ['word', '©', 'word'] },
  { label: '"word×word"', full: 'word×word', parts: ['word', '×', 'word'] },
  { label: '"a€b"',       full: 'a€b',       parts: ['a', '€', 'b'] },
  { label: '"1€2"',       full: '1€2',        parts: ['1', '€', '2'] },
  { label: '"€£"',        full: '€£',         parts: ['€', '£'] },
  { label: '"©®"',        full: '©®',         parts: ['©', '®'] },
  { label: '"→←"',        full: '→←',         parts: ['→', '←'] },
];

async function main() {
  console.log('probe-symbol-context — model:', MODEL);
  console.log('Calibrating overhead...');

  const raw0 = await rawApi('Hi');
  overhead = raw0 - our('Hi');
  console.log(`overhead: ${overhead}  (our Hi=${our('Hi')}, apiRaw Hi=${raw0})`);
  await sleep(DELAY_MS);

  // ─── Approach 1: Differential context probing ────────────────────────────────
  section('Approach 1: Differential context probing');
  console.log('Fetching anchor token counts...');

  const anchorLetterBase = await api('aa');       await sleep(DELAY_MS);
  const anchorDigitBase  = await api('11');       await sleep(DELAY_MS);
  const anchorSpaceBase  = await api('a  a');     await sleep(DELAY_MS); // space context: embed in letters to avoid whitespace-only rejection
  const anchorWordBase   = await api('wordword'); await sleep(DELAY_MS);

  console.log(`anchors: "aa"=${anchorLetterBase} "11"=${anchorDigitBase} "a  a"=${anchorSpaceBase} "wordword"=${anchorWordBase}`);
  console.log('');

  const header = pad('symbol', 18) + padL('letter', 8) + padL('digit', 7) + padL('space', 7) + padL('word', 7) + padL('bpe', 6) + '  verdict';
  console.log(header);
  console.log('-'.repeat(header.length));

  const diffResults = [];

  for (const [sym, name] of SYMBOLS) {
    const letterCost = (await api('a' + sym + 'a')) - anchorLetterBase; await sleep(DELAY_MS);
    const digitCost  = (await api('1' + sym + '1')) - anchorDigitBase;  await sleep(DELAY_MS);
    const spaceCost  = (await api('a ' + sym + ' a')) - anchorSpaceBase;  await sleep(DELAY_MS);
    const wordCost   = (await api('word' + sym + 'word')) - anchorWordBase; await sleep(DELAY_MS);
    const bpe        = our(sym);

    const costs = [letterCost, digitCost, spaceCost, wordCost];
    const minCost = Math.min(...costs);
    const maxCost = Math.max(...costs);
    const consistent = minCost === maxCost;

    let verdict = '';
    if (consistent) {
      const delta = bpe - letterCost;
      if (delta === 0) verdict = '✓ match';
      else if (delta > 0) verdict = `OVER+${delta} (all ctx=${letterCost})`;
      else verdict = `UNDER${delta} (all ctx=${letterCost})`;
    } else {
      const avgCost = (costs.reduce((a, b) => a + b, 0) / costs.length).toFixed(1);
      verdict = `varies [${minCost}-${maxCost}] avg=${avgCost}, bpe=${bpe}`;
    }

    const label = `${sym} (${name})`;
    console.log(
      pad(label, 18) +
      padL(letterCost, 8) +
      padL(digitCost, 7) +
      padL(spaceCost, 7) +
      padL(wordCost, 7) +
      padL(bpe, 6) +
      '  ' + verdict,
    );

    diffResults.push({ sym, name, letterCost, digitCost, spaceCost, wordCost, bpe, consistent, minCost, maxCost });
  }

  // ─── Approach 2: Pre-tokenization boundary test ──────────────────────────────
  section('Approach 2: Pre-tokenization boundary test');
  console.log('Does Claude split at symbol/letter boundaries (like OpenAI regex) or do pure byte BPE?');
  console.log('If sum == full → boundary respected. If sum > full → cross-boundary byte merges exist.');
  console.log('');

  const bHeader = pad('test', 22) + padL('full', 7) + padL('sum', 7) + padL('delta', 8) + '  interpretation';
  console.log(bHeader);
  console.log('-'.repeat(bHeader.length));

  for (const { label, full, parts } of BOUNDARY_TESTS) {
    const fullCount = await api(full); await sleep(DELAY_MS);
    const partCounts = [];
    for (const p of parts) {
      partCounts.push(await api(p)); await sleep(DELAY_MS);
    }
    const sumCount = partCounts.reduce((a, b) => a + b, 0);
    const delta = fullCount - sumCount;
    let interp = '';
    if (delta === 0) interp = 'boundary respected (or no merges possible)';
    else if (delta < 0) interp = `cross-boundary merges (saves ${-delta} tokens)`;
    else interp = `UNEXPECTED: full costs more than parts (+${delta})`;

    console.log(pad(label, 22) + padL(fullCount, 7) + padL(sumCount, 7) + padL(delta > 0 ? '+' + delta : delta, 8) + '  ' + interp);
  }

  // ─── Approach 3: Concatenated symbol probes ──────────────────────────────────
  section('Approach 3: Concatenated symbol probes');
  console.log('Multi-symbol strings vs sum of individual marginal costs (from approach 1).');
  console.log('Negative delta = cross-symbol byte merges exist → byte-level BPE without boundary splitting.');
  console.log('');

  // Build a lookup: sym → letter marginal cost from approach 1
  const costByChar = new Map(diffResults.map(r => [r.sym, r.letterCost]));

  const cHeader = pad('group', 24) + padL('full', 7) + padL('sum', 7) + padL('delta', 8) + '  note';
  console.log(cHeader);
  console.log('-'.repeat(cHeader.length));

  for (const { label, chars } of CONCAT_GROUPS) {
    const fullCount = await api(chars); await sleep(DELAY_MS);
    const sumCost = [...chars].reduce((acc, ch) => {
      const c = costByChar.get(ch);
      return acc + (c !== undefined ? c : our(ch));
    }, 0);
    const delta = fullCount - sumCost;
    let note = '';
    if (delta === 0) note = 'no cross-symbol merges';
    else if (delta < 0) note = `${-delta} fewer tokens than expected — byte-level merges span symbols`;
    else note = `+${delta} MORE tokens than expected (unusual)`;

    console.log(pad(label, 24) + padL(fullCount, 7) + padL(sumCost, 7) + padL(delta > 0 ? '+' + delta : delta, 8) + '  ' + note);
  }

  // ─── Summary: actionable fixes ───────────────────────────────────────────────
  section('Summary: actionable vocab fixes');
  console.log('Symbols with consistent marginal cost across all 4 contexts can be reliably fixed:');
  console.log('');

  const fixes = diffResults.filter(r => r.consistent && r.letterCost !== r.bpe);
  if (fixes.length === 0) {
    console.log('No consistent mismatches found.');
  } else {
    for (const r of fixes) {
      const delta = r.bpe - r.letterCost;
      const action = delta > 0
        ? `bpe-lite overcounts by ${delta} → need to reduce tokens for "${r.sym}"`
        : `bpe-lite undercounts by ${-delta} → need to increase tokens for "${r.sym}"`;
      console.log(`  ${r.sym} (${r.name}): api=${r.letterCost} bpe=${r.bpe} → ${action}`);
    }
  }

  const varies = diffResults.filter(r => !r.consistent);
  if (varies.length > 0) {
    console.log('\nSymbols with context-dependent tokenization (harder to fix — byte-level context matters):');
    for (const r of varies) {
      console.log(`  ${r.sym} (${r.name}): letter=${r.letterCost} digit=${r.digitCost} space=${r.spaceCost} word=${r.wordCost} bpe=${r.bpe}`);
    }
  }

  console.log('\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
