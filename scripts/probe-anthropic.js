#!/usr/bin/env node
'use strict';

/**
 * Targeted probe script: reverse-engineers Claude's tokenizer.
 * Run:  ANTHROPIC_API_KEY=sk-... node scripts/probe-anthropic.js
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 110;

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
function section(title) { console.log('\n' + '─'.repeat(55)); console.log(title); console.log('─'.repeat(55)); }

async function probe(label, text) {
  // Skip whitespace-only (API rejects it)
  if (/^\s+$/.test(text)) { console.log(label.padEnd(32) + '  (skipped — whitespace only)'); return null; }
  const o = our(text);
  const a = await apiCount(text);
  const d = o - a;
  const flag = d !== 0 ? (d > 0 ? ' ← OVER +' + d : ' ← UNDER ' + d) : ' ✓';
  console.log(label.padEnd(32) + 'our=' + String(o).padStart(3) + '  api=' + String(a).padStart(3) + flag);
  await sleep(DELAY_MS);
  return { label, text, our: o, api: a, delta: d };
}

async function main() {
  console.log('probe-anthropic — model:', MODEL);

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

  // ─── Repeated chars ───────────────────────────────────────────────────────
  section('Repeated chars — a×N');
  for (const n of [1,2,3,4,5,6,7,8,9,10,12,16,24,32,48,64]) {
    await probe('a×' + n, 'a'.repeat(n));
  }

  section('Repeated chars — other letters');
  for (const ch of ['b','x','z','A','B']) {
    for (const n of [2,3,4,8]) {
      await probe(ch + '×' + n, ch.repeat(n));
    }
  }

  // ─── Symbols — individually ───────────────────────────────────────────────
  section('Symbols (bare)');
  const symbols = [
    ['©','copyright'], ['®','registered'], ['™','trademark'],
    ['§','section'], ['¶','pilcrow'], ['†','dagger'], ['‡','dagger2'],
    ['•','bullet'], ['…','ellipsis'], ['←','left-arrow'], ['→','right-arrow'],
    ['↑','up-arrow'], ['↓','down-arrow'], ['↔','lr-arrow'],
    ['≠','neq'], ['≤','leq'], ['≥','geq'], ['≈','approx'],
    ['±','plus-minus'], ['×','times'], ['÷','divide'],
    ['√','sqrt'], ['∞','infinity'], ['∑','sum'], ['∏','product'], ['∫','integral'],
    ['−','minus-sign'],
  ];
  for (const [sym, name] of symbols) {
    await probe(name + ' (' + sym + ')', sym);
  }

  section('Symbols (space-prefixed)');
  for (const [sym, name] of symbols) {
    await probe(' ' + name, ' ' + sym);
  }

  // ─── Currency ─────────────────────────────────────────────────────────────
  section('Currency symbols');
  for (const [sym, name] of [['€','euro'],['£','pound'],['¥','yen'],['₿','bitcoin'],['₹','rupee'],['¢','cent'],['₩','won'],['₽','ruble']]) {
    await probe(name, sym);
    await probe(' ' + name, ' ' + sym);
  }

  // ─── Emoji ────────────────────────────────────────────────────────────────
  section('Emoji (bare)');
  const emojis = [
    ['\u{1F600}','grin'],  ['\u{1F602}','joy'],   ['\u{1F914}','think'],
    ['\u{1F389}','party'], ['\u{1F680}','rocket'], ['\u{1F30D}','earth'],
    ['\u{1F525}','fire'],  ['\u{1F4A1}','bulb'],   ['\u{2764}\uFE0F','heart-emoji'],
    ['\u{1F40D}','snake'], ['\u{1F98A}','fox'],    ['\u{1F308}','rainbow'],
    ['\u{2B50}','star'],   ['\u{1F3B5}','music'],  ['\u{1F3C6}','trophy'],
    ['\u{2615}','coffee'], ['\u{1F4BB}','laptop'],
    ['\u{1F44D}','thumbsup'], ['\u{1F60D}','heart-eyes'],
  ];
  for (const [emoji, name] of emojis) {
    await probe(name, emoji);
  }

  section('Emoji (space-prefixed)');
  for (const [emoji, name] of emojis) {
    await probe(' ' + name, ' ' + emoji);
  }

  // ─── Arabic ───────────────────────────────────────────────────────────────
  section('Arabic individual chars');
  const arabicSample = 'يغيرالذكاء';
  const seenA = new Set();
  for (const ch of arabicSample) {
    if (seenA.has(ch)) continue; seenA.add(ch);
    await probe('U+' + ch.codePointAt(0).toString(16) + ' (' + ch + ')', ch);
  }

  section('Arabic bigrams');
  for (let i = 0; i < arabicSample.length - 1; i++) {
    await probe(arabicSample[i] + arabicSample[i+1], arabicSample[i] + arabicSample[i+1]);
  }

  // ─── CJK ──────────────────────────────────────────────────────────────────
  section('CJK individual chars');
  const cjkSample = '人工知能世界機械';
  const seenC = new Set();
  for (const ch of cjkSample) {
    if (seenC.has(ch)) continue; seenC.add(ch);
    await probe('U+' + ch.codePointAt(0).toString(16) + ' (' + ch + ')', ch);
  }

  section('CJK bigrams');
  for (let i = 0; i < cjkSample.length - 1; i++) {
    await probe(cjkSample[i] + cjkSample[i+1], cjkSample[i] + cjkSample[i+1]);
  }

  console.log('\nDone.');
}

main().catch(e => { console.error(e.message); process.exit(1); });
