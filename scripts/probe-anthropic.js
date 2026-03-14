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

  // ─── Arabic — all common letters individually ─────────────────────────────
  section('Arabic letters — D8 block (U+0620–U+063F)');
  const arabicLettersD8 = [
    '\u0621', // ء  hamza      D8 A1
    '\u0622', // آ  alef madda D8 A2
    '\u0623', // أ  alef hmza  D8 A3
    '\u0624', // ؤ  waw hamza  D8 A4
    '\u0625', // إ  alef below D8 A5
    '\u0626', // ئ  ya hamza   D8 A6
    '\u0627', // ا  alef       D8 A7
    '\u0628', // ب  ba         D8 A8
    '\u0629', // ة  ta marbuta D8 A9
    '\u062A', // ت  ta         D8 AA
    '\u062B', // ث  tha        D8 AB
    '\u062C', // ج  jeem       D8 AC
    '\u062D', // ح  hha        D8 AD
    '\u062E', // خ  kha        D8 AE
    '\u062F', // د  dal        D8 AF
    '\u0630', // ذ  dhal       D8 B0
    '\u0631', // ر  ra         D8 B1
    '\u0632', // ز  zain       D8 B2
    '\u0633', // س  seen       D8 B3
    '\u0634', // ش  sheen      D8 B4
    '\u0635', // ص  sad        D8 B5
    '\u0636', // ض  dad        D8 B6
    '\u0637', // ط  ta emph    D8 B7
    '\u0638', // ظ  dha        D8 B8
    '\u0639', // ع  ain        D8 B9
    '\u063A', // غ  ghain      D8 BA
  ];
  for (const ch of arabicLettersD8) {
    await probe('U+' + ch.codePointAt(0).toString(16) + ' (' + ch + ')', ch);
  }

  section('Arabic letters — D9 block (U+0641–U+064A)');
  const arabicLettersD9 = [
    '\u0641', // ف  fa         D9 81
    '\u0642', // ق  qaf        D9 82
    '\u0643', // ك  kaf        D9 83
    '\u0644', // ل  lam        D9 84
    '\u0645', // م  meem       D9 85
    '\u0646', // ن  noon       D9 86
    '\u0647', // ه  ha         D9 87
    '\u0648', // و  waw        D9 88
    '\u0649', // ى  alef maks  D9 89
    '\u064A', // ي  ya         D9 8A
    '\u064B', // ◌ً  fathatan   D9 8B
    '\u064C', // ◌ٌ  dammatan   D9 8C
    '\u064E', // ◌َ  fatha      D9 8E
    '\u064F', // ◌ُ  damma      D9 8F
    '\u0650', // ◌ِ  kasra      D9 90
    '\u0651', // ◌ّ  shadda     D9 91
  ];
  for (const ch of arabicLettersD9) {
    await probe('U+' + ch.codePointAt(0).toString(16) + ' (' + ch + ')', ch);
  }

  section('Arabic common bigrams');
  const arabicBigrams = [
    'ال', 'لا', 'في', 'من', 'عل', 'ها', 'ية', 'وا',
    'ان', 'ما', 'نا', 'ير', 'كا', 'ار', 'اء', 'لل',
    'ذي', 'غي', 'كذ', 'اك', 'لذ', 'را', 'ذا', 'كاء',
  ];
  for (const pair of arabicBigrams) {
    await probe(JSON.stringify(pair), pair);
  }

  // ─── Whitespace ───────────────────────────────────────────────────────────
  section('Repeated spaces');
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]) {
    await probe('space×' + n, ' '.repeat(n));
  }

  section('Repeated tabs');
  for (const n of [1, 2, 3, 4, 6, 8]) {
    await probe('tab×' + n, '\t'.repeat(n));
  }

  section('Mixed whitespace');
  for (const [label, text] of [
    ['2sp+tab', '  \t'],
    ['tab+2sp', '\t  '],
    ['4sp+nl', '    \n'],
    ['sp+tab+sp', ' \t '],
    ['2tab', '\t\t'],
    ['nl+4sp', '\n    '],
  ]) {
    await probe(label, text);
  }

  // ─── Repeated punctuation ─────────────────────────────────────────────────
  section('Repeated punctuation chars');
  for (const ch of ['-', '_', '.', '*', '=', '#', '~', '/']) {
    for (const n of [2, 3, 4, 8, 16]) {
      await probe(ch + '×' + n, ch.repeat(n));
    }
  }

  // ─── More emoji ───────────────────────────────────────────────────────────
  section('More emoji (bare)');
  const moreEmojis = [
    ['\u{1F923}', 'rofl'],   ['\u{1F970}', 'smiling-hearts'],
    ['\u{1F621}', 'angry'],  ['\u{1F622}', 'cry'],
    ['\u{1F631}', 'scream'], ['\u{1F643}', 'upside-down'],
    ['\u{1F4AF}', '100'],    ['\u{1F525}', 'fire2'],
    ['\u{2728}',  'sparkles'], ['\u{1F973}', 'party-face'],
    ['\u{1F91D}', 'handshake'], ['\u{1F4AA}', 'flexed'],
    ['\u{1F9E0}', 'brain'],  ['\u{1F499}', 'blue-heart'],
    ['\u{1F4B0}', 'money-bag'], ['\u{1F4C8}', 'chart-up'],
    ['\u{26A1}',  'lightning'], ['\u{1F48E}', 'gem'],
  ];
  for (const [emoji, name] of moreEmojis) {
    await probe(name, emoji);
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
