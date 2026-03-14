#!/usr/bin/env node
'use strict';

/**
 * Accuracy test: bpe-lite "anthropic" provider vs Anthropic count_tokens API.
 * Also compares o200k (via js-tiktoken) as an alternative base tokenizer.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... node scripts/anthropic-accuracy.js
 *
 * Notes:
 * - Requires an Anthropic API key (count_tokens is cheap — fractions of a cent).
 * - The API counts tokens for the entire message structure, not raw text.
 *   We calibrate the fixed overhead once using a known 1-token string, then
 *   subtract it from all subsequent API results.
 * - bpe-lite "anthropic" is a cl100k approximation; Anthropic has not released
 *   the Claude 4+ tokenizer. This script measures how close the approximation is.
 */

const { countTokens } = require('../src/index');

function tryRequire(name) {
  try { return require(name); } catch { return null; }
}
const jsTiktoken = tryRequire('js-tiktoken');
const encO200k = jsTiktoken ? jsTiktoken.getEncoding('o200k_base') : null;

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) {
  console.error('Error: ANTHROPIC_API_KEY environment variable is not set.');
  process.exit(1);
}

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 120;

const SAMPLES = [
  // Short prose
  { name: 'short prose (en)', text: 'The quick brown fox jumps over the lazy dog.' },
  { name: 'greeting', text: 'Hello, world!' },
  { name: '2-sentence prose', text: 'Machine learning models process text as tokens. Token boundaries often differ from word boundaries.' },

  // Long prose
  {
    name: 'long prose (~200 tokens)',
    text:
      'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
      'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ' +
      'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ' +
      'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ' +
      'Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, ' +
      'eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.',
  },

  // Code
  {
    name: 'js function',
    text:
      'function fibonacci(n) {\n' +
      '  if (n <= 1) return n;\n' +
      '  return fibonacci(n - 1) + fibonacci(n - 2);\n' +
      '}\n' +
      'console.log(fibonacci(10)); // 55',
  },
  {
    name: 'python snippet',
    text:
      'def quicksort(arr):\n' +
      '    if len(arr) <= 1:\n' +
      '        return arr\n' +
      '    pivot = arr[len(arr) // 2]\n' +
      '    left = [x for x in arr if x < pivot]\n' +
      '    middle = [x for x in arr if x == pivot]\n' +
      '    right = [x for x in arr if x > pivot]\n' +
      '    return quicksort(left) + middle + quicksort(right)',
  },
  {
    name: 'json object',
    text: '{"name":"bpe-lite","version":"0.4.3","description":"Offline BPE tokenizer","keywords":["tokenizer","bpe","openai","anthropic","gemini"],"license":"MIT"}',
  },
  {
    name: 'html snippet',
    text: '<div class="container">\n  <h1>Hello, <span class="highlight">World</span>!</h1>\n  <p>This is a <a href="https://example.com">link</a>.</p>\n</div>',
  },

  // Numbers
  { name: 'integers', text: '0 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765' },
  { name: 'floats & math', text: '3.14159265358979 2.71828182845905 1.41421356237 0.57721566490 1.61803398874' },
  { name: 'hex & binary', text: '0xFF 0x1A2B3C4D 0b10101010 0o777 255 26 170 511' },
  { name: 'arithmetic', text: '(3 + 4) * 2 - 1 / 0.5 = 13. sqrt(144) = 12. 2^10 = 1024. log2(256) = 8.' },

  // Emoji & symbols
  { name: 'emoji', text: '😀 😂 🤔 🎉 🚀 🌍 🔥 💡 ❤️ 🐍 🦊 🌈 ⭐ 🎵 🏆' },
  { name: 'mixed emoji prose', text: "I love coding 💻 and coffee ☕. Let's build something amazing 🚀 together! 🎉" },
  { name: 'symbols', text: '© ® ™ § ¶ † ‡ • … ← → ↑ ↓ ↔ ≠ ≤ ≥ ± × ÷ √ ∞ ∑ ∏ ∫' },
  { name: 'currency', text: '$100.00 €85.50 £72.30 ¥12,000 ₿0.00234 ₹8,250 CHF 95.00 CAD 135.00' },

  // Multilingual
  { name: 'french', text: 'Le renard brun rapide saute par-dessus le chien paresseux. La vie est belle et le monde est grand.' },
  { name: 'spanish', text: 'El rápido zorro marrón salta sobre el perro perezoso. La inteligencia artificial cambia el mundo.' },
  { name: 'chinese (mandarin)', text: '人工智能正在改变世界。机器学习模型可以理解和生成自然语言。深度学习是人工智能的重要分支。' },
  { name: 'japanese', text: '人工知能は世界を変えています。機械学習モデルは自然言語を理解し生成することができます。' },
  { name: 'arabic', text: 'يغير الذكاء الاصطناعي العالم. يمكن لنماذج التعلم الآلي فهم اللغة الطبيعية وتوليدها.' },

  // Edge cases
  { name: 'single char', text: 'A' },
  { name: 'repeated chars', text: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' },
  { name: 'whitespace-heavy', text: 'word1     word2\t\tword3\n\n\nword4   \t  word5' },
  { name: 'url', text: 'https://api.anthropic.com/v1/messages/count_tokens?model=claude-3-5-sonnet-20241022&version=2023-06-01' },
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function padRight(s, w) {
  s = String(s);
  return s.length >= w ? s : s + ' '.repeat(w - s.length);
}

function padLeft(s, w) {
  s = String(s);
  return s.length >= w ? s : ' '.repeat(w - s.length) + s;
}

async function countTokensApi(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: {
      'x-api-key': API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: text }],
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  const data = await res.json();
  return data.input_tokens;
}

async function measureOverhead() {
  // "Hi" tokenizes to 1 token in cl100k.
  // The API adds a fixed overhead for message structure; measure it here.
  const apiRaw = await countTokensApi('Hi');
  const bpe = countTokens('Hi', 'anthropic');
  return apiRaw - bpe;
}

function summarize(label, results, key) {
  const pcts = results.map((r) => r[key]).sort((a, b) => a - b);
  const mean = pcts.reduce((s, v) => s + v, 0) / pcts.length;
  const median = pcts[Math.floor(pcts.length / 2)];
  const p95 = pcts[Math.floor(pcts.length * 0.95)];
  const max = pcts[pcts.length - 1];
  const maxSample = results.find((r) => r[key] === max);
  const within = (thr) => results.filter((r) => r[key] <= thr).length;
  const pct = (n) => `${((n / results.length) * 100).toFixed(1)}%`;

  console.log(`\n${label}`);
  console.log('─'.repeat(40));
  console.log(`exact:           ${within(0)}  (${pct(within(0))})`);
  console.log(`within 1%:       ${within(1)}  (${pct(within(1))})`);
  console.log(`within 2%:       ${within(2)}  (${pct(within(2))})`);
  console.log(`within 5%:       ${within(5)}  (${pct(within(5))})`);
  console.log(`within 10%:      ${within(10)}  (${pct(within(10))})`);
  console.log(`mean abs err:    ${mean.toFixed(2)}%`);
  console.log(`median abs err:  ${median.toFixed(2)}%`);
  console.log(`p95 abs err:     ${p95.toFixed(2)}%`);
  console.log(`max abs err:     ${max.toFixed(2)}%  (${maxSample.name})`);
}

async function main() {
  console.log(`bpe-lite anthropic accuracy`);
  console.log(`model: ${MODEL}`);
  console.log(`samples: ${SAMPLES.length}`);
  if (!encO200k) console.log('Note: js-tiktoken not installed — o200k column will be skipped.');
  console.log('');

  console.log('Calibrating message overhead...');
  const overhead = await measureOverhead();
  console.log(`Message overhead: ${overhead} token(s) (subtracted from all API results)\n`);
  await sleep(DELAY_MS);

  const results = [];
  const colName = 28;

  const header = encO200k
    ? [padRight('sample', colName), padLeft('api', 6), padLeft('cl100k', 7), padLeft('cl%', 7), padLeft('o200k', 7), padLeft('o200k%', 8)].join('  ')
    : [padRight('sample', colName), padLeft('api', 6), padLeft('cl100k', 7), padLeft('cl%', 7)].join('  ');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const sample of SAMPLES) {
    const apiRaw = await countTokensApi(sample.text);
    const api = apiRaw - overhead;
    const cl = countTokens(sample.text, 'anthropic');
    const clDelta = cl - api;
    const clPct = api > 0 ? (Math.abs(clDelta) / api) * 100 : 0;

    const result = { name: sample.name, api, cl, clPct };

    let line = [
      padRight(sample.name, colName),
      padLeft(api, 6),
      padLeft(cl, 7),
      padLeft(`${clPct.toFixed(1)}%`, 7),
    ];

    if (encO200k) {
      const o2 = encO200k.encode(sample.text).length;
      const o2Delta = o2 - api;
      const o2Pct = api > 0 ? (Math.abs(o2Delta) / api) * 100 : 0;
      result.o2 = o2;
      result.o2Pct = o2Pct;
      line.push(padLeft(o2, 7), padLeft(`${o2Pct.toFixed(1)}%`, 8));
    }

    results.push(result);
    console.log(line.join('  '));
    await sleep(DELAY_MS);
  }

  console.log(`\nSummary (${SAMPLES.length} samples)`);
  summarize('cl100k (bpe-lite current)', results, 'clPct');
  if (encO200k) summarize('o200k (js-tiktoken)', results, 'o2Pct');
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
