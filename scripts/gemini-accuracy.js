#!/usr/bin/env node
'use strict';

/**
 * Accuracy test: bpe-lite "gemini" provider vs Google Generative AI countTokens API.
 *
 * Run:
 *   GEMINI_API_KEY=... node scripts/gemini-accuracy.js
 *
 * Notes:
 * - Requires a Google AI Studio API key (free at aistudio.google.com).
 * - bpe-lite uses the Gemma 3 vocabulary (262,144 tokens). This script measures
 *   how well that matches the tokenization used by the live Gemini API.
 */

const { countTokens } = require('../src/index');

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('Error: GEMINI_API_KEY environment variable is not set.');
  process.exit(1);
}

const MODEL = process.env.GEMINI_MODEL || 'gemini-2.0-flash';
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
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:countTokens?key=${API_KEY}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts: [{ text }] }] }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  const data = await res.json();
  return data.totalTokens;
}

async function main() {
  console.log(`bpe-lite gemini accuracy`);
  console.log(`model: ${MODEL}`);
  console.log(`samples: ${SAMPLES.length}`);
  console.log('');

  // Verify API is reachable and check for any overhead
  process.stdout.write('Checking API with known text... ');
  const apiCheck = await countTokensApi('Hello');
  const bpeCheck = countTokens('Hello', 'gemini');
  console.log(`API: ${apiCheck}, bpe-lite: ${bpeCheck}`);
  await sleep(DELAY_MS);

  const results = [];
  const colName = 28;

  const header = [
    padRight('sample', colName),
    padLeft('api', 6),
    padLeft('bpe', 6),
    padLeft('delta', 7),
    padLeft('err%', 7),
  ].join('  ');
  console.log('');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const sample of SAMPLES) {
    const api = await countTokensApi(sample.text);
    const bpe = countTokens(sample.text, 'gemini');
    const delta = bpe - api;
    const pct = api > 0 ? (Math.abs(delta) / api) * 100 : 0;

    results.push({ name: sample.name, api, bpe, delta, pct });

    const deltaStr = delta === 0 ? '0' : delta > 0 ? `+${delta}` : String(delta);
    console.log(
      [
        padRight(sample.name, colName),
        padLeft(api, 6),
        padLeft(bpe, 6),
        padLeft(deltaStr, 7),
        padLeft(`${pct.toFixed(2)}%`, 7),
      ].join('  '),
    );

    await sleep(DELAY_MS);
  }

  const pcts = results.map((r) => r.pct).sort((a, b) => a - b);
  const mean = pcts.reduce((s, v) => s + v, 0) / pcts.length;
  const median = pcts[Math.floor(pcts.length / 2)];
  const p95 = pcts[Math.floor(pcts.length * 0.95)];
  const max = pcts[pcts.length - 1];
  const maxSample = results.find((r) => r.pct === max);
  const within = (thr) => results.filter((r) => r.pct <= thr).length;
  const pct = (n) => `${((n / results.length) * 100).toFixed(1)}%`;

  console.log('');
  console.log('Summary');
  console.log('─'.repeat(40));
  console.log(`samples:         ${results.length}`);
  console.log(`exact:           ${within(0)}  (${pct(within(0))})`);
  console.log(`within 1%:       ${within(1)}  (${pct(within(1))})`);
  console.log(`within 2%:       ${within(2)}  (${pct(within(2))})`);
  console.log(`within 5%:       ${within(5)}  (${pct(within(5))})`);
  console.log(`within 10%:      ${within(10)}  (${pct(within(10))})`);
  console.log('');
  console.log(`mean abs err:    ${mean.toFixed(2)}%`);
  console.log(`median abs err:  ${median.toFixed(2)}%`);
  console.log(`p95 abs err:     ${p95.toFixed(2)}%`);
  console.log(`max abs err:     ${max.toFixed(2)}%  (${maxSample.name})`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
