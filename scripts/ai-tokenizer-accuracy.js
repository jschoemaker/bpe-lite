'use strict';
/**
 * Compare ai-tokenizer (claude encoding) vs bpe-lite vs Anthropic API.
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/ai-tokenizer-accuracy.js
 */

const { countTokens } = require('../src/index');
const { Tokenizer } = require('../node_modules/ai-tokenizer/dist/index.cjs');
const claudeEnc = require('../node_modules/ai-tokenizer/dist/encoding/claude.cjs');

const aiTok = new Tokenizer(claudeEnc);

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 120;

const SAMPLES = [
  { name: 'short prose (en)', text: 'The quick brown fox jumps over the lazy dog.' },
  { name: 'greeting', text: 'Hello, world!' },
  { name: '2-sentence prose', text: 'Machine learning models process text as tokens. Token boundaries often differ from word boundaries.' },
  { name: 'long prose (~200 tokens)', text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.' },
  { name: 'js function', text: 'function fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n - 1) + fibonacci(n - 2);\n}\nconsole.log(fibonacci(10)); // 55' },
  { name: 'python snippet', text: 'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)' },
  { name: 'json object', text: '{"name":"bpe-lite","version":"0.4.3","description":"Offline BPE tokenizer","keywords":["tokenizer","bpe","openai","anthropic","gemini"],"license":"MIT"}' },
  { name: 'html snippet', text: '<div class="container">\n  <h1>Hello, <span class="highlight">World</span>!</h1>\n  <p>This is a <a href="https://example.com">link</a>.</p>\n</div>' },
  { name: 'integers', text: '0 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765' },
  { name: 'floats & math', text: '3.14159265358979 2.71828182845905 1.41421356237 0.57721566490 1.61803398874' },
  { name: 'hex & binary', text: '0xFF 0x1A2B3C4D 0b10101010 0o777 255 26 170 511' },
  { name: 'arithmetic', text: '(3 + 4) * 2 - 1 / 0.5 = 13. sqrt(144) = 12. 2^10 = 1024. log2(256) = 8.' },
  { name: 'emoji', text: '😀 😂 🤔 🎉 🚀 🌍 🔥 💡 ❤️ 🐍 🦊 🌈 ⭐ 🎵 🏆' },
  { name: 'mixed emoji prose', text: "I love coding 💻 and coffee ☕. Let's build something amazing 🚀 together! 🎉" },
  { name: 'symbols', text: '© ® ™ § ¶ † ‡ • … ← → ↑ ↓ ↔ ≠ ≤ ≥ ± × ÷ √ ∞ ∑ ∏ ∫' },
  { name: 'currency', text: '$100.00 €85.50 £72.30 ¥12,000 ₿0.00234 ₹8,250 CHF 95.00 CAD 135.00' },
  { name: 'french', text: 'Le renard brun rapide saute par-dessus le chien paresseux. La vie est belle et le monde est grand.' },
  { name: 'spanish', text: 'El rápido zorro marrón salta sobre el perro perezoso. La inteligencia artificial cambia el mundo.' },
  { name: 'chinese (mandarin)', text: '人工智能正在改变世界。机器学习模型可以理解和生成自然语言。深度学习是人工智能的重要分支。' },
  { name: 'japanese', text: '人工知能は世界を変えています。機械学習モデルは自然言語を理解し生成することができます。' },
  { name: 'arabic', text: 'يغير الذكاء الاصطناعي العالم. يمكن لنماذج التعلم الآلي فهم اللغة الطبيعية وتوليدها.' },
  { name: 'single char', text: 'A' },
  { name: 'repeated chars', text: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' },
  { name: 'whitespace-heavy', text: 'word1     word2\t\tword3\n\n\nword4   \t  word5' },
  { name: 'url', text: 'https://api.anthropic.com/v1/messages/count_tokens?model=claude-3-5-sonnet-20241022&version=2023-06-01' },
];

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function pr(s, w) { s = String(s); return s.length >= w ? s : s + ' '.repeat(w - s.length); }
function pl(s, w) { s = String(s); return s.length >= w ? s : ' '.repeat(w - s.length) + s; }

async function countApi(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: { 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: text }] }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return (await res.json()).input_tokens;
}

function summarize(label, results, key) {
  const pcts = results.map(r => r[key]).sort((a, b) => a - b);
  const mean = pcts.reduce((s, v) => s + v, 0) / pcts.length;
  const median = pcts[Math.floor(pcts.length / 2)];
  const p95 = pcts[Math.floor(pcts.length * 0.95)];
  const max = pcts[pcts.length - 1];
  const maxSample = results.find(r => r[key] === max);
  const within = t => results.filter(r => r[key] <= t).length;
  const pct = n => `${((n / results.length) * 100).toFixed(1)}%`;
  console.log(`\n${label}`);
  console.log('─'.repeat(44));
  console.log(`exact:           ${within(0)}  (${pct(within(0))})`);
  console.log(`within 5%:       ${within(5)}  (${pct(within(5))})`);
  console.log(`within 10%:      ${within(10)}  (${pct(within(10))})`);
  console.log(`mean abs err:    ${mean.toFixed(2)}%`);
  console.log(`median abs err:  ${median.toFixed(2)}%`);
  console.log(`p95 abs err:     ${p95.toFixed(2)}%`);
  console.log(`max abs err:     ${max.toFixed(2)}%  (${maxSample.name})`);
}

async function main() {
  console.log(`ai-tokenizer claude encoding vs bpe-lite vs Anthropic API`);
  console.log(`model: ${MODEL} | samples: ${SAMPLES.length}`);
  console.log(`ai-tokenizer claude: ${claudeEnc.name}, pattern: ${claudeEnc.pat_str}`);
  console.log('Calibrating overhead...');
  const apiRaw = await countApi('Hi');
  const overhead = apiRaw - countTokens('Hi', 'anthropic');
  console.log(`Overhead: ${overhead} token(s)\n`);
  await sleep(DELAY_MS);

  const W = 28;
  const hdr = [pr('sample', W), pl('api', 6), pl('ai-tok', 8), pl('ai%', 7), pl('bpe-lite', 9), pl('bpe%', 7)].join('  ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  const results = [];
  for (const s of SAMPLES) {
    const api = (await countApi(s.text)) - overhead;
    const ai = aiTok.count(s.text);
    const bpe = countTokens(s.text, 'anthropic');
    const aiPct = api > 0 ? (Math.abs(ai - api) / api) * 100 : 0;
    const bpePct = api > 0 ? (Math.abs(bpe - api) / api) * 100 : 0;
    results.push({ name: s.name, api, ai, aiPct, bpe, bpePct });
    const aiDir = ai > api ? `+${ai-api}` : ai < api ? `${ai-api}` : '=';
    const bpeDir = bpe > api ? `+${bpe-api}` : bpe < api ? `${bpe-api}` : '=';
    console.log([
      pr(s.name, W),
      pl(api, 6),
      pl(`${ai}(${aiDir})`, 8),
      pl(`${aiPct.toFixed(1)}%`, 7),
      pl(`${bpe}(${bpeDir})`, 9),
      pl(`${bpePct.toFixed(1)}%`, 7),
    ].join('  '));
    await sleep(DELAY_MS);
  }

  summarize('ai-tokenizer (claude encoding)', results, 'aiPct');
  summarize('bpe-lite (modified Xenova)', results, 'bpePct');
}

main().catch(e => { console.error(e.message); process.exit(1); });
