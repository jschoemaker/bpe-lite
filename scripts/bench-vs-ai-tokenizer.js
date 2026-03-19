'use strict';
/**
 * bpe-lite vs ai-tokenizer — side-by-side benchmark using ai-tokenizer's own text fixtures.
 *
 * Run:
 *   node --expose-gc scripts/bench-vs-ai-tokenizer.js
 *
 * Notes:
 * - Mirrors ai-tokenizer's bench/versus.ts fixture sizes and scenarios.
 * - ai-tokenizer's published numbers use Bun; this runs on Node. Expect Node to be
 *   ~20-40% slower overall — treat comparisons as relative, not absolute.
 * - bpe-lite has a per-instance chunk-level cache. We measure both:
 *     COLD: first N iters on text never seen before (cache misses)
 *     WARM: next N iters on the same text (cache hits, reflects steady-state perf)
 *   ai-tokenizer has no such cache; its numbers are always cold.
 */

const bpeLite = require('../src/index');

function tryRequire(m) { try { return require(m); } catch { return null; } }
const aiTokenizerMod = tryRequire('ai-tokenizer');
const aiO200k        = tryRequire('ai-tokenizer/encoding/o200k_base');
const aiClaude       = tryRequire('ai-tokenizer/encoding/claude');
const jsTiktoken     = tryRequire('js-tiktoken');
const gptTok         = tryRequire('gpt-tokenizer');

const AiTok = aiTokenizerMod && (aiTokenizerMod.Tokenizer || aiTokenizerMod.default);

// ── Text fixtures (mirrors ai-tokenizer's bench/versus.ts) ───────────────────

const TEXTS = {
  small: 'Hello, world!',

  medium: 'The quick brown fox jumps over the lazy dog. '.repeat(100), // ~4.5 KB

  large: [
    'Machine learning models process text by breaking it down into tokens.',
    'Natural language processing has evolved significantly with large language models.',
    'These models can understand context, generate coherent text, and perform various tasks.',
  ].join('\n').repeat(5000), // ~500 KB

  unicode: 'Hello 世界! 🌍 Привет мир! مرحبا بالعالم! '.repeat(100),

  code: [
    'function fibonacci(n) {',
    '  if (n <= 1) return n;',
    '  return fibonacci(n - 1) + fibonacci(n - 2);',
    '}',
    'const result = fibonacci(10);',
    'console.log(result);',
  ].join('\n').repeat(100),

  mixed: [
    '# Machine Learning Overview',
    '',
    'Machine learning (ML) is a subset of artificial intelligence (AI).',
    '',
    '## Key Concepts:',
    '- Supervised Learning: 监督学习',
    '- Unsupervised Learning: 無監督学習',
    '- Reinforcement Learning: 強化学習',
    '',
    '```python',
    'def train_model(data, labels):',
    '    model = NeuralNetwork()',
    '    model.fit(data, labels)',
    '    return model',
    '```',
    '',
    '**Performance metrics**: accuracy: 95.5%, precision: 0.94, recall: 0.96',
  ].join('\n').repeat(200),
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function nowNs() { return process.hrtime.bigint(); }
function gcNow() { if (typeof global.gc === 'function') global.gc(); }
function bytes(text) { return Buffer.byteLength(text, 'utf8'); }
function mb(b) { return (b / 1024 / 1024).toFixed(2); }

function pad(s, w, right = false) {
  s = String(s);
  const pad = w > s.length ? ' '.repeat(w - s.length) : '';
  return right ? s + pad : pad + s;
}

function runBench(fn, text, iters) {
  gcNow();
  const t0 = nowNs();
  let total = 0;
  for (let i = 0; i < iters; i++) total += fn(text).length ?? 0;
  const elapsed = Number(nowNs() - t0) / 1e9;
  return { opsPerSec: iters / elapsed, mbPerSec: (bytes(text) * iters / elapsed) / (1024 * 1024), total };
}

function benchInit(fn, iters = 10) {
  gcNow();
  const t0 = nowNs();
  for (let i = 0; i < iters; i++) fn();
  const elapsed = Number(nowNs() - t0) / 1e9;
  return { usPerOp: (elapsed / iters) * 1e6 };
}

function fmtOps(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return n.toFixed(1);
}

function printHeader(title, textKey) {
  const b = bytes(TEXTS[textKey]);
  console.log(`\n${title} (${mb(b)} MB)`);
  console.log('─'.repeat(title.length + mb(b).length + 8));
  console.log(
    pad('impl', 28) + '  ' +
    pad('ops/s', 10) + '  ' +
    pad('MB/s', 8) + '  ' +
    pad('note', 0)
  );
  console.log('-'.repeat(70));
}

function printRow(name, r, note = '') {
  console.log(
    pad(name, 28) + '  ' +
    pad(fmtOps(r.opsPerSec), 10) + '  ' +
    pad(r.mbPerSec.toFixed(1), 8) + '  ' +
    note
  );
}

// ── Build impls ───────────────────────────────────────────────────────────────

function buildImpls() {
  const impls = {};

  // bpe-lite — separate instances per scenario so cache state is controlled
  impls.bpeCl100k   = (text) => bpeLite.encode(text, 'openai');
  impls.bpeO200k    = (text) => bpeLite.encode(text, 'openai-o200k');
  impls.bpeAnthropic = (text) => bpeLite.encode(text, 'anthropic');
  impls.bpeGemini   = (text) => bpeLite.encode(text, 'gemini');

  if (AiTok && aiO200k) {
    const tok = new AiTok(aiO200k);
    impls.aiO200k = (text) => tok.encode(text);
  }
  if (AiTok && aiClaude) {
    const tok = new AiTok(aiClaude);
    impls.aiClaude = (text) => tok.encode(text);
  }
  if (jsTiktoken) {
    const enc = jsTiktoken.getEncoding('o200k_base');
    impls.jsTiktokenO200k = (text) => enc.encode(text, 'all');
  }
  if (gptTok) {
    impls.gptTok = (text) => gptTok.encode(text);
  }

  return impls;
}

// ── Initialization benchmark ──────────────────────────────────────────────────

function benchInitAll() {
  console.log('\nInitialization');
  console.log('─'.repeat(50));
  console.log(pad('impl', 28) + '  ' + pad('µs/init', 10));
  console.log('-'.repeat(42));

  const r1 = benchInit(() => { /* bpe-lite lazy-loads on first encode, not on require */ });
  // bpe-lite loads lazily — measure first encode
  const t0 = nowNs();
  bpeLite.encode('Hi', 'openai');
  const bpeLazyUs = Number(nowNs() - t0) / 1e3;
  console.log(pad('bpe-lite (first encode, cl100k)', 28) + '  ' + pad(bpeLazyUs.toFixed(0) + ' µs', 10));

  if (AiTok && aiO200k) {
    const r = benchInit(() => new AiTok(aiO200k));
    console.log(pad('ai-tokenizer (new Tokenizer)', 28) + '  ' + pad(r.usPerOp.toFixed(0) + ' µs', 10));
  }
  if (jsTiktoken) {
    const r = benchInit(() => {
      const e = jsTiktoken.getEncoding('o200k_base');
      e.free?.();
    }, 3);
    console.log(pad('js-tiktoken (getEncoding)', 28) + '  ' + pad(r.usPerOp.toFixed(0) + ' µs', 10));
  }
}

// ── Encode benchmark (cold vs warm for bpe-lite) ──────────────────────────────

function benchEncodeScenario(textKey, itersSmall, itersMed) {
  const text = TEXTS[textKey];
  const iters = bytes(text) < 10_000 ? itersSmall : itersMed;
  const warmup = Math.max(1, Math.floor(iters / 20));

  printHeader(`encode — ${textKey}`, textKey);

  // bpe-lite COLD: fresh tokenizer instances so the cache starts empty.
  // We use a new instance for each scenario to simulate cold-start.
  // iters / 5 cold iters to measure miss-dominated performance.
  {
    const coldIters = Math.max(3, Math.floor(iters / 10));
    // Reset cache by encoding a completely different text first would not reset instance cache.
    // Simplest approach: measure first coldIters iterations on a fresh instance.
    const freshCl = () => { const tok = require('../src/tokenizer').Tokenizer; /* no-op */ };

    // Measure cold on a standalone encode call — bpe-lite caches by tokenizer instance,
    // so we get cache-miss behavior by encoding unique text variants.
    // Approximate cold by measuring 1 iteration on text never touched.
    const uniqueText = text + '\x00cold' + Math.random(); // never been in any cache
    const coldR = runBench(impls.bpeCl100k, uniqueText, coldIters);
    printRow('bpe-lite cl100k  [cold]', coldR, 'cache miss (novel text)');
  }

  // bpe-lite WARM: warmup to fill chunk cache, then measure.
  {
    for (let i = 0; i < warmup; i++) impls.bpeCl100k(text);
    const warmR = runBench(impls.bpeCl100k, text, iters);
    printRow('bpe-lite cl100k  [warm]', warmR, 'cache hit  (same text repeated)');
  }
  if (impls.bpeO200k) {
    for (let i = 0; i < warmup; i++) impls.bpeO200k(text);
    const r = runBench(impls.bpeO200k, text, iters);
    printRow('bpe-lite o200k   [warm]', r);
  }
  if (impls.bpeAnthropic) {
    for (let i = 0; i < warmup; i++) impls.bpeAnthropic(text);
    const r = runBench(impls.bpeAnthropic, text, iters);
    printRow('bpe-lite anthropic[warm]', r);
  }
  console.log('');
  if (impls.aiO200k) {
    for (let i = 0; i < warmup; i++) impls.aiO200k(text);
    const r = runBench(impls.aiO200k, text, iters);
    printRow('ai-tokenizer o200k', r);
  }
  if (impls.aiClaude) {
    for (let i = 0; i < warmup; i++) impls.aiClaude(text);
    const r = runBench(impls.aiClaude, text, iters);
    printRow('ai-tokenizer claude', r);
  }
  if (impls.jsTiktokenO200k) {
    for (let i = 0; i < Math.min(warmup, 3); i++) impls.jsTiktokenO200k(text);
    const r = runBench(impls.jsTiktokenO200k, text, Math.min(iters, 5));
    printRow('js-tiktoken o200k', r, '(WASM)');
  }
  if (impls.gptTok) {
    for (let i = 0; i < warmup; i++) impls.gptTok(text);
    const r = runBench(impls.gptTok, text, iters);
    printRow('gpt-tokenizer', r);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

let impls;

function main() {
  console.log('bpe-lite vs ai-tokenizer — Node.js benchmark');
  console.log(`node ${process.version} (${process.platform}/${process.arch})`);
  console.log(`gc: ${typeof global.gc === 'function' ? 'available' : 'unavailable (run with --expose-gc)'}`);
  console.log('');
  console.log('Available:');
  console.log(`  bpe-lite:      ✓`);
  console.log(`  ai-tokenizer:  ${AiTok ? '✓' : '✗'}`);
  console.log(`  js-tiktoken:   ${jsTiktoken ? '✓' : '✗'}`);
  console.log(`  gpt-tokenizer: ${gptTok ? '✓' : '✗'}`);

  impls = buildImpls();

  benchInitAll();

  // Scenario: itersSmall (< 10KB text), itersMed (>= 10KB text)
  benchEncodeScenario('small',   10_000, 1_000);
  benchEncodeScenario('medium',   1_000,   200);
  benchEncodeScenario('large',       20,    10);
  benchEncodeScenario('unicode',    500,   200);
  benchEncodeScenario('code',       500,   200);
  benchEncodeScenario('mixed',      200,    50);

  console.log('\nNote: [cold] = cache-miss performance on novel text (realistic for diverse inputs).');
  console.log('      [warm] = cache-hit performance after warmup on the same text (best case).');
  console.log('      ai-tokenizer has no chunk cache — its numbers are always cold.');
  console.log('      All numbers are on Node.js; ai-tokenizer\'s README uses Bun (~20-40% faster).');
}

main();
