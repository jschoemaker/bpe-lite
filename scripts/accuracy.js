'use strict';
/**
 * Accuracy benchmark — bpe-lite vs ai-tokenizer vs Anthropic API.
 * Reads pre-fetched expected counts from corpus-expected.json (run fetch-corpus.js first).
 *
 * Run:
 *   node scripts/accuracy.js
 */

const fs = require('fs');
const path = require('path');
const { countTokens } = require('../src/index');

function tryRequire(name) {
  try { return require(name); } catch { return null; }
}
const aiTokenizer = tryRequire('ai-tokenizer');
const claudeEnc = tryRequire('ai-tokenizer/encoding/claude');
const aiTok = aiTokenizer && claudeEnc ? new (aiTokenizer.Tokenizer || aiTokenizer.default)(claudeEnc) : null;

const CORPUS_PATH = path.join(__dirname, 'corpus-expected.json');
if (!fs.existsSync(CORPUS_PATH)) {
  console.error('corpus-expected.json not found. Run first:\n  ANTHROPIC_API_KEY=sk-... node scripts/fetch-corpus.js');
  process.exit(1);
}

const corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, 'utf8'));

// ── Wilson score CI for a proportion ─────────────────────────────────────────
function wilsonCI(k, n, z = 1.96) {
  if (n === 0) return [0, 1];
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const margin = (z / denom) * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function fmt(n, decimals = 1) { return n.toFixed(decimals); }
function pct(k, n) { return `${((k / n) * 100).toFixed(1)}%`; }
function ciStr(k, n) {
  const [lo, hi] = wilsonCI(k, n);
  return `±${(((hi - lo) / 2) * 100).toFixed(1)}%`;
}

function padR(s, w) { s = String(s); return s.length >= w ? s : s + ' '.repeat(w - s.length); }
function padL(s, w) { s = String(s); return s.length >= w ? s : ' '.repeat(w - s.length) + s; }

// ── Compute stats for a set of results (null errKey values are excluded) ───────
function stats(results, errKey) {
  const eligible = results.filter(r => r[errKey] !== null && r[errKey] !== undefined);
  const errs = eligible.map(r => r[errKey]).sort((a, b) => a - b);
  const n = errs.length;
  if (n === 0) return { n: 0, mean: 0, median: 0, p95: 0, max: 0, maxSample: null, within: () => 0 };
  const mean = errs.reduce((s, v) => s + v, 0) / n;
  const median = errs[Math.floor(n / 2)];
  const p95 = errs[Math.floor(n * 0.95)];
  const max = errs[n - 1];
  const maxSample = eligible.find(r => r[errKey] === max);
  const within = t => eligible.filter(r => r[errKey] <= t).length;
  return { n, mean, median, p95, max, maxSample, within };
}

// ── Print per-category summary row ────────────────────────────────────────────
function printCategoryTable(title, grouped, impls) {
  const categories = Object.keys(grouped).sort();
  const W = 16;
  const implW = 22;

  console.log(`\n${title}`);
  console.log('─'.repeat(title.length));

  // Header
  const hdr = [padR('category', W), padR('n', 4)].concat(
    impls.map(imp => padR(imp.label, implW))
  ).join('  ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  for (const cat of categories) {
    const rows = grouped[cat];
    const n = rows.length;
    const parts = [padR(cat, W), padR(n, 4)];
    for (const imp of impls) {
      const s = stats(rows, imp.errKey);
      const w10 = s.within(10);
      parts.push(padR(`${pct(w10, n)} ±${(((wilsonCI(w10, n)[1] - wilsonCI(w10, n)[0]) / 2) * 100).toFixed(1)}% (μ=${fmt(s.mean)}%)`, implW));
    }
    console.log(parts.join('  '));
  }
}

// ── Print overall summary block ────────────────────────────────────────────────
function printSummary(label, results, errKey) {
  const s = stats(results, errKey);
  const n = s.n;
  const excluded = results.length - n;
  console.log(`\n${label}`);
  console.log('─'.repeat(44));
  console.log(`n (eligible):    ${n}  (${excluded} excluded — expected < 5 tokens)`);
  console.log(`exact:           ${s.within(0)}  (${pct(s.within(0), n)})`);
  console.log(`within 5%:       ${s.within(5)}  (${pct(s.within(5), n)})`);
  console.log(`within 10%:      ${s.within(10)}  (${pct(s.within(10), n)})  ${ciStr(s.within(10), n)} Wilson 95% CI`);
  console.log(`mean abs err:    ${fmt(s.mean)}%`);
  console.log(`median abs err:  ${fmt(s.median)}%`);
  console.log(`p95 abs err:     ${fmt(s.p95)}%`);
  if (s.maxSample) console.log(`max abs err:     ${fmt(s.max)}%  (${s.maxSample.name})`);
}

// ── Per-sample table ──────────────────────────────────────────────────────────
function printSampleTable(results, impls) {
  const W = 32;
  const hdr = [padR('sample', W), padL('expected', 9)].concat(
    impls.flatMap(imp => [padL(imp.label, 8), padL(imp.label.slice(0, 5) + '%', 7)])
  ).join('  ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  let lastCat = null;
  for (const r of results) {
    if (r.category !== lastCat) {
      if (lastCat !== null) console.log('');
      console.log(`[${r.category}]`);
      lastCat = r.category;
    }
    const parts = [padR(r.name, W), padL(r.expected, 9)];
    for (const imp of impls) {
      const val = r[imp.valKey];
      const err = r[imp.errKey];
      const diff = val - r.expected;
      const dirStr = diff > 0 ? `+${diff}` : diff < 0 ? `${diff}` : '=';
      const errStr = err === null ? '  n/a ' : padL(`${fmt(err)}%`, 7);
      parts.push(padL(`${val}(${dirStr})`, 8), errStr);
    }
    console.log(parts.join('  '));
  }
}

function main() {
  console.log(`accuracy — bpe-lite vs ai-tokenizer vs Anthropic API`);
  console.log(`corpus: ${corpus.samples.length} samples | model: ${corpus.model} | fetched: ${corpus.fetchedAt}`);
  if (!aiTok) console.log('Note: ai-tokenizer not available — ai-tok column will show N/A');

  const impls = [
    { label: 'bpe-lite', valKey: 'bpeVal', errKey: 'bpePct' },
  ];
  if (aiTok) impls.push({ label: 'ai-tok', valKey: 'aiVal', errKey: 'aiPct' });

  // Samples with expected < 5 tokens are excluded from % error stats: the ±1
  // boundary calibration artifact (overhead varies by first-char class) represents
  // >20% relative error at that scale, making the % meaningless.
  const MIN_TOKENS_FOR_PCT = 5;

  const results = corpus.samples.map(s => {
    const bpeVal = countTokens(s.text, 'anthropic');
    const countable = s.expected >= MIN_TOKENS_FOR_PCT;
    const bpePct = countable ? (Math.abs(bpeVal - s.expected) / s.expected) * 100 : null;
    const row = { category: s.category, name: s.name, expected: s.expected, countable, bpeVal, bpePct };
    if (aiTok) {
      const aiVal = aiTok.count(s.text);
      row.aiVal = aiVal;
      row.aiPct = countable ? (Math.abs(aiVal - s.expected) / s.expected) * 100 : null;
    }
    return row;
  });

  // Group by category
  const grouped = {};
  for (const r of results) {
    (grouped[r.category] ??= []).push(r);
  }

  console.log('\n');
  printSampleTable(results, impls);

  printCategoryTable('Per-category: within-10% rate  (μ=mean abs err%)', grouped, impls);

  for (const imp of impls) {
    printSummary(`Overall — ${imp.label}`, results, imp.errKey);
  }
}

main();
