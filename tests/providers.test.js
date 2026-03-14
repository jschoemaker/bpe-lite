'use strict';

const { countTokens, encode, decode, openai, anthropic, gemini } = require('../src/index');

let passed = 0;
let failed = 0;

function assert(condition, label) {
  if (condition) {
    console.log(`  ✓ ${label}`);
    passed++;
  } else {
    console.log(`  ✗ ${label}`);
    failed++;
  }
}

function assertEqual(a, b, label) {
  const ok = JSON.stringify(a) === JSON.stringify(b);
  if (!ok) console.log(`    expected: ${JSON.stringify(b)}\n    got:      ${JSON.stringify(a)}`);
  assert(ok, label);
}

console.log('Provider API tests');
console.log('─'.repeat(40));

// ── countTokens API ──────────────────────────────────────────────────────────

assertEqual(countTokens('Hello, world!', 'openai'), 4, 'openai: "Hello, world!" = 4');
assertEqual(countTokens('Hello, world!', 'anthropic'), 4, 'anthropic: "Hello, world!" = 4');
assertEqual(countTokens('Hello, world!', 'gemini'), 4, 'gemini: "Hello, world!" = 4');

assertEqual(countTokens('', 'openai'), 0, 'openai: empty = 0');
assertEqual(countTokens('', 'gemini'), 0, 'gemini: empty = 0');

assertEqual(countTokens(' ', 'openai'), 1, 'openai: space = 1');

// Default provider is openai
assertEqual(countTokens('hello'), countTokens('hello', 'openai'), 'default provider = openai');

// ── encode / decode API ──────────────────────────────────────────────────────

const ids = encode('Hello', 'openai');
assert(Array.isArray(ids) && ids.length > 0, 'encode returns non-empty array');
assertEqual(decode(ids, 'openai'), 'Hello', 'decode inverts encode');

// ── Tokenizer instances ──────────────────────────────────────────────────────

const tok = openai();
assert(typeof tok.encode === 'function', 'openai() has encode');
assert(typeof tok.decode === 'function', 'openai() has decode');
assert(typeof tok.count === 'function', 'openai() has count');
assertEqual(tok.count('Hello, world!'), 4, 'openai instance: count');
assertEqual(tok.decode(tok.encode('test')), 'test', 'openai instance: round-trip');

const atok = anthropic();
assertEqual(atok.count('Hello, world!'), 4, 'anthropic instance: count');

const gtok = gemini();
assertEqual(gtok.count('Hello, world!'), 4, 'gemini instance: count');
assert(gtok.count('The quick brown fox') > 0, 'gemini instance: encodes text');

// ── Lazy caching — same instance returned ────────────────────────────────────

assert(openai() === openai(), 'openai() returns cached instance');
assert(gemini() === gemini(), 'gemini() returns cached instance');

// ── Counts are plausible (within 2x of length/4 heuristic) ──────────────────

const sample = 'This is a sample sentence for token count validation.';
const heuristic = Math.ceil(sample.length / 4);
const actual = countTokens(sample, 'openai');
assert(actual >= heuristic * 0.5 && actual <= heuristic * 2,
  `openai count (${actual}) within 2x of heuristic (${heuristic})`);

const actualGemini = countTokens(sample, 'gemini');
assert(actualGemini >= heuristic * 0.5 && actualGemini <= heuristic * 2,
  `gemini count (${actualGemini}) within 2x of heuristic (${heuristic})`);

// ── Cross-provider round-trips ────────────────────────────────────────────────

const texts = [
  'Hello, world!',
  'function foo() { return 42; }',
  'The quick brown fox jumps over the lazy dog.',
  '{"key": "value", "num": 123}',
];

for (const t of texts) {
  const rt = decode(encode(t, 'openai'), 'openai');
  assertEqual(rt, t, `openai round-trip: ${JSON.stringify(t).slice(0, 30)}`);
}

for (const t of texts) {
  const rt = gtok.decode(gtok.encode(t));
  assertEqual(rt, t, `gemini round-trip: ${JSON.stringify(t).slice(0, 30)}`);
}

console.log('');
console.log(`Providers: ${passed} passed, ${failed} failed`);
console.log('');

module.exports = { passed, failed };
