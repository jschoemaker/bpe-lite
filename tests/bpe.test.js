'use strict';

const { encodeTiktoken, decodeTiktoken } = require('../src/bpe');
const fs = require('fs');
const path = require('path');

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

const zlib = require('zlib');
const vocabData = JSON.parse(
  zlib.gunzipSync(fs.readFileSync(path.join(__dirname, '..', 'vocabs', 'openai.json.gz')))
);

console.log('BPE (tiktoken) engine tests');
console.log('─'.repeat(40));

// Basic encoding
const helloIds = encodeTiktoken('Hello, world!', vocabData);
assert(Array.isArray(helloIds), 'encode returns array');
assertEqual(helloIds.length, 4, '"Hello, world!" = 4 tokens');

// Empty string
assertEqual(encodeTiktoken('', vocabData), [], 'empty string = []');

// Single space
const spaceIds = encodeTiktoken(' ', vocabData);
assertEqual(spaceIds.length, 1, 'single space = 1 token');

// Known token counts
assert(encodeTiktoken('The quick brown fox', vocabData).length === 4, '"The quick brown fox" = 4 tokens');
assert(encodeTiktoken('hello', vocabData).length === 1, '"hello" = 1 token');

// Decode round-trip
const text1 = 'Hello, world!';
const rt1 = decodeTiktoken(encodeTiktoken(text1, vocabData), vocabData);
assertEqual(rt1, text1, 'round-trip: "Hello, world!"');

const text2 = 'The quick brown fox jumps over the lazy dog.';
const rt2 = decodeTiktoken(encodeTiktoken(text2, vocabData), vocabData);
assertEqual(rt2, text2, 'round-trip: pangram');

const text3 = 'function add(a, b) { return a + b; }';
const rt3 = decodeTiktoken(encodeTiktoken(text3, vocabData), vocabData);
assertEqual(rt3, text3, 'round-trip: code snippet');

// Numbers
assert(encodeTiktoken('12345', vocabData).length >= 1, 'numbers encode');
const rtNum = decodeTiktoken(encodeTiktoken('12345', vocabData), vocabData);
assertEqual(rtNum, '12345', 'round-trip: numbers');

// Unicode
const unicodeText = 'café';
const rtUnicode = decodeTiktoken(encodeTiktoken(unicodeText, vocabData), vocabData);
assertEqual(rtUnicode, unicodeText, 'round-trip: unicode (café)');

// Long text
const longText = 'Hello '.repeat(100);
const longIds = encodeTiktoken(longText, vocabData);
assert(longIds.length > 0, 'long text encodes');
assertEqual(decodeTiktoken(longIds, vocabData), longText, 'round-trip: long text');

// Decode empty
assertEqual(decodeTiktoken([], vocabData), '', 'decode [] = ""');

console.log('');
console.log(`BPE: ${passed} passed, ${failed} failed`);
console.log('');

module.exports = { passed, failed };
