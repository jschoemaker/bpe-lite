#!/usr/bin/env node
'use strict';

/**
 * Compatibility check against reference JS tokenizers.
 *
 * Focuses on OpenAI encodings:
 * - bpe-lite provider "openai"      ↔ js-tiktoken "cl100k_base"
 * - bpe-lite provider "openai-o200k"↔ js-tiktoken "o200k_base"
 *
 * Run:
 *   node scripts/compat.js
 */

const bpeLite = require('../src/index');

function tryRequire(name) {
  try {
    // eslint-disable-next-line import/no-dynamic-require, global-require
    return require(name);
  } catch {
    return null;
  }
}

function stableJson(x) {
  return JSON.stringify(Array.isArray(x) ? x : Array.from(x));
}

function fail(label, details) {
  console.error(`✗ ${label}`);
  if (details) console.error(details);
  process.exitCode = 1;
}

function ok(label) {
  console.log(`✓ ${label}`);
}

function randInt(n) {
  return Math.floor(Math.random() * n);
}

function randomString() {
  const parts = [
    'hello', 'world', 'café', '😀', '—', 'こんにちは', 'مرحبا', '12345',
    '\n', '\r\n', '\t', ' ', '  ', 'function foo() { return 42; }',
    '{"key":"value","n":123}', '<|endoftext|>', '<|fim_prefix|>',
  ];
  let out = '';
  const count = 2 + randInt(10);
  for (let i = 0; i < count; i++) out += parts[randInt(parts.length)];
  return out;
}

function compareOne({ label, a, b }) {
  const sa = stableJson(a);
  const sb = stableJson(b);
  if (sa !== sb) {
    fail(label, `  expected: ${sb.slice(0, 240)}${sb.length > 240 ? '…' : ''}\n  got:      ${sa.slice(0, 240)}${sa.length > 240 ? '…' : ''}`);
    return false;
  }
  ok(label);
  return true;
}

function main() {
  const jsTiktoken = tryRequire('js-tiktoken');
  if (!jsTiktoken) {
    console.error('Missing dependency: js-tiktoken');
    console.error('Install with: npm install --no-fund --no-audit js-tiktoken');
    process.exit(2);
  }

  const { getEncoding } = jsTiktoken;
  const cl100k = getEncoding('cl100k_base');
  const o200k = getEncoding('o200k_base');

  const fixtures = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    '😀 café 12345',
    'hello<|endoftext|>world',
    '<|endoftext|>',
    '<|fim_prefix|>test<|fim_suffix|>',
    ' ' + 'Hello '.repeat(20),
  ];

  for (const text of fixtures) {
    compareOne({
      label: `openai(cl100k) ids match for ${JSON.stringify(text).slice(0, 60)}`,
      a: bpeLite.encode(text, 'openai'),
      b: cl100k.encode(text, 'all'),
    });
    compareOne({
      label: `openai-o200k ids match for ${JSON.stringify(text).slice(0, 60)}`,
      a: bpeLite.encode(text, 'openai-o200k'),
      b: o200k.encode(text, 'all'),
    });
  }

  // Light fuzzing: random unicode-ish strings, including some special-token-like substrings.
  for (let i = 0; i < 50; i++) {
    const text = randomString();
    compareOne({
      label: `fuzz[${i}] openai ids match`,
      a: bpeLite.encode(text, 'openai'),
      b: cl100k.encode(text, 'all'),
    });
    compareOne({
      label: `fuzz[${i}] openai-o200k ids match`,
      a: bpeLite.encode(text, 'openai-o200k'),
      b: o200k.encode(text, 'all'),
    });
  }

  console.log('');
  console.log(process.exitCode ? 'Compatibility: FAIL' : 'Compatibility: OK');
}

main();

