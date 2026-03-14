#!/usr/bin/env node
'use strict';

/**
 * Micro-benchmarks for bpe-lite vs common JS tokenizers.
 *
 * Run:
 *   node --expose-gc scripts/bench.js
 *
 * Notes:
 * - Numbers vary by machine; treat as directional.
 * - We benchmark "encode" because it dominates token counting work.
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

const jsTiktoken = tryRequire('js-tiktoken');
const aiTokenizer = tryRequire('ai-tokenizer');
const gptTokenizer = tryRequire('gpt-tokenizer');
const gptTokenizerO200k = tryRequire('gpt-tokenizer/model/gpt-4o');

function nowNs() {
  return process.hrtime.bigint();
}

function formatNumber(n) {
  if (!Number.isFinite(n)) return String(n);
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}k`;
  return `${n.toFixed(2)}`;
}

function padRight(s, w) {
  s = String(s);
  return s.length >= w ? s : s + ' '.repeat(w - s.length);
}

function makeText(kind) {
  if (kind === 'small') return 'Hello, world! café 😀 12345';
  if (kind === 'medium') {
    return (
      'This is a medium-sized benchmark string. ' +
      'It contains punctuation, numbers (12345), unicode like café and 😀, ' +
      'and multiple sentences.\n'
    ).repeat(40);
  }
  // large
  const para = (
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
    'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ' +
    'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ' +
    'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ' +
    'café 😀 12345\n'
  );
  // Keep "large" reasonably sized so bpe-lite finishes on slower machines.
  // If you want a heavier run, increase repeats locally.
  return para.repeat(120); // ~50–80KB depending on platform newlines
}

function gcIfPossible() {
  if (typeof global.gc === 'function') global.gc();
}

function benchOne({ name, fn, text, iters, warmup }) {
  // warmup
  for (let i = 0; i < warmup; i++) fn(text);

  gcIfPossible();

  let totalTokens = 0;
  const t0 = nowNs();
  for (let i = 0; i < iters; i++) {
    const ids = fn(text);
    totalTokens += ids.length;
  }
  const t1 = nowNs();

  const seconds = Number(t1 - t0) / 1e9;
  const opsPerSec = iters / seconds;
  const tokensPerSec = totalTokens / seconds;
  const bytesPerSec = (Buffer.byteLength(text, 'utf8') * iters) / seconds;

  return {
    name,
    iters,
    seconds,
    opsPerSec,
    tokensPerSec,
    bytesPerSec,
  };
}

function printTable(title, rows) {
  console.log('');
  console.log(title);
  console.log('─'.repeat(title.length));

  const header = [
    padRight('impl', 20),
    padRight('ops/s', 12),
    padRight('tokens/s', 12),
    padRight('MB/s', 10),
    padRight('iters', 8),
    'time(s)',
  ].join('  ');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const r of rows) {
    const line = [
      padRight(r.name, 20),
      padRight(formatNumber(r.opsPerSec), 12),
      padRight(formatNumber(r.tokensPerSec), 12),
      padRight(formatNumber(r.bytesPerSec / (1024 * 1024)), 10),
      padRight(r.iters, 8),
      r.seconds.toFixed(3),
    ].join('  ');
    console.log(line);
  }
}

function limitTextByBytes(text, maxBytes) {
  if (Buffer.byteLength(text, 'utf8') <= maxBytes) return text;
  let lo = 0;
  let hi = text.length;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    const chunk = text.slice(0, mid);
    if (Buffer.byteLength(chunk, 'utf8') <= maxBytes) lo = mid;
    else hi = mid;
  }
  return text.slice(0, lo);
}

function main() {
  const encCl100k = jsTiktoken ? jsTiktoken.getEncoding('cl100k_base') : null;
  const encO200k = jsTiktoken ? jsTiktoken.getEncoding('o200k_base') : null;

  const AiTok = aiTokenizer ? (aiTokenizer.Tokenizer || aiTokenizer.default) : null;
  const aiCl100k = aiTokenizer ? tryRequire('ai-tokenizer/encoding/cl100k_base') : null;
  const aiO200k = aiTokenizer ? tryRequire('ai-tokenizer/encoding/o200k_base') : null;
  const aiClaude = aiTokenizer ? tryRequire('ai-tokenizer/encoding/claude') : null;
  const aiTokCl = AiTok && aiCl100k ? new AiTok(aiCl100k) : null;
  const aiTokO2 = AiTok && aiO200k ? new AiTok(aiO200k) : null;
  const aiTokClaude = AiTok && aiClaude ? new AiTok(aiClaude) : null;

  const openaiImpls = [
    { name: 'bpe-lite cl100k', fn: (t) => bpeLite.encode(t, 'openai') },
    { name: 'bpe-lite o200k', fn: (t) => bpeLite.encode(t, 'openai-o200k') },
  ];

  const anthropicImpls = [
    { name: 'bpe-lite anthropic', fn: (t) => bpeLite.encode(t, 'anthropic') },
  ];

  const geminiImpls = [
    { name: 'bpe-lite gemini', fn: (t) => bpeLite.encode(t, 'gemini') },
  ];

  if (gptTokenizer && gptTokenizerO200k) {
    openaiImpls.push(
      { name: 'gpt-tokenizer cl100k', fn: (t) => gptTokenizer.encode(t) },
      { name: 'gpt-tokenizer o200k', fn: (t) => gptTokenizerO200k.encode(t) },
    );
  } else {
    console.log('Note: gpt-tokenizer not installed; skipping gpt-tokenizer benchmarks.');
  }

  if (encCl100k && encO200k) {
    openaiImpls.push(
      { name: 'js-tiktoken cl100k', fn: (t) => encCl100k.encode(t, 'all') },
      { name: 'js-tiktoken o200k', fn: (t) => encO200k.encode(t, 'all') },
    );
  } else {
    console.log('Note: js-tiktoken not installed; skipping js-tiktoken benchmarks.');
  }

  if (aiTokCl && aiTokO2) {
    openaiImpls.push(
      { name: 'ai-tokenizer cl100k', fn: (t) => aiTokCl.encode(t, 'all') },
      { name: 'ai-tokenizer o200k', fn: (t) => aiTokO2.encode(t, 'all') },
    );
  } else {
    console.log('Note: ai-tokenizer not installed; skipping ai-tokenizer OpenAI benchmarks.');
  }

  if (aiTokClaude) {
    anthropicImpls.push({ name: 'ai-tokenizer claude', fn: (t) => aiTokClaude.encode(t, 'all') });
  } else {
    console.log('Note: ai-tokenizer claude encoding not available; skipping Anthropic comparison.');
  }

  if (aiTokO2) {
    geminiImpls.push({ name: 'ai-tokenizer o200k', fn: (t) => aiTokO2.encode(t, 'all') });
  } else {
    console.log('Note: ai-tokenizer o200k encoding not available; skipping Gemini comparison.');
  }

  const scenarios = [
    // Defaults tuned to finish quickly on slower JS implementations.
    { kind: 'small', iters: 10_000, warmup: 500 },
    { kind: 'medium', iters: 400, warmup: 50 },
    { kind: 'large', iters: 8, warmup: 1 },
  ];

  console.log('bpe-lite benchmark');
  console.log(`node ${process.version} (${process.platform}/${process.arch})`);
  console.log(`gc: ${typeof global.gc === 'function' ? 'available' : 'not available (run with --expose-gc)'}`);
  console.log('Note: bpe-lite "anthropic" uses Xenova/claude-tokenizer (65k BPE); "gemini" uses Gemma3 SPM. Cross-provider comparisons are not apples-to-apples.');

  for (const s of scenarios) {
    const text = makeText(s.kind);
    const gemText =
      s.kind === 'small'
        ? text
        : s.kind === 'medium'
        ? limitTextByBytes(text, 5440)
        : text; // large: same text as OpenAI/Anthropic for a fair comparison

    const groups = [
      { title: `OpenAI encodings — ${s.kind} text`, impls: openaiImpls, iters: s.iters, warmup: s.warmup, text },
      { title: `Anthropic — ${s.kind} text`, impls: anthropicImpls, iters: s.iters, warmup: s.warmup, text },
      {
        title: `Gemini — ${s.kind} text (truncated)`,
        impls: geminiImpls,
        iters: s.kind === 'small' ? Math.min(2_000, s.iters) : s.kind === 'medium' ? 20 : s.iters,
        warmup: s.kind === 'small' ? 50 : 3,
        text: gemText,
      },
    ];

    for (const g of groups) {
      const rows = [];
      const bytes = Buffer.byteLength(g.text, 'utf8');
      for (const impl of g.impls) {
        rows.push(benchOne({ name: impl.name, fn: impl.fn, text: g.text, iters: g.iters, warmup: g.warmup }));
      }
      rows.sort((a, b) => b.tokensPerSec - a.tokensPerSec);
      printTable(`${g.title} (bytes=${bytes})`, rows);
    }
  }
}

main();
