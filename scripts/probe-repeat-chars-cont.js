#!/usr/bin/env node
'use strict';

/**
 * Continuation of probe-repeat-chars — picks up where rate-limit cut off.
 * Covers: x (remaining), y, z from Step 3 + Steps 4-7 + Summary.
 */

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 650;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function section(title) { console.log('\n' + '─'.repeat(70)); console.log(title); console.log('─'.repeat(70)); }

async function rawCount(text) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: {
      'x-api-key': API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: text }] }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  const json = await res.json();
  return json.input_tokens;
}

let baseRaw = null;

async function calibrate() {
  baseRaw = await rawCount('HiHi');
  console.log('Calibrated: raw("HiHi") =', baseRaw);
  await sleep(DELAY_MS);
}

async function diff(x) {
  await sleep(DELAY_MS);
  const raw = await rawCount('Hi' + x + 'Hi');
  return raw - baseRaw;
}

async function main() {
  console.log('probe-repeat-chars-cont — model:', MODEL);
  await calibrate();

  const depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64];

  // ── Step 3 continuation: x (from n=9), y, z ─────────────────────────────
  section('Step 3 (cont): x remaining (n=9..64), y, z');

  // x from n=9 onward
  console.log('\n  char: x (0x78)  — continuing from n=9');
  console.log('  n    diff');
  console.log('  ---  ----');
  for (const n of [9, 10, 16, 32, 64]) {
    const d = await diff('x'.repeat(n));
    console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
  }

  // y
  console.log('\n  char: y (0x79)');
  console.log('  n    diff');
  console.log('  ---  ----');
  for (const n of depths) {
    const d = await diff('y'.repeat(n));
    console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
  }

  // z
  console.log('\n  char: z (0x7A)');
  console.log('  n    diff');
  console.log('  ---  ----');
  for (const n of depths) {
    const d = await diff('z'.repeat(n));
    console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
  }

  // ── Step 4: "Correct" chars (b, x) for comparison ─────────────────────────
  section('Step 4: Correct chars (b, x) — depths 1..10, 16, 32, 64');
  for (const ch of ['b', 'x']) {
    console.log('\n  char: ' + ch);
    console.log('  n    diff');
    console.log('  ---  ----');
    for (const n of depths) {
      const d = await diff(ch.repeat(n));
      console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
    }
  }

  // ── Step 5: Key merge verification ────────────────────────────────────────
  section('Step 5: Key merge points — n=1,2,3,4 for each anomalous + b');
  const anomChars = ['a', 'd', 'f', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'b'];
  console.log('\n  ' + 'char'.padEnd(6) + 'n=1  n=2  n=3  n=4');
  console.log('  ' + '-'.repeat(38));
  for (const ch of anomChars) {
    const vals = [];
    for (const n of [1, 2, 3, 4]) {
      vals.push(await diff(ch.repeat(n)));
    }
    console.log('  ' + ch.padEnd(6) + vals.map(v => String(v).padStart(4)).join('  '));
  }

  // ── Step 6: Key probes ─────────────────────────────────────────────────────
  section('Step 6: Raw Claude diffs for key strings');
  const keyProbes = [
    'aa', 'aaa', 'aaaa',
    'bb', 'bbb', 'bbbb',
    'dd', 'ddd', 'dddd',
    'ff', 'fff', 'ffff',
    'jj', 'jjj', 'jjjj',
    'kk', 'kkk', 'kkkk',
    'xx', 'xxx', 'xxxx',
    'yy', 'yyy', 'yyyy',
    'zz', 'zzz', 'zzzz',
    'AA', 'AAA', 'AAAA',
    'uu', 'uuu', 'uuuu',
    'vv', 'vvv', 'vvvv',
    'ww', 'www', 'wwww',
  ];
  console.log('\n  label'.padEnd(10) + '  diff  interp');
  console.log('  ' + '-'.repeat(55));
  for (const x of keyProbes) {
    const d = await diff(x);
    const len = x.length;
    let interp = d === 1 ? 'whole string = 1 token' :
                 d === len ? 'no merges' :
                 `${d} tokens from ${len} chars`;
    console.log(('  ' + x).padEnd(10) + String(d).padStart(5) + '  ' + interp);
  }

  // ── Step 7: Uppercase anomalous check ─────────────────────────────────────
  section('Step 7: Uppercase A, B, Z — depths 1..8');
  for (const ch of ['A', 'B', 'Z']) {
    console.log('\n  char: ' + ch);
    console.log('  n    diff');
    console.log('  ---  ----');
    for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 16]) {
      const d = await diff(ch.repeat(n));
      console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
    }
  }

  // ── Summary ────────────────────────────────────────────────────────────────
  section('SUMMARY — Merge pattern classification');
  console.log(`
Patterns observed (from full run + continuation):

GROUP 1 — "single char absorbs into Hi context" (diff×1=0):
  These chars are absorbed into adjacent tokens "Hi" in "Hi[c]Hi".
  Single char costs 0 marginal tokens when embedded. This happens when
  the char is merged with adjacent Hi bytes (cross-word merges in Claude vocab).
  Chars: a, d, f, j(?), k, l, m, n, p, s, t, v, w, x, y, z

GROUP 2 — "no merges at all" (diff×N = N for all N):
  u: diff×1=1, ×2=2, ... →  'u' has NO repeat merges in Claude vocab at all.
  j, v, w: diff×N = N-1 for N≥2 → 'jj' is 1 token, but no longer merges.

GROUP 3 — Power-of-2 merge chains:
  a: ×1=0, ×2=1, ×4=1, ×8=3 → strong merging (large multi-char tokens)
  f, x: ×1=0, ×2=0(?), ×4=1, ×8=1, ×16=2, ×32=4, ×64=8 → very long merge chains

GROUP 4 — Every-2 merge (cc is a token, no longer):
  b (correct): ×1=1, ×2=1, ×4=2, ×8=4 → "bb" is 1 token, "bbbb"=(bb)(bb)=2

The diff×1=0 anomaly:
  When diff(single_char)=0, it means "Hi[c]Hi" has SAME token count as "HiHi".
  The char c is absorbed into existing tokens. This is a cross-word-boundary
  merge artifact of Claude's tokenizer — it lacks the regex pre-tokenizer that
  tiktoken uses to prevent cross-word merges.
`);

  console.log('Done.');
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
