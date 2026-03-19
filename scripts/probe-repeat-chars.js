#!/usr/bin/env node
'use strict';

/**
 * Probe Claude Haiku 4.5 repeated-char tokenization using differential method.
 * diff("Hi" + X + "Hi") - diff("HiHi") cancels overhead exactly.
 *
 * Run: ANTHROPIC_API_KEY=sk-... node scripts/probe-repeat-chars.js
 */

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 650;
const OVERHEAD = 7;

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

let baseRaw = null; // raw count for "HiHi"

async function diff(x) {
  await sleep(DELAY_MS);
  const raw = await rawCount('Hi' + x + 'Hi');
  return raw - baseRaw;
}

async function main() {
  console.log('probe-repeat-chars — model:', MODEL, '  overhead:', OVERHEAD);

  // ── Step 1: Calibrate ──────────────────────────────────────────────────────
  section('Step 1: Calibrate base');
  const rawHi = await rawCount('Hi');
  console.log('raw("Hi") =', rawHi, '  → overhead =', rawHi - 1, '  (expected', OVERHEAD, ')');
  await sleep(DELAY_MS);

  baseRaw = await rawCount('HiHi');
  const hiHiText = baseRaw - OVERHEAD;
  console.log('raw("HiHi") =', baseRaw, '  → text tokens =', hiHiText, '  (expected 2)');
  await sleep(DELAY_MS);

  // ── Step 2: All 26 lowercase ×4 ───────────────────────────────────────────
  section('Step 2: All 26 lowercase letters ×4  (diff method)');
  console.log('char  diff  notes');
  console.log('----  ----  -----');
  const anomalous = [];
  for (let i = 0; i < 26; i++) {
    const ch = String.fromCharCode(97 + i); // a-z
    const x4 = ch.repeat(4);
    const d = await diff(x4);
    // BPE with "xx" merge would give 2 tokens for x×4 ("xx"+"xx")
    // BPE with no merge would give 4 tokens
    // Expected: 2 if "xx" merge exists and no "xxxx" or weird interactions
    const expected = 2; // default assumption: "ch+ch" merge exists → 2 tokens
    const flag = d !== expected ? ' <-- ANOMALOUS (expected ' + expected + ')' : '';
    console.log(('  ' + ch + '×4').padEnd(6) + String(d).padStart(4) + flag);
    if (d !== expected) anomalous.push({ ch, d4: d });
  }

  // ── Step 3: Probe anomalous + known-correct chars at depths 1-10, 16, 32, 64 ──
  const depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64];
  const charsToProbe = [];

  // Add anomalous chars found above
  for (const { ch } of anomalous) {
    if (!charsToProbe.includes(ch)) charsToProbe.push(ch);
  }
  // Always add known anomalous from prior knowledge
  for (const ch of ['a', 'z', 'A']) {
    if (!charsToProbe.includes(ch)) charsToProbe.push(ch);
  }

  section('Step 3: Anomalous chars — depths 1..10, 16, 32, 64');
  for (const ch of charsToProbe) {
    console.log('\n  char: ' + ch + ' (0x' + ch.charCodeAt(0).toString(16).toUpperCase() + ')');
    console.log('  n    diff');
    console.log('  ---  ----');
    for (const n of depths) {
      const d = await diff(ch.repeat(n));
      console.log(('  ' + n).padEnd(5) + String(d).padStart(4));
    }
  }

  // ── Step 4: "Correct" chars for comparison ─────────────────────────────────
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

  // ── Step 5: Verify "aa" is truly 1 token ──────────────────────────────────
  section('Step 5: Verify key merge points for anomalous chars');
  const verifyChars = [...new Set([...charsToProbe, 'b'])];
  console.log('\n  Testing n=1,2,3,4 for each char:');
  console.log(('  char').padEnd(8) + 'n=1  n=2  n=3  n=4');
  console.log('  ' + '-'.repeat(40));
  for (const ch of verifyChars) {
    const vals = [];
    for (const n of [1, 2, 3, 4]) {
      vals.push(await diff(ch.repeat(n)));
    }
    console.log(('  ' + ch).padEnd(8) + vals.map(v => String(v).padStart(4)).join('  '));
  }

  // ── Step 6: Merge inference ────────────────────────────────────────────────
  section('Step 6: Merge structure inference');
  console.log('\nFor each anomalous char, explain what the diff sequence implies:');
  console.log('(Notation: "xx"=two-char merge, "xxx"=three-char merge, etc.)');

  for (const ch of charsToProbe) {
    console.log('\n  char: ' + ch);
    // Re-probe n=1..6 for inference
    const diffs = {};
    for (const n of [1, 2, 3, 4, 5, 6]) {
      diffs[n] = await diff(ch.repeat(n));
    }
    console.log('  diffs: ' + Object.entries(diffs).map(([k,v]) => `×${k}=${v}`).join('  '));

    // Try to infer
    const d1 = diffs[1], d2 = diffs[2], d3 = diffs[3], d4 = diffs[4];
    const d5 = diffs[5], d6 = diffs[6];

    const inferences = [];

    // d1 tells us the token count for a single char (should always be 1)
    if (d1 !== 1) inferences.push(`WARNING: single ${ch} costs ${d1} tokens (expected 1)`);

    // d2: 1 means "cc" is a merge, 2 means no merge
    if (d2 === 1) {
      inferences.push(`"${ch}${ch}" IS a merge (d2=1)`);
    } else if (d2 === 2) {
      inferences.push(`"${ch}${ch}" is NOT a merge (d2=2)`);
    }

    // d3 given d2=1 (cc is a merge):
    //   d3=1 → "ccc" is a merge (c + cc → ccc), or (cc + c → ccc)
    //   d3=2 → no 3-char merge; "ccc" tokenizes as (cc)(c) or (c)(cc) = 2 tokens
    if (d2 === 1) {
      if (d3 === 1) inferences.push(`"${ch.repeat(3)}" IS a merge (d3=1)`);
      else if (d3 === 2) inferences.push(`"${ch.repeat(3)}" is NOT a merge → splits as (${ch+ch})(${ch}) = 2 tokens`);
    }

    // d4 given known d2,d3:
    if (d2 === 1 && d3 === 1) {
      // cccc: if cccc is a merge → d4=1; else (ccc+c) or (c+ccc) → 2; or (cc+cc) → 2
      if (d4 === 1) inferences.push(`"${ch.repeat(4)}" IS a merge (d4=1)`);
      else if (d4 === 2) inferences.push(`"${ch.repeat(4)}" splits into 2 tokens: likely (${ch.repeat(3)})(${ch}) or (${ch+ch})(${ch+ch})`);
      else if (d4 === 3) inferences.push(`"${ch.repeat(4)}" = 3 tokens (unexpected with 3-char merge)`);
    } else if (d2 === 1 && d3 === 2) {
      // cc is a merge, ccc is not
      // cccc: (cc)(cc) = 2 tokens expected
      if (d4 === 2) inferences.push(`"${ch.repeat(4)}" = (${ch+ch})(${ch+ch}) = 2 tokens ✓`);
      else if (d4 === 3) {
        inferences.push(`"${ch.repeat(4)}" = 3 tokens! BPE prioritizes something unexpected.`);
        inferences.push(`  Possible: Claude has "ccc" but our vocab doesn't, or regex pre-tokenization differs.`);
        inferences.push(`  Or: Claude has NO "cc" merge (our vocab wrong), and tokenizes a×4 as (a)(a)(a)(a)→collapse? Unlikely.`);
        inferences.push(`  Most likely: Claude has "ccc" as a token. "cccc" = (ccc)(c) = 2... but d4=3 contradicts this.`);
        inferences.push(`  Actually: if "cc" is NOT in Claude (but IS in our vocab), then d2=1 is correct (cc=1 in ours and Claude) but d4=3 means Claude sees (a)(a)(a)(a) with some 3-char merge that gives 3 tokens from 4 chars? That can't add up.`);
        inferences.push(`  Wait — re-check: diff=3 for ×4 means "Hi"+"aaaa"+"Hi" has 3 more tokens than "HiHi". So "aaaa" in context = 3 tokens.`);
        inferences.push(`  If Claude has "aaa" but not "aa": "aaaa" = (a)(aaa) or (aaa)(a) = 2? No that's 2.`);
        inferences.push(`  If Claude has neither "aa" nor "aaa": "aaaa" = 4 tokens. But d4=3.`);
        inferences.push(`  CONCLUSION: Claude may have NO "aa" merge for this char. d2=1 in ours but Claude gives d2=? Let us check below.`);
      }
    }

    // d5, d6 patterns
    if (d4 !== undefined) {
      const rate5 = d5 - d4;
      const rate6 = d6 - d5;
      inferences.push(`Growth: ×4→×5: +${rate5}, ×5→×6: +${rate6}`);
    }

    for (const inf of inferences) console.log('    ' + inf);
  }

  // ── Step 7: Cross-check — probe ONLY with Claude (no our-BPE comparison) ───
  section('Step 7: Raw Claude diffs for "aa", "aaa" etc. (key verification)');
  const keyProbes = [
    ['aa', 'aa'],
    ['aaa', 'aaa'],
    ['aaaa', 'aaaa'],
    ['bb', 'bb'],
    ['bbb', 'bbb'],
    ['bbbb', 'bbbb'],
    ['zz', 'zz'],
    ['zzz', 'zzz'],
    ['zzzz', 'zzzz'],
    ['AA', 'AA'],
    ['AAA', 'AAA'],
    ['AAAA', 'AAAA'],
  ];
  console.log('\n  label'.padEnd(14) + 'diff  interpretation');
  console.log('  ' + '-'.repeat(50));
  for (const [label, x] of keyProbes) {
    const d = await diff(x);
    // Interpret
    let interp = '';
    const len = x.length;
    if (d === 1) interp = `whole string is 1 token`;
    else if (d === len) interp = `no merges (each char separate)`;
    else if (d < len && d > 1) interp = `partial merging (${d} tokens from ${len} chars)`;
    console.log(('  ' + label).padEnd(14) + String(d).padStart(4) + '  ' + interp);
  }

  // ── Summary ────────────────────────────────────────────────────────────────
  section('SUMMARY');
  console.log('\nAnomaly check (all lowercase ×4, expected diff=2):');
  console.log('Anomalous chars found: ' + (anomalous.length ? anomalous.map(a => a.ch + '×4=' + a.d4).join(', ') : 'none'));
  console.log('\nKey findings:');
  console.log('  - diff("X") = 1 for any single ASCII letter (sanity check)');
  console.log('  - diff("cc") = 1 means "cc" is a token in Claude vocab');
  console.log('  - diff("cc") = 2 means "cc" is NOT a token (no merge)');
  console.log('  - If "aa" gives diff=1 but "aaaa" gives diff=3:');
  console.log('      → Claude has "aaa" token but NOT "aa" is wrong assumption');
  console.log('      → OR Claude regex splits differently, isolating each "a" from the Hi context');
  console.log('      → Most likely: Claude tokenizer has "aaa" (3-char) and NOT "aaaa"');
  console.log('         so "aaaa" = (aaa)(a) = 2 tokens? But that gives diff=2, not 3.');
  console.log('      → UNLESS "aa" is not a merge and a×4 = 4 separate tokens but wait...');
  console.log('      → Differential method rules out overhead. diff=3 for ×4 is solid.');
  console.log('  - This is a BPE ordering issue: if Claude has "aaa" but not "aa",');
  console.log('      "aaaa" BPE: start [a,a,a,a], merge [a,a,a]→[aaa], result [aaa,a] = 2 tokens');
  console.log('      But diff=3 implies 3 tokens. So Claude also lacks "aaa"?');
  console.log('      Then "aaaa" = 4 tokens → diff=4. Still not 3.');
  console.log('  - diff=3 from 4 chars requires exactly ONE merge happening.');
  console.log('      This means Claude has EXACTLY ONE of: "aa" or "aaa" or "aaaa" but arranged');
  console.log('      such that only one merge fires on "aaaa".');
  console.log('      If "aa" exists: "aaaa"=[aa,aa]=2. If "aaa" exists: "aaaa"=[aaa,a] or [a,aaa]=2.');
  console.log('      NONE of these give 3. diff=3 is impossible with standard BPE unless...');
  console.log('      the regex pre-tokenizer splits "aaaa" into ["a","a","a","a"] in CONTEXT.');
  console.log('  - CRITICAL HYPOTHESIS: Claude regex may limit repeat-char runs differently');
  console.log('      when adjacent to word chars (Hi...Hi context vs bare "aaaa").');

  console.log('\nDone.');
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
