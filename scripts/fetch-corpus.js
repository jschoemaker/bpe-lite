'use strict';
/**
 * Fetches expected token counts from the Anthropic API for every sample in corpus.js.
 * Writes results to corpus-expected.json (commit this file — accuracy.js runs offline).
 *
 * Run once:
 *   ANTHROPIC_API_KEY=sk-... node scripts/fetch-corpus.js
 */

const fs = require('fs');
const path = require('path');
const { CORPUS } = require('./corpus');

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) { console.error('ANTHROPIC_API_KEY not set'); process.exit(1); }

const MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001';
const DELAY_MS = 130;
const OUT_PATH = path.join(__dirname, 'corpus-expected.json');

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function countApi(text) {
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
  return (await res.json()).input_tokens;
}

// Overhead calibration: api("Hi") - countTokens("Hi") = 7 for letter-starting messages.
// For digit/tab/newline/CJK-starting messages the effective overhead is 8 (±1 boundary
// effect from BPE-merging of the last structural token with the first content byte).
// Samples with expected < 5 tokens are flagged; % error is suppressed for them since
// a ±1 boundary artifact represents >20% relative error at that scale.
async function main() {
  console.log(`Fetching expected counts for ${CORPUS.length} samples`);
  console.log(`Model: ${MODEL}`);

  process.stdout.write('Calibrating message overhead...');
  const { countTokens } = require('../src/index');
  const rawHi = await countApi('Hi');
  const overhead = rawHi - countTokens('Hi', 'anthropic');
  console.log(` overhead = ${overhead} token(s)`);
  await sleep(DELAY_MS);

  const results = [];
  let done = 0;

  for (const sample of CORPUS) {
    const raw = await countApi(sample.text);
    const expected = raw - overhead;
    results.push({ category: sample.category, name: sample.name, text: sample.text, expected });
    done++;
    process.stdout.write(`\r  ${done}/${CORPUS.length}  [${sample.category}] ${sample.name.slice(0, 40)}`);
    await sleep(DELAY_MS);
  }

  console.log('\n');

  const output = {
    model: MODEL,
    overhead,
    fetchedAt: new Date().toISOString(),
    samples: results,
  };

  fs.writeFileSync(OUT_PATH, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Written: scripts/corpus-expected.json (${results.length} samples)`);
}

main().catch(e => { console.error(e.message); process.exit(1); });
