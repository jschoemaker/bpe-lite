# bpe-lite

Offline BPE tokenizer for OpenAI, Anthropic, and Gemini. Zero dependencies, no network calls at runtime. Works in any Node 18+ environment including Docker and edge runtimes. Ships CJS and ESM. TypeScript types included.

```js
const { countTokens } = require('bpe-lite');

countTokens('Hello, world!', 'openai-o200k') // → 4  (GPT-4o, o1, o3, o4, GPT-4.1, GPT-5)
countTokens('Hello, world!', 'openai')       // → 4  (GPT-4, GPT-3.5)
countTokens('Hello, world!', 'anthropic')    // → 4
countTokens('Hello, world!', 'gemini')       // → 4
```

## Install

```bash
npm install bpe-lite
```

## Usage

Both CommonJS and ESM are supported:

```js
// CommonJS
const { countTokens, encode, decode, isWithinTokenLimit, openai, openaiO200k, anthropic, gemini } = require('bpe-lite');

// ESM
import { countTokens, encode, decode, isWithinTokenLimit, openai, openaiO200k, anthropic, gemini } from 'bpe-lite';
```

```js
// Count tokens
countTokens('Your text here', 'openai-o200k'); // → number (GPT-4o, o1, o3, o4, GPT-4.1, GPT-5)
countTokens('Your text here', 'openai');       // → number (GPT-4, GPT-3.5)

// Encode / decode
const ids = encode('Hello', 'openai-o200k');   // → [13225]
decode(ids, 'openai-o200k');                   // → 'Hello'

// Check token limit — short-circuits early on long text, more efficient than encode()
// Returns the token count if within the limit, false if exceeded
isWithinTokenLimit('Hello, world!', 10, 'openai-o200k'); // → 4
isWithinTokenLimit('Hello, world!', 3,  'openai-o200k'); // → false

// Tokenizer instances — lazy-loaded and cached per provider
const tok = openaiO200k();
tok.encode('Hello');          // → [13225]
tok.decode([13225]);          // → 'Hello'
tok.count('Hello, world!');   // → 4
```

## Providers

| Provider | Vocab | Models | Accuracy |
|----------|-------|--------|----------|
| `openai-o200k` | o200k_base (200k) | GPT-4o, o1, o3, o4, GPT-4.1, GPT-5 | Exact — vocab sourced from OpenAI |
| `openai` | cl100k_base (100k) | GPT-4, GPT-3.5 | Exact — vocab sourced from OpenAI |
| `anthropic` | Xenova/claude-tokenizer (65k BPE) | Claude | See accuracy section below |
| `gemini` | Gemma 3 SPM (262k) | Gemini | See accuracy section below |

Vocab files are bundled in the package — no network required at runtime or install time.

## Accuracy — Anthropic

Anthropic has not released the Claude tokenizer. bpe-lite uses [`Xenova/claude-tokenizer`](https://huggingface.co/Xenova/claude-tokenizer), a community reverse-engineering of the ~65k BPE vocabulary, with hand-tuned byte-level corrections applied on top.

### Benchmark methodology

Tested against `claude-haiku-4-5-20251001` via the Anthropic `count_tokens` API on a 120-sample stratified corpus across 12 categories. The corpus deliberately over-represents difficult content (Arabic, symbols, emoji, numbers) to expose systematic failures — overall numbers are lower than you would see on typical prose-only workloads by design.

### Overall results (120 samples, 114 eligible)

| Metric | bpe-lite | ai-tokenizer |
|--------|----------|--------------|
| Within 5% | 46.5% | 18.4% |
| Within 10% | 62.3% ±8.8% CI | 37.7% ±8.8% CI |
| Mean abs error | 9.4% | 16.0% |
| Median abs error | 5.7% | 13.6% |
| Max abs error | 42.9% | 82.6% |

### Per-category breakdown

| Category | Within 10% | Mean error | Notes |
|----------|-----------|------------|-------|
| `code-js` | 100% | 4.2% | |
| `english-prose` | 90% | 5.5% | |
| `code-python` | 90% | 4.8% | |
| `structured` | 90% | 3.6% | JSON, HTML, XML, Markdown, SQL |
| `numbers` | 80% | 7.3% | |
| `hex-binary` | 80% | 5.3% | |
| `urls` | 80% | 3.6% | |
| `cjk` | 40% | 8.8% | |
| `short` | 30% | 6.8% | |
| `emoji` | 20% | 17.7% | ZWJ sequences, flags, skin tones |
| `symbols` | 10% | 17.6% | Cross-byte merges unreplicable |
| `arabic` | 0% | 26.1% | Structural vocabulary gap — unfixable |

For prose, code, structured data, and URLs — the dominant content types in real-world prompts — bpe-lite is within 10% on 80–100% of samples. Arabic and symbol-cluster-heavy content cannot be accurately estimated without the actual Claude tokenizer.

### Why bpe-lite outperforms ai-tokenizer on Claude

ai-tokenizer's `claude` encoding uses `\p{N}+` (greedy, unlimited digit chunks). Current Claude models use `\p{N}{1,3}` (1–3 digits). This causes 20–43% errors on anything involving numbers — including code, hex, and data. bpe-lite uses the correct pattern.

ai-tokenizer also does not have a Gemini encoding: all Gemini models are mapped to OpenAI's `o200k_base` vocabulary with a fudge multiplier. This is wrong by construction — see below.

## Accuracy — Gemini

bpe-lite implements the full Gemma 3 SentencePiece BPE algorithm using the actual Gemini vocabulary. On a 25-sample test against the Gemini API, bpe-lite scored 100% exact (no failures found; 25 samples is a limited basis — treat this as a lower bound, not a guarantee across all content types).

ai-tokenizer does not implement Gemini natively. Inspecting their bundled source (`dist/index.js`), every Gemini model is defined as `"encoding": "o200k_base"` with a `"contentMultiplier": 1.08` fudge factor — it runs the OpenAI vocabulary through a multiplier rather than using Gemini's actual tokenizer. bpe-lite uses the actual Gemma 3 vocabulary and SentencePiece algorithm.

## Performance

Benchmarked on Node v24 (win32/x64). Run `node --expose-gc scripts/bench.js` locally for numbers on your hardware.

**Large text (~500 KB) — ops/s**

| impl | cl100k | Anthropic | Gemini | note |
|------|-------:|----------:|-------:|------|
| bpe-lite | 291 | 289 | 998 | |
| ai-tokenizer | 291 | 291 | 215 | Gemini column uses o200k — wrong algorithm |
| js-tiktoken | 30 | — | — | WASM overhead |

bpe-lite matches ai-tokenizer throughput for OpenAI and Anthropic large text. On Gemini, bpe-lite's SPM engine is ~4.6x faster — the ai-tokenizer column there is not a valid comparison since it uses a different algorithm.

**Small text cold vs warm**

bpe-lite maintains a per-instance chunk cache. For repeated text patterns (e.g. the same prompt template encoded thousands of times), cache hits eliminate BPE work entirely:

| scenario | bpe-lite | ai-tokenizer |
|----------|----------|--------------|
| Small text, cold (novel input) | ~352k ops/s | ~453k ops/s |
| Small text, warm (repeated input) | ~1.45M ops/s | ~453k ops/s |

For diverse, non-repeating inputs, ai-tokenizer is ~29% faster on very short strings. For any repeated-text workload, bpe-lite is ~3x faster.

**Initialization**

bpe-lite lazy-loads the gzipped vocab on first encode — one-time cost of ~235ms per provider per process. Negligible for any persistent process. Relevant only for cold serverless invocations that encode once and exit.

## API

### `countTokens(text, provider?)`

Returns the number of tokens in `text`. Default provider: `'openai'`.

### `encode(text, provider?)`

Returns an array of token ids.

### `decode(ids, provider?)`

Decodes an array of token ids back to a string.

### `isWithinTokenLimit(text, limit, provider?)`

Returns the token count if `text` is within `limit` tokens, or `false` if exceeded. More efficient than `encode()` for long texts — short-circuits as soon as the limit is crossed.

### Tokenizer instances

`openai()`, `openaiO200k()`, `anthropic()`, `gemini()` each return a cached `Tokenizer` object with `.encode()`, `.decode()`, and `.count()` methods. Instances are created once per provider per process.

## Why not tiktoken?

`tiktoken` is accurate for OpenAI but requires Rust/WASM native bindings, which can break in Docker containers, edge runtimes, and serverless environments. `bpe-lite` is pure JavaScript — it runs anywhere Node 18+ runs, with no native compilation step.

## License

MIT
