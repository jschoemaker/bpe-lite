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

| Provider | Vocab | Tokens | Models | Accuracy |
|----------|-------|--------|--------|----------|
| `openai-o200k` | o200k_base | 199,998 | GPT-4o, o1, o3, o4, GPT-4.1, GPT-5 | Exact — vocab sourced directly from OpenAI's CDN |
| `openai` | cl100k_base | 100,256 | GPT-4, GPT-3.5 | Exact — vocab sourced directly from OpenAI's CDN |
| `anthropic` | Xenova/claude-tokenizer | 65,000 | Claude | ~68% within 10%, mean error 9.3% — Anthropic has not released the tokenizer. Vocab sourced from Xenova's reverse-engineered 65k BPE (HuggingFace) with NFKC normalization, symbol/emoji byte-level corrections, and probe-based merge adjustments. Typical prose and code: ~95% within 10%. Outliers: repeated chars (39%), Arabic (30%), Japanese (20%), currency (16%), symbols (15%). Tested against Claude 4 API across 25 diverse samples. |
| `gemini` | Gemma 3 | 262,144 | Gemini | ~99.7% — 92% exact, mean error 0.34% across 25 diverse samples vs Gemini 2.0 Flash API |

Vocab files are bundled in the package — no network required at runtime or install time.

## Performance

Benchmarked on Node v24 (win32/x64). Benchmark command: `node --expose-gc scripts/bench.js`.

**OpenAI cl100k — large text (~54 KB)**

| impl | ops/s | tokens/s | MB/s |
|------|------:|---------:|-----:|
| bpe-lite | **317** | **3.89M** | **16.8** |
| ai-tokenizer | 264 | 3.23M | 14.0 |
| js-tiktoken | 31 | 376k | 1.6 |

**Anthropic — large text (~54 KB)**

| impl | ops/s | tokens/s | MB/s |
|------|------:|---------:|-----:|
| bpe-lite | **346** | **4.23M** | **18.3** |
| ai-tokenizer | 309 | 5.64M | 16.4 |

**Gemini — large text (~54 KB)**

| impl | ops/s | tokens/s | MB/s | note |
|------|------:|---------:|-----:|------|
| bpe-lite | **1,090** | **12.2M** | **57.9** | actual Gemma3 SPM |
| ai-tokenizer | 240 | 2.68M | 12.7 | o200k BPE — different algorithm, different results |

ai-tokenizer does not implement Gemini tokenization. The row above uses their o200k encoding on the same input string; it produces different token ids and counts than the Gemini tokenizer, so it is not a real comparison.

Numbers vary by machine — run the bench script locally for results on your hardware.

## API

### `countTokens(text, provider?)`

Returns the number of tokens in `text`. Default provider: `'openai'`.

### `encode(text, provider?)`

Returns an array of token ids.

### `decode(ids, provider?)`

Decodes an array of token ids back to a string.

### `isWithinTokenLimit(text, limit, provider?)`

Returns the token count if `text` is within `limit` tokens, or `false` if exceeded. More efficient than `encode()` for long texts — the tiktoken engine short-circuits as soon as the limit is crossed.

### Tokenizer instances

`openai()`, `openaiO200k()`, `anthropic()`, `gemini()` each return a cached `Tokenizer` object with `.encode()`, `.decode()`, and `.count()` methods. Instances are created once per provider per process.

## Why not tiktoken?

`tiktoken` is accurate for OpenAI but requires Rust/WASM native bindings, which can break in Docker containers, edge runtimes, and serverless environments. `bpe-lite` is pure JavaScript — it runs anywhere Node 18+ runs, with no native compilation step.

## Caveats

- **Anthropic**: Anthropic has not released the Claude tokenizer. The vocab is sourced from [Xenova/claude-tokenizer](https://huggingface.co/Xenova/claude-tokenizer), a community reverse-engineering of the ~65k BPE vocab. NFKC normalization and probe-based merge corrections are applied. Accuracy varies by text type — common prose and code are usually within 10%, but Arabic, Japanese, repeated characters, and some symbol/emoji combinations diverge significantly.
- **Node version**: Requires Node 18+ for Unicode property escapes (`\p{L}`, `\p{N}`) in the pre-tokenization regex.

## License

MIT
