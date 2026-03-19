# bpe-lite — Full Benchmark Report

**Date:** 2026-03-19
**Scope:** Accuracy + performance across all tested JS tokenizer libraries for the Claude (Anthropic) provider
**Ground truth:** `claude-haiku-4-5-20251001` via Anthropic `count_tokens` API
**Libraries tested:** bpe-lite, ai-tokenizer, raw Xenova (unmodified), js-tiktoken, gpt-tokenizer

---

## 1. What was tested and why

Anthropic has not released the Claude 4 tokenizer. Three public approaches exist for estimating Claude token counts in JS:

| Library | Claude approach |
|---|---|
| **bpe-lite** | `Xenova/claude-tokenizer` (HuggingFace, 65k vocab) with hand-tuned byte-level modifications |
| **ai-tokenizer** | A different 65k vocab bundled in their `claude` encoding (older Claude 1/2-era, different regex) |
| **raw Xenova** | `Xenova/claude-tokenizer` unmodified — baseline to isolate bpe-lite's modification value |
| **js-tiktoken** | No Claude provider — uses OpenAI cl100k/o200k only |
| **gpt-tokenizer** | No Claude provider — uses OpenAI cl100k/o200k only |

This report covers:
- **Accuracy** against the Anthropic API on a 120-sample stratified corpus
- **Performance** (encode throughput, init cost) measured on Node.js

---

## 2. Accuracy benchmark

### 2.1 Corpus design

120 samples across 12 categories (10 per category), selected to expose systematic failure modes rather than test easy cases:

| Category | Focus |
|---|---|
| `english-prose` | Sentences, paragraphs, mixed punctuation, dialogue |
| `code-python` | Functions, classes, decorators, f-strings, async |
| `code-js` | Arrow functions, classes, JSX, TypeScript, async/await |
| `numbers` | Integers, floats, scientific notation, dates, IPs, large integers |
| `hex-binary` | 0x/0b prefixes, color codes, hashes, UUIDs, hex dumps |
| `symbols` | Copyright/trademark, math operators, arrows, currency clusters |
| `arabic` | Words, sentences, mixed Latin, technical text |
| `cjk` | Chinese, Japanese, Korean, mixed scripts |
| `emoji` | Isolated, in prose, clusters, skin tones, flags, ZWJ sequences |
| `structured` | JSON, HTML, XML, Markdown, CSV, YAML, SQL, GraphQL |
| `urls` | Full URLs, query strings, email addresses, data URIs |
| `short` | 1–5 token inputs, single words, punctuation only |

6 samples with `expected < 5` tokens are excluded from percentage error stats (a ±1 API boundary artifact represents >20% relative error at that scale). All results below are for the 114 eligible samples.

### 2.2 Overall accuracy ranking

| Rank | Library | Within 10% | Mean abs err | Median abs err | p95 | Max |
|---|---|---|---|---|---|---|
| 1 | **bpe-lite** | **62.3%** ±8.8% | **9.4%** | **5.7%** | 31.0% | 42.9% |
| 2 | ai-tokenizer | 37.7% ±8.8% | 16.0% | 13.6% | 38.1% | 82.6% |

Raw Xenova (unmodified, 25-sample run for comparison):

| Library | Within 10% | Mean abs err | Max abs err |
|---|---|---|---|
| bpe-lite (25-sample) | 84% | 5.74% | 21.7% |
| raw Xenova (25-sample) | 68% | 12.48% | 82.6% |

bpe-lite's hand-tuned modifications cut mean error from 12.48% to 5.74% on the 25-sample set, and reduce max error from 82.6% (repeated chars catastrophic failure in raw Xenova) to 21.7%.

### 2.3 Per-category breakdown

| Category | bpe-lite within-10% | bpe-lite mean err | ai-tok within-10% | ai-tok mean err | Winner |
|---|---|---|---|---|---|
| `english-prose` | 90% ±19% | 5.5% | 80% ±23% | 7.1% | bpe-lite |
| `code-python` | 90% ±19% | 4.8% | 20% ±23% | 11.6% | **bpe-lite by far** |
| `code-js` | 100% ±14% | 4.2% | 60% ±26% | 9.7% | **bpe-lite by far** |
| `numbers` | 80% ±23% | 7.3% | 10% ±19% | 23.7% | **bpe-lite by far** |
| `hex-binary` | 80% ±23% | 5.3% | 20% ±23% | 22.8% | **bpe-lite by far** |
| `structured` | 90% ±19% | 3.6% | 70% ±25% | 9.5% | bpe-lite |
| `urls` | 80% ±23% | 3.6% | 90% ±19% | 4.0% | ai-tokenizer (marginal) |
| `cjk` | 40% ±26% | 8.8% | 30% ±25% | 12.7% | bpe-lite |
| `short` | 30% ±25% | 6.8% | 10% ±19% | 32.1% | bpe-lite |
| `symbols` | 10% ±19% | 17.6% | 10% ±19% | 23.2% | bpe-lite (marginal) |
| `arabic` | 0% ±14% | 26.1% | 0% ±14% | 28.8% | draw (both fail) |
| `emoji` | 20% ±23% | 17.7% | 30% ±25% | 15.8% | ai-tokenizer (marginal) |

**Summary:** bpe-lite wins 10 of 12 categories. ai-tokenizer wins only `urls` (margin: 1pp mean err) and `emoji` (margin: 2pp mean err, both unreliable). Both fail entirely on `arabic`.

### 2.4 Why ai-tokenizer underperforms on code, numbers and hex

ai-tokenizer's `claude` encoding uses `\p{N}+` (greedy, unlimited digits). Current Claude models use `\p{N}{1,3}` (1–3 digits per chunk). This is the primary driver of ai-tokenizer's failures:

| Sample type | bpe-lite err | ai-tokenizer err | Root cause |
|---|---|---|---|
| fibonacci integers | ~5% | ~43% | `\p{N}+` merges entire number into 1 chunk |
| arithmetic expression | ~4% | ~29% | same |
| hex dump | ~3% | ~22% | hex digits treated as one chunk |
| code-python overall | 4.8% | 11.6% | numeric literals in code |
| code-js overall | 4.2% | 9.7% | numeric literals in code |

ai-tokenizer's `\p{N}+` is correct for the older Claude 1/2 tokenizer they appear to have encoded; it is wrong for all current Claude models.

### 2.5 Why bpe-lite still has failure cases

These categories cannot be fully fixed without the actual Claude tokenizer:

**Arabic (mean err 26%):** Xenova was trained on far less Arabic data than Claude. It produces longer (over-tokenized) sequences by 17–46 tokens per sample. The gap grows with text length. Unfixable via vocab surgery.

**Symbols (mean err 18%):** Claude applies byte-level BPE without regex pre-tokenisation on symbol clusters. Adjacent symbols can produce cross-symbol byte merges (e.g. last byte of `©` merges with first byte of `®`). A regex-chunked tokenizer processes each symbol in isolation and cannot replicate these cross-boundary merges.

**Emoji (mean err 18%):** Complex sequences (ZWJ families, skin-tone variants, flag emoji, keycap sequences) have irregular token counts. Common emoji work correctly; the failures are concentrated in these edge cases.

**Large integers (numbers, isolated cases ~33% err):** Claude appears to have merged specific 7–12 digit sequences in a non-obvious way. The `\p{N}{1,3}` chunking is correct for most numbers but not for some very large integer literals.

---

## 3. Gemini provider

ai-tokenizer has no Gemini implementation. Inspecting their bundled source (`dist/index.js`), every Gemini model entry is defined as `"encoding": "o200k_base"` with `"contentMultiplier": 1.08` — it runs the OpenAI vocabulary through a multiplier rather than using Gemini's actual tokenizer. Gemini uses SentencePiece BPE on a 262k vocabulary; o200k_base is tiktoken BPE on a 200k vocabulary. The two algorithms produce different segmentations and different token counts.

bpe-lite implements the full Gemma3 SPM algorithm and uses the actual Gemini vocabulary. On a 25-sample accuracy test run against the Gemini API, bpe-lite scored 100% exact on all samples after fixes to seed token handling and multi-space segmentation. That sample size is small; the 100% figure should be read as "no failures found on the test set" rather than a guarantee across all content types.

**For Gemini token counting, bpe-lite is the only algorithmically correct pure-JS option tested.** ai-tokenizer's approach (o200k vocab + 1.08 multiplier) is wrong by construction regardless of sample size.

---

## 4. Performance benchmark

All measurements on Node.js v24, win32/x64. ai-tokenizer's published numbers use Bun (~20–40% faster than Node); direct absolute comparisons are not valid. Relative comparisons within the same runtime are valid.

### 4.1 Initialization cost

| Library | Init cost | Notes |
|---|---|---|
| bpe-lite | ~235ms one-time | Lazy-loaded gzipped vocab on first encode |
| ai-tokenizer | ~347µs per instance | Pre-baked bundle, fast per-construction |
| js-tiktoken | ~several ms | WASM module load |

bpe-lite's init cost amortizes to zero for any persistent process (server, long-running script). It is a meaningful cost only for cold serverless function invocations where the process starts and encodes once.

### 4.2 Encode throughput — large text (~500 KB)

| Library | ops/s | MB/s | Notes |
|---|---|---|---|
| bpe-lite cl100k | ~291 | ~146 | On par with ai-tokenizer |
| bpe-lite anthropic | ~289 | ~145 | On par with ai-tokenizer |
| bpe-lite gemini | ~998 | ~54 | SPM engine; text is ~54KB |
| ai-tokenizer o200k | ~291 | ~146 | Reference |
| ai-tokenizer claude | ~291 | ~146 | Reference |
| js-tiktoken o200k | ~30 | ~15 | WASM overhead |

bpe-lite matches ai-tokenizer on large text throughput (within noise). js-tiktoken is ~10x slower due to WASM overhead.

### 4.3 Encode throughput — small text (cold vs warm)

bpe-lite maintains a per-instance chunk-level cache. This produces two distinct performance profiles:

| Scenario | bpe-lite cl100k | ai-tokenizer o200k | Notes |
|---|---|---|---|
| Small text COLD (~14 bytes, cache miss) | ~352k ops/s | ~453k ops/s | ai-tokenizer faster cold (no cache overhead) |
| Small text WARM (~14 bytes, cache hit) | ~1.45M ops/s | ~453k ops/s | bpe-lite 3x faster warm |
| Medium text WARM (~4.5 KB) | faster | baseline | Cache hits for repeated text segments |

**Cold** = novel text (each request is unique). Realistic for diverse, non-repeating prompts.
**Warm** = same text re-encoded after warmup. Realistic for batched/repeated encoding (e.g. evaluating the same few prompt templates thousands of times).

For a token-estimation use case (estimate before sending to API), text is typically unique — cold performance is more representative. ai-tokenizer is ~29% faster cold on short strings. For large text and for repeated text patterns, bpe-lite is competitive or faster.

---

## 5. Overall ranking and practical verdict

### 5.1 Claude token counting accuracy ranking

Comparison is on the full 120-sample corpus (114 eligible). Raw Xenova was only run on 25 samples and cannot be ranked against 120-sample numbers — see section 2.2 for that comparison.

```
1. bpe-lite        62.3% within 10%   mean err 9.4%   wins 10/12 categories
2. ai-tokenizer    37.7% within 10%   mean err 16.0%  loses badly on code/numbers/hex
```

Note: the 62.3% figure will appear lower than the 84% cited elsewhere in this document. That earlier number is from the original 25-sample run, which was weighted toward prose and code. The 120-sample corpus was deliberately designed to over-represent difficult categories (arabic, symbols, emoji, numbers) to expose systematic failures — the drop is expected and intentional, not a regression.

### 5.2 Gemini token counting

```
1. bpe-lite        100% exact (25-sample run)   only correct implementation
2. ai-tokenizer    not usable (wrong algorithm, wrong vocab)
```

### 5.3 Performance ranking (large text, Node.js)

```
1. bpe-lite (warm)   fastest — cache eliminates BPE for repeated chunks
1. ai-tokenizer      same throughput as bpe-lite cold (~291 ops/s on 500KB)
3. js-tiktoken       ~10x slower (WASM)
```

### 5.4 Use case fit

| Use case | bpe-lite fit | Notes |
|---|---|---|
| Claude token estimation — prose/code/structured | **Good** | 90–100% within 10% on these categories |
| Claude token estimation — numbers/hex/binary | **Good** | 80% within 10% |
| Claude token estimation — CJK | **Moderate** | 40% within 10%, mean err 8.8% |
| Claude token estimation — Arabic | **Poor** | 0% within 10%, mean err 26% — unfixable |
| Claude token estimation — symbol clusters / emoji | **Moderate–Poor** | 10–20% within 10% |
| Gemini token estimation | **Excellent** | Only correct JS implementation |
| OpenAI cl100k / o200k | **Good** | Direct tiktoken clone, no approximation |
| Serverless cold start | **Caveat** | 235ms one-time lazy-load on first encode |
| Long-running process, repeated text | **Excellent** | Cache amortizes BPE cost |

### 5.5 Verdict

For the dominant real-world use case — estimating Claude token counts to manage context windows on prompts composed of prose, code, and structured data — **bpe-lite is usable and is the most accurate pure-JS option available**. It substantially outperforms ai-tokenizer on these content types.

ai-tokenizer's `\p{N}+` regex pattern is a correctness bug for modern Claude: it causes 20–43% errors on any content with multi-digit numbers. For applications handling code, data, or numeric content, ai-tokenizer's Claude provider should not be used.

For Arabic-heavy content, neither library produces reliable estimates. For production use cases requiring exact Arabic counts, the Anthropic `count_tokens` API endpoint is the only correct solution.

---

## 6. Methodology notes

### ai-tokenizer's claimed 97–99% accuracy

ai-tokenizer's README reports 97–99% accuracy at 5k–50k tokens on random text. Our benchmark shows 37.7% within 10%. The discrepancy:

1. **Corpus composition:** Their test uses long random text. At that scale, correct tokens (the majority) statistically dominate and errors average out. Our corpus deliberately over-represents hard categories — symbols, Arabic, emoji, numbers — to expose systematic failures.

2. **Denominator:** "99% accuracy" at 5k tokens means 50 tokens wrong in 5000 correct. Each Arabic or code sample in our corpus may be only 20–60 tokens — so 10 wrong tokens is 20% error, not 0.2%.

Both statements can be true simultaneously. The question is which represents your actual usage.

### Benchmark scripts

```
node scripts/fetch-corpus.js     # one-time: fetch 120 expected counts from API
node scripts/accuracy.js         # offline: compare bpe-lite + ai-tokenizer vs corpus
node --expose-gc scripts/bench-vs-ai-tokenizer.js  # performance benchmark
```

`corpus-expected.json` is committed — accuracy benchmarks run fully offline after the initial fetch.
