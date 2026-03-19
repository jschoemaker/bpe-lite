# bpe-lite accuracy benchmark — report

**Date:** 2026-03-19
**Model tested against:** `claude-haiku-4-5-20251001` via Anthropic `count_tokens` API
**Tokenizers compared:** bpe-lite (modified Xenova), ai-tokenizer (claude encoding), raw Xenova (unmodified)

---

## 1. Background

bpe-lite is a zero-dependency JS tokenizer supporting OpenAI (cl100k / o200k), Anthropic (Xenova/claude-tokenizer, 65k BPE), and Gemini (Gemma3 SPM). Anthropic has not released the Claude 4 tokenizer, so the Anthropic provider is a reverse-engineered approximation sourced from `Xenova/claude-tokenizer` on HuggingFace, with hand-tuned modifications.

This report documents the construction of a stratified accuracy benchmark and its results.

---

## 2. Benchmark corpus

### Design

120 samples across 12 categories (10 per category):

| Category | Focus |
|---|---|
| `english-prose` | sentences, paragraphs, mixed punctuation, dialogue |
| `code-python` | functions, classes, decorators, f-strings, async |
| `code-js` | arrow functions, classes, JSX, TypeScript, async/await |
| `numbers` | integers, floats, scientific notation, dates, IPs |
| `hex-binary` | 0x/0b prefixes, color codes, hashes, UUIDs, hex dumps |
| `symbols` | copyright/trademark, math operators, arrows, currency clusters |
| `arabic` | words, sentences, mixed Latin, technical text |
| `cjk` | Chinese, Japanese, Korean, mixed scripts |
| `emoji` | isolated, in prose, clusters, skin tones, flags, ZWJ sequences |
| `structured` | JSON, HTML, XML, Markdown, CSV, YAML, SQL, GraphQL |
| `urls` | full URLs, query strings, email addresses, data URIs |
| `short` | 1–5 token inputs, single words, punctuation only |

### Files

- `scripts/corpus.js` — 120 sample definitions (category, name, text)
- `scripts/fetch-corpus.js` — fetches expected counts from the Anthropic API, writes `scripts/corpus-expected.json` (committed; benchmark runs offline)
- `scripts/accuracy.js` — offline runner; reads corpus-expected.json, compares both tokenizers, outputs per-sample table and per-category summary with Wilson 95% CI

---

## 3. Calibration and a discovered overhead artifact

Expected counts are computed as `api_raw(text) - overhead`, where `overhead = api("Hi") - countTokens("Hi") = 7`.

A **±1 overhead artifact** was discovered: the last structural token of the Anthropic message template BPE-merges with certain first characters of content, making the effective overhead 7 or 8 depending on the first character:

```
"Hi"   raw=8  overhead=7   letter start  — our calibration anchor
"1"    raw=9  overhead=8   digit start
"©"    raw=9  overhead=8   2-byte UTF-8, C2xx range
"→"    raw=8  overhead=7   3-byte UTF-8, E2/86 range
"Hi1"  raw=9  net=2        Hi=1 + 1=1 — digit contributes 1 token in context ✓
"1Hi"  raw=10 net=3        boundary effect inflates count by 1
```

The artifact only matters for `expected < 5` tokens — at that scale ±1 is more than 20% relative error. For longer samples it is negligible.

**Resolution:** 6 samples with `expected < 5` are excluded from percentage error calculations and shown as `n/a`. All other samples are unaffected.

We also investigated and ruled out a "prefix neutralisation" approach (`api("Hi. " + text) - api("Hi. ")`): while it eliminates the digit-boundary artifact, the trailing space in the prefix gets absorbed into the first chunk of text (the BPE regex treats it as a leading space), corrupting token counts for short-string samples by a different ±1. The overhead subtraction approach with exclusion of tiny samples is the most honest solution.

---

## 4. What ai-tokenizer uses for Claude

ai-tokenizer's `claude` encoding is a **different vocabulary** from Xenova/claude-tokenizer:

- 64,995 tokens total (64,241 string-keyed + 754 binary)
- Special tokens: `EOT`, `META`, `META_START`, `META_END`, `SOS` — characteristic of an older Claude 1/2-era tokenizer
- Regex pattern uses `\p{N}+` (greedy, unlimited digits) instead of `\p{N}{1,3}` (1–3 digits)

The `\p{N}+` pattern is ai-tokenizer's primary weakness: it chunks multi-digit numbers as a single unit, whereas Claude uses 1–3 digit chunks. This causes severe errors on anything involving numbers (43% error on fibonacci integers, 29% on arithmetic, 22% on hex).

ai-tokenizer also does **not have a Gemini encoding** — all Gemini models in their registry are mapped to `o200k_base` (OpenAI's vocabulary) with a fudge multiplier of 1.08. This produces completely wrong results for Gemini.

---

## 5. Results — full 120-sample benchmark

### Overall summary (114 eligible samples, 6 excluded as expected < 5)

| Metric | bpe-lite | ai-tokenizer |
|---|---|---|
| Exact | 11 (9.6%) | 9 (7.9%) |
| Within 5% | 53 (46.5%) | 21 (18.4%) |
| Within 10% | 71 (62.3%) ±8.8% CI | 43 (37.7%) ±8.8% CI |
| Mean abs err | 9.4% | 16.0% |
| Median abs err | 5.7% | 13.6% |
| p95 abs err | 31.0% | 38.1% |
| Max abs err | 42.9% (single emoji repeated) | 82.6% (repeated chars) |

### Per-category breakdown (within-10% rate, mean abs err)

| Category | bpe-lite within-10% | bpe-lite mean err | ai-tok within-10% | ai-tok mean err |
|---|---|---|---|---|
| `english-prose` | 90% ±19% | 5.5% | 80% ±23% | 7.1% |
| `code-python` | 90% ±19% | 4.8% | 20% ±23% | 11.6% |
| `code-js` | 100% ±14% | 4.2% | 60% ±26% | 9.7% |
| `numbers` | 80% ±23% | 7.3% | 10% ±19% | 23.7% |
| `hex-binary` | 80% ±23% | 5.3% | 20% ±23% | 22.8% |
| `symbols` | 10% ±19% | 17.6% | 10% ±19% | 23.2% |
| `arabic` | 0% ±14% | 26.1% | 0% ±14% | 28.8% |
| `cjk` | 40% ±26% | 8.8% | 30% ±25% | 12.7% |
| `emoji` | 20% ±23% | 17.7% | 30% ±25% | 15.8% |
| `structured` | 90% ±19% | 3.6% | 70% ±25% | 9.5% |
| `urls` | 80% ±23% | 3.6% | 90% ±19% | 4.0% |
| `short` | 30% ±25% | 6.8% | 10% ±19% | 32.1% |

---

## 6. Comparison with raw Xenova (unmodified)

We also ran raw Xenova (no modifications applied) against the same API to isolate the effect of bpe-lite's hand-tuning:

| Metric | raw Xenova | bpe-lite |
|---|---|---|
| Within 10% | 68% | 84% (25-sample run) |
| Mean abs err | 12.48% | 5.74% (25-sample run) |
| Max abs err | 82.6% (repeated chars) | 21.7% (symbols) |

Key modifications that drive the improvement:

- **Repeated-byte merges deleted** — Xenova has `aaa`, `aaaa` etc.; Claude does not. Fixes `repeated chars` from 82.6% to 4.3%.
- **Emoji byte-pair injections** — Xenova merges full 4-byte emoji to 1 token; Claude uses 3–4 tokens. Injecting `[9F,91]`, `[9F,92]`, `[9F,98]` and `[20,F0]` pairs; deleting full emoji merges. Cuts emoji error from 26% to 8%.
- **Symbol path engineering** — Deleted over-merged 3-byte tokens (`↑↓↔≈≠≤≥∞∑∫`); injected `[E2,88]` and `[E2,82]` prefix pairs for correct 2-token bare paths. Reduces symbol error from 37.7% to 21.7%.
- **CJK/Japanese injections** — Added missing single-char tokens (`世 機 械 習 モ 語`). Drops Japanese error from 20% to 3%.
- **Whitespace sequence injections** — space×3..32, tab×2..8, nl×2..8 at rank 0. Fixes whitespace-heavy inputs.
- **Space+symbol merge deletions** — Xenova has ` £`, ` ±`, ` ≤`, ` ≥` merged; Claude does not. Deleted these.
- **NFKC normalisation** — Applied before BPE (`normalize: 'NFKC'`). Fixes `™→TM`, `…→...`, etc.

---

## 7. Known unresolvable issues

These categories cannot be fully fixed without the actual Claude tokenizer:

**Arabic (mean err 26%):** Xenova was trained on far less Arabic data than Claude. It has fewer Arabic merges, producing longer token sequences. Every Arabic sample is over-tokenized by 17–46 tokens. The gap grows with text length.

**Symbols (mean err 18%):** Claude tokenizes symbols using byte-level BPE without regex pre-tokenisation. Adjacent symbols can form cross-symbol byte merges (e.g. the last byte of `©` and the first byte of `®` may merge). Our regex-chunked approach processes each symbol in isolation, making these cross-boundary merges unreplicable. Some symbols also have different space-prefixed merge behaviour than Xenova.

**Emoji (mean err 18%):** Complex emoji sequences (ZWJ families, skin-tone variants, keycap sequences, symbol-like emoji) have irregular token counts that don't follow a simple pattern. bpe-lite handles the common cases but ZWJ sequences, flag emoji, and symbol-like emoji have 14–43% errors.

**Large integers (numbers, 33% err):** `1000000 9999999 ...` — these contain 7–12 digit numbers. The `\p{N}{1,3}` pattern chunks them into 1–3 digit groups as expected. However Claude appears to have merged some specific digit sequences differently. The current sample shows bpe-lite over-counting by 12 tokens (48 vs 36).

---

## 8. Comparison notes vs ai-tokenizer's published accuracy

ai-tokenizer's README claims 97–99% accuracy for Claude models at 5k–50k tokens, measured on random text. Our benchmark shows 37.7% within 10% on our 120-sample corpus. The discrepancy has two explanations:

1. **Test corpus composition:** ai-tokenizer tests on long random text (5k–50k tokens). At that scale, errors average out and the overall percentage is dominated by the majority of tokens which tokenize correctly. Our corpus deliberately over-represents hard categories (symbols, Arabic, emoji, numbers) that expose systematic failures.

2. **Number pattern flaw:** ai-tokenizer's `\p{N}+` regex is correct for the older Claude 1/2 tokenizer they appear to have encoded, but wrong for current Claude models which use `\p{N}{1,3}`. On random prose this matters little; on code and data it causes large errors.

For the specific use case of estimating token counts on real-world diverse inputs, bpe-lite's mean error of 9.4% (with a 62% within-10% rate) is substantially more reliable than ai-tokenizer's 16% mean error and 37.7% within-10% rate.

---

## 9. Benchmark scripts summary

```
node scripts/fetch-corpus.js     # one-time: fetch 120 expected counts from API
node scripts/accuracy.js         # offline: compare bpe-lite + ai-tokenizer vs corpus
```

The corpus-expected.json is committed and does not need to be re-fetched unless the corpus changes or a new model is tested.
