/**
 * Build script — downloads and processes vocab files for all three providers.
 * Run once at dev time: node scripts/build-vocabs.js
 * Output files are committed to the repo (no network needed at runtime).
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const VOCABS_DIR = path.join(__dirname, '..', 'vocabs');

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return res.text();
}

// ─── Shared tiktoken parser ───────────────────────────────────────────────────

function parseTiktokenFile(text) {
  const vocab = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const spaceIdx = trimmed.lastIndexOf(' ');
    if (spaceIdx === -1) continue;
    const token = trimmed.slice(0, spaceIdx);
    const rank = parseInt(trimmed.slice(spaceIdx + 1), 10);
    vocab[token] = rank;
  }
  return vocab;
}

// ─── OpenAI cl100k_base ───────────────────────────────────────────────────────

async function buildOpenAI() {
  console.log('Fetching OpenAI cl100k_base...');
  const url = 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken';
  const vocab = parseTiktokenFile(await fetchText(url));
  console.log(`  Parsed ${Object.keys(vocab).length} tokens`);

  // GPT-4 / cl100k_base pre-tokenization pattern
  const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  const data = {
    engine: 'tiktoken',
    pattern,
    vocab,
    specialTokens: {
      '<|endoftext|>': 100257,
      '<|fim_prefix|>': 100258,
      '<|fim_middle|>': 100259,
      '<|fim_suffix|>': 100260,
      '<|endofprompt|>': 100276,
    },
  };

  const outPath = path.join(VOCABS_DIR, 'openai.json.gz');
  fs.writeFileSync(outPath, zlib.gzipSync(JSON.stringify(data)));
  console.log(`  Written: vocabs/openai.json.gz (${(fs.statSync(outPath).size / 1024 / 1024).toFixed(1)} MB)`);
  return data;
}

// ─── OpenAI o200k_base (GPT-4o, o1, o3, o4, GPT-4.1, GPT-5) ─────────────────

async function buildOpenAIModern() {
  console.log('Fetching OpenAI o200k_base...');
  const url = 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken';
  const vocab = parseTiktokenFile(await fetchText(url));
  console.log(`  Parsed ${Object.keys(vocab).length} tokens`);

  // o200k_base pre-tokenization pattern (from tiktoken source)
  const pattern = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  const data = {
    engine: 'tiktoken',
    pattern,
    vocab,
    specialTokens: {
      '<|endoftext|>': 199999,
      '<|endofprompt|>': 200018,
    },
  };

  const outPath = path.join(VOCABS_DIR, 'openai-o200k.json.gz');
  fs.writeFileSync(outPath, zlib.gzipSync(JSON.stringify(data)));
  console.log(`  Written: vocabs/openai-o200k.json.gz (${(fs.statSync(outPath).size / 1024 / 1024).toFixed(1)} MB)`);
}

// ─── Anthropic (Xenova/claude-tokenizer — reverse-engineered ~65k vocab) ──────

// GPT-2 byte_to_unicode mapping (inverted): unicode char → byte value.
// Used to convert Xenova's GPT-2-encoded vocab strings to raw bytes.
function buildGPT2CharToByte() {
  const bs = [];
  const cs = [];
  for (let b = 0x21; b <= 0x7E; b++) { bs.push(b); cs.push(b); } // '!' – '~'
  for (let b = 0xA1; b <= 0xAC; b++) { bs.push(b); cs.push(b); } // '¡' – '¬'
  for (let b = 0xAE; b <= 0xFF; b++) { bs.push(b); cs.push(b); } // '®' – 'ÿ'
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const map = new Map();
  for (let i = 0; i < bs.length; i++) map.set(String.fromCodePoint(cs[i]), bs[i]);
  return map;
}

async function buildAnthropic() {
  console.log('Fetching Anthropic/Claude tokenizer from Xenova (HuggingFace)...');
  const url = 'https://huggingface.co/Xenova/claude-tokenizer/resolve/main/tokenizer.json';
  const hfTokenizer = JSON.parse(await fetchText(url));

  const { merges: hfMerges } = hfTokenizer.model;
  console.log(`  Merges: ${hfMerges.length}`);

  const charToByte = buildGPT2CharToByte();

  function gpt2StrToLatin1(str) {
    let out = '';
    for (const c of str) {
      const b = charToByte.get(c);
      if (b === undefined) throw new Error(`Unknown GPT-2 char: ${c} (U+${c.codePointAt(0).toString(16)})`);
      out += String.fromCharCode(b);
    }
    return out;
  }

  // Build vocab: base64(bytes) → rank.
  // Merged tokens get rank = merge index (lower = higher BPE priority).
  // Single-byte base tokens get high ranks — they're leaves, never merged results.
  const vocab = {};
  const BASE_RANK = hfMerges.length + 256;

  for (let byte = 0; byte < 256; byte++) {
    const b64 = Buffer.from([byte]).toString('base64');
    vocab[b64] = BASE_RANK + byte;
  }

  for (let i = 0; i < hfMerges.length; i++) {
    const sp = hfMerges[i].indexOf(' ');
    const merged = hfMerges[i].slice(0, sp) + hfMerges[i].slice(sp + 1);
    const bin = gpt2StrToLatin1(merged);
    const b64 = Buffer.from(bin, 'latin1').toString('base64');
    vocab[b64] = i;
  }

  // Remove over-merged sequences: Claude encodes these at byte level, but Xenova
  // has them as merged tokens. Removing them forces byte-level fallback.
  // For each char: delete all multi-byte prefix slices (len 2..N) from vocab,
  // so intermediate merges can't happen either.
  // NFKC is applied first, so ™→TM, …→..., etc. before BPE.
  const byteOnlyChars = [
    // Latin-1 supplement symbols (2-byte UTF-8): api=2, our=1 without fix
    // Removed: § (api=1 confirmed), £ (api=1), © (api=1 confirmed), × (api=1 confirmed)
    '\u00AE', // ®  (api=2)
    '\u00B1', // ±  (api=2)
    '\u00B6', // ¶  (api=2)
    '\u00F7', // ÷  (api=2)
    // Math / arrows / punctuation (3-byte UTF-8): api=4 (byte-level), our=1-3 without fix
    // Removed from list: † (injected below), • (api=1), → (api=1), − (api=1), √ (handled below)
    //                    ← (handled below via [86,90] injection)
    // ‡ handled separately below (can't delete [E2,80] prefix — † and • need it)
    '\u2191', // ↑  (api=4 — all bytes separate)
    '\u2193', // ↓  (api=4)
    '\u2194', // ↔  (api=4)
    '\u2248', // ≈  (api=4) — confirmed by probe
    '\u2260', // ≠  (api=4)
    '\u2264', // ≤  (api=4)
    '\u2265', // ≥  (api=4)
    '\u221E', // ∞  (api=4)
    '\u2211', // ∑  (api=4)
    '\u220F', // ∏  (api=4)
    '\u222B', // ∫  (api=4)
    // ™ → NFKC → "TM": delete "TM" merged token so it stays as 2 chars
    'TM',
    // Emoji (4-byte UTF-8): api=3-4, our=1-2 without fix
    '\u{1F600}', // 😀
    '\u{1F602}', // 😂
    '\u{1F914}', // 🤔
    '\u{1F389}', // 🎉
    '\u{1F680}', // 🚀
    '\u{1F30D}', // 🌍
    '\u{1F525}', // 🔥
    '\u{1F4A1}', // 💡
    '\u{2764}',  // ❤ (heavy heart, U+2764, 3-byte)
    '\u{1F40D}', // 🐍
    '\u{1F98A}', // 🦊
    '\u{1F308}', // 🌈
    '\u{2B50}',  // ⭐ (U+2B50, 3-byte)
    '\u{1F3B5}', // 🎵
    '\u{1F3C6}', // 🏆
    '\u{2615}',  // ☕ (U+2615, 3-byte)
    '\u{1F4BB}', // 💻
    '\u{1F44D}', // 👍 (api=3, confirmed by probe)
    '\u{1F60D}', // 😍 (api=3, confirmed by probe)
  ];

  for (const ch of byteOnlyChars) {
    const bytes = Buffer.from(ch, 'utf8');
    // Delete all contiguous sub-sequences of length 2..N to prevent any merge path,
    // including suffix merges (e.g. for € = [E2,82,AC], also delete [82,AC]).
    for (let start = 0; start < bytes.length; start++) {
      for (let len = 2; start + len <= bytes.length; len++) {
        delete vocab[bytes.slice(start, start + len).toString('base64')];
      }
    }
    // Also delete the exact space-prefixed sequence so " ©" etc. can't merge to 1 token.
    // We do NOT delete space+prefix intermediates (e.g. [0x20,0xC2]) to avoid breaking
    // non-blocklisted chars that share the same byte prefix (e.g. § shares 0xC2 with ©).
    const spacedBytes = Buffer.concat([Buffer.from([0x20]), bytes]);
    delete vocab[spacedBytes.toString('base64')];
  }

  // ‡ (U+2021 = [E2,80,A1]): block only the full sequence and its suffix [80,A1].
  // Do NOT delete the [E2,80] prefix — † (U+2020) and • (U+2022) share it and
  // Claude keeps those as 1-token merges (probe: api=1).
  {
    const b = Buffer.from('\u2021', 'utf8'); // [E2, 80, A1]
    delete vocab[b.toString('base64')];        // full: [E2,80,A1]
    delete vocab[b.slice(1).toString('base64')]; // suffix: [80,A1]
  }

  // Delete [20,E2] (space + E2 prefix byte) — rank 4676 in Xenova.
  // Context probes confirm Claude does NOT merge space+E2 for any arrow/math symbol:
  // " ↑" " ↓" " ↔" " ≠" etc. are all 4 separate bytes in Claude (api=4).
  // All Claude 1-token E2-range symbols (•, €, −, →) have their own suffix pair
  // paths that work without [20,E2] (e.g. [86,92] enables "→", [82,AC] enables "€").
  delete vocab[Buffer.from([0x20, 0xE2]).toString('base64')];

  // Delete [20,E2,80], [20,E2,88], [20,E2,89] — 3-byte space+prefix intermediates.
  // Prevents partial merges in the E2-80/88/89 Unicode sub-ranges.
  for (const hex of ['20E280', '20E288', '20E289']) {
    const key = Buffer.from(hex.match(/../g).map(h => parseInt(h, 16))).toString('base64');
    delete vocab[key];
  }

  // ← (U+2190 = [E2,86,90]): neither "←" nor [86,90] are in Xenova — inject both.
  // Path: [86,90] merges first, then [E2]+[86,90]→"←", then [20]+[←]=2 tokens.
  // Context probe: api=2. " ←"=[20,E2,86,90] is absent from Xenova → stays 2.
  vocab[Buffer.from([0x86, 0x90]).toString('base64')] = hfMerges.length + 200;
  vocab[Buffer.from([0xE2, 0x86, 0x90]).toString('base64')] = hfMerges.length + 201;
  delete vocab[Buffer.from([0x20, 0xE2, 0x86, 0x90]).toString('base64')]; // " ←" absent anyway

  // √ (U+221A = [E2,88,9A]): inject [88,9A] pair so the path [E2]+[88,9A]→"√"
  // works, giving " √" = 2 tokens [space]+[√]. Context probe: api=2.
  // "√"=[E2,88,9A] is already in Xenova at rank 54655; [88,9A] is not, inject it.
  // " √"=[20,E2,88,9A] is absent from Xenova → naturally gives 2 tokens.
  vocab[Buffer.from([0x88, 0x9A]).toString('base64')] = hfMerges.length + 202;

  // Delete Xenova's aggressive repeated-byte merges (length > 2).
  // Probe confirmed: Claude only keeps pairs (e.g. "aa"), not triples/quadruples.
  // Xenova has "aaa", "aaaa", "aaaaaaaa" etc. → hugely over-merges repeated chars.
  for (const b64 of Object.keys(vocab)) {
    const bytes = Buffer.from(b64, 'base64');
    if (bytes.length > 2 && bytes.every(b => b === bytes[0])) {
      delete vocab[b64];
    }
  }

  // Inject single-char tokens that Xenova never learned to merge but Claude has.
  // Arabic: 2-byte UTF-8 chars absent from Xenova's merge list → each produces 2
  // byte tokens instead of 1. Ranks assigned just below BASE_RANK so they merge
  // with lowest priority among merged tokens (appropriate for less-common chars).
  // Japanese/CJK: 3-byte chars whose 2-byte prefix IS in Xenova (verified) but
  // whose full 3-byte sequence is not — just need to add the full token.
  const injectChars = [
    // " †" (space + dagger, U+2020 = [E2,80,A0]): Xenova has "†" (rank 38612) and
    // [E2,80] (rank 216) but lacks the space-prefixed merge " †". Inject it so
    // the path [E2,80]+[A0]="†" → [20]+"†"→" †" completes. Context probe: api=1.
    ' \u2020', // " †" [20,E2,80,A0]
    // Arabic chars missing from Xenova (confirmed: Xenova has ا,ل,م,ن,ر,و,ي etc.
    // but not these — each drops to 2 raw byte tokens)
    '\u063A', // غ  ghain       D8 BA
    '\u0630', // ذ  dhaal       D8 B0
    '\u0621', // ء  hamza       D8 A1
    '\u0622', // آ  alef+madda  D8 A2
    // Japanese/CJK single chars missing from Xenova (2-byte prefix is present)
    '\u4E16', // 世  E4 B8 96
    '\u6A5F', // 機  E6 A9 9F
    '\u68B0', // 械  E6 A2 B0
    '\u7FD2', // 習  E7 BF 92
    '\u30E2', // モ  E3 83 A2
    '\u8A9E', // 語  E8 AA 9E
  ];
  // BASE_RANK = hfMerges.length + 256; single-byte tokens use BASE_RANK+0..255.
  // The range [hfMerges.length, hfMerges.length+255] is unused — perfect for injected chars.
  // They rank below single bytes (so they merge) but above no existing merged token.
  let injectRank = hfMerges.length; // first unused rank slot
  for (const ch of injectChars) {
    const b64 = Buffer.from(ch, 'utf8').toString('base64');
    if (!(b64 in vocab)) {
      vocab[b64] = injectRank++;
    }
  }

  // Remove Xenova multi-char CJK merges that Claude does NOT have.
  // Probed against Claude API: these pairs give api=2 (separate chars) but our
  // Xenova vocab merges them to 1 token, causing under-counting.
  const removeMultiCharCJK = [
    '正在', // U+6B63 U+5728 — zh "currently"  (probe: api≠our, confirmed over-merge)
    '学习', // U+5B66 U+4E60 — zh "study/learn" (appears 2x in zh sample)
    '模型', // U+6A21 U+578B — zh "model"
    '可以', // U+53EF U+4EE5 — zh "can/able to"
    // NOTE: "生成" (U+751F U+6210) KEPT — Claude has this merged in both zh and ja
  ];
  for (const ch of removeMultiCharCJK) {
    const b64 = Buffer.from(ch, 'utf8').toString('base64');
    delete vocab[b64];
  }

  // ByteLevel pre-tokenization pattern (GPT-2 / cl100k family)
  const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  const data = {
    engine: 'tiktoken',
    pattern,
    vocab,
    normalize: 'NFKC',
    specialTokens: {},
    note: 'Reverse-engineered Claude tokenizer via Xenova/claude-tokenizer (HuggingFace). ~65k vocab.',
  };

  const outPath = path.join(VOCABS_DIR, 'anthropic.json.gz');
  fs.writeFileSync(outPath, zlib.gzipSync(JSON.stringify(data)));
  console.log(`  Written: vocabs/anthropic.json.gz (${(fs.statSync(outPath).size / 1024 / 1024).toFixed(1)} MB)`);
}

// ─── Gemini (Gemma 3 tokenizer) ───────────────────────────────────────────────

async function buildGemini() {
  console.log('Fetching Gemini/Gemma3 tokenizer from HuggingFace...');
  const url = 'https://huggingface.co/unsloth/gemma-3-1b-it/resolve/main/tokenizer.json';
  const text = await fetchText(url);
  const hfTokenizer = JSON.parse(text);

  const model = hfTokenizer.model;
  if (!model || model.type !== 'BPE') {
    throw new Error(`Unexpected model type: ${model && model.type}`);
  }

  const vocab = model.vocab;           // { token_string: id, ... }
  // Merges may be strings "a b" or arrays ["a", "b"] depending on HF tokenizers version
  const merges = model.merges.map(m => Array.isArray(m) ? `${m[0]} ${m[1]}` : m);

  const count = Object.keys(vocab).length;
  console.log(`  Vocab size: ${count} tokens, ${merges.length} merges`);

  const data = {
    engine: 'spm',
    vocab,
    merges,
  };

  const outPath = path.join(VOCABS_DIR, 'gemini.json.gz');
  fs.writeFileSync(outPath, zlib.gzipSync(JSON.stringify(data)));
  console.log(`  Written: vocabs/gemini.json.gz (${(fs.statSync(outPath).size / 1024 / 1024).toFixed(1)} MB)`);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  fs.mkdirSync(VOCABS_DIR, { recursive: true });
  console.log('Building vocab files...\n');

  const openaiData = await buildOpenAI();
  console.log();
  await buildAnthropic();
  console.log();
  await buildOpenAIModern();
  console.log();
  await buildGemini();
  console.log('\nDone. All vocab files built.');
}

main().catch(err => {
  console.error('ERROR:', err.message);
  process.exit(1);
});
