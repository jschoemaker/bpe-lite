'use strict';

/**
 * SentencePiece BPE encoder for Gemini/Gemma3.
 * Vocab keys are token strings (▁ for space), values are token ids.
 * Merges are ordered list of "token_a token_b" strings — applied by rank (index = priority).
 */

const { MinHeap } = require('./bpe');

const SPACE_CHAR = '\u2581'; // ▁

function buildPreparedSPM(vocabData) {
  const { vocab, merges } = vocabData;

  const mergeRank = new Map();
  for (let i = 0; i < merges.length; i++) {
    mergeRank.set(merges[i], i);
  }

  const idToStr = new Map();
  for (const [str, id] of Object.entries(vocab)) {
    idToStr.set(id, str);
  }

  // Seed tokens: multi-char vocab entries with no producing merge.
  // In the original SentencePiece model these were user-defined symbols
  // (HTML tags, special tokens, etc.) that the encoder recognizes atomically
  // before BPE. We handle the common '<' case with a greedy longest-match
  // lookup keyed by '<' — the only first-char with multiple seed tokens.
  const producible = new Set();
  for (const m of merges) {
    const sp = m.indexOf(' ');
    if (sp !== -1) producible.add(m.slice(0, sp) + m.slice(sp + 1));
  }
  const seedsByAngleBracket = [];
  for (const [str, id] of Object.entries(vocab)) {
    if (str.length > 1 && str[0] === '<' && !producible.has(str)) {
      seedsByAngleBracket.push({ str, id, chars: [...str] });
    }
  }
  seedsByAngleBracket.sort((a, b) => b.chars.length - a.chars.length);

  return {
    vocab,
    merges,
    mergeRank,
    idToStr,
    seedsByAngleBracket,
    // opt A — segment-level cache: each word segment → ids[]
    // Generalises across different inputs (same words reused across texts).
    // Note: 1 of 514,906 Gemma merges crosses a ▁ boundary ("> ▁</"),
    // making this negligibly imprecise for that HTML pattern.
    cache: new Map(),
    // opt B — per-instance grow-only scratch
    scratch: { str: null, ids: null, prev: null, next: null, ver: null, alive: null, cap: 0, heap: new MinHeap() },
  };
}

// opt B — grow SPM scratch arrays only when needed
function ensureScratch(scratch, n) {
  if (n <= scratch.cap) return;
  const cap = n * 2;
  scratch.str   = new Array(cap);
  scratch.ids   = new Int32Array(cap);
  scratch.prev  = new Int32Array(cap);
  scratch.next  = new Int32Array(cap);
  scratch.ver   = new Int32Array(cap);
  scratch.alive = new Uint8Array(cap);
  scratch.cap   = cap;
}

/**
 * Encode text using SentencePiece BPE.
 * @param {string} text
 * @param {Object} vocabData  — { engine, vocab, merges }
 * @returns {number[]}
 */
function encodeSPM(text, vocabData) {
  return encodeSPMPrepared(text, buildPreparedSPM(vocabData));
}

/**
 * Hot path: scan normalized text purely from cache.
 * Returns true if every segment was a cache hit; false on first miss.
 * Isolated from encodeSegment so V8 can keep this function optimised
 * even when encodeSPMPrepared is called with wildly different text lengths.
 *
 * Segmentation: each run of ▁ chars plus the following non-▁ word is one segment.
 * Leading ▁s are included so BPE can naturally merge ▁▁→138, ▁▁▁→139, etc.
 */
function _scanFromCache(normalized, cache, result) {
  let i = 0;
  while (i < normalized.length) {
    // Count leading ▁ chars
    const segStart = i;
    while (i < normalized.length && normalized[i] === SPACE_CHAR) i++;
    const spaceCount = i - segStart;

    if (i === normalized.length) {
      // Trailing spaces only
      if (spaceCount > 0) {
        const seg = normalized.slice(segStart, i);
        const segIds = cache.get(seg);
        if (segIds === undefined) return false;
        for (let j = 0; j < segIds.length; j++) result.push(segIds[j]);
      }
      break;
    }

    // Collect word chars (non-▁)
    while (i < normalized.length && normalized[i] !== SPACE_CHAR) i++;

    // Emit all leading ▁s + word as one segment; BPE merges ▁▁, ▁▁▁, etc.
    const seg = normalized.slice(segStart, i);
    const segIds = cache.get(seg);
    if (segIds === undefined) return false;
    for (let j = 0; j < segIds.length; j++) result.push(segIds[j]);
  }
  return true;
}

// Cold-path helper — kept separate so it is never inlined into the hot loop.
function _encodeAndCache(seg, vocab, mergeRank, scratch, cache, seeds) {
  const ids = encodeSegment(seg, vocab, mergeRank, scratch, seeds);
  cache.set(seg, ids);
  return ids;
}

function encodeSPMPrepared(text, prepared) {
  if (!text) return [];

  const { vocab, mergeRank, scratch, cache, seedsByAngleBracket } = prepared;

  // Normalize: replace spaces with ▁ (Gemma3: no ▁ prepend for first char)
  const normalized = text.replace(/ /g, SPACE_CHAR);

  // Fast path: serve every segment from the segment cache.
  // After the first call, this path handles all subsequent calls for common text.
  const result = [];
  if (_scanFromCache(normalized, cache, result)) return result;

  // Cold path: at least one segment is missing — encode everything from scratch.
  // (Simpler to re-scan than to continue from the miss point.)
  result.length = 0;
  let i = 0;
  while (i < normalized.length) {
    const segStart = i;
    while (i < normalized.length && normalized[i] === SPACE_CHAR) i++;
    const spaceCount = i - segStart;

    if (i === normalized.length) {
      if (spaceCount > 0) {
        const seg = normalized.slice(segStart, i);
        const segIds = cache.get(seg) ?? _encodeAndCache(seg, vocab, mergeRank, scratch, cache, seedsByAngleBracket);
        for (let j = 0; j < segIds.length; j++) result.push(segIds[j]);
      }
      break;
    }

    while (i < normalized.length && normalized[i] !== SPACE_CHAR) i++;

    const seg = normalized.slice(segStart, i);
    const segIds = cache.get(seg) ?? _encodeAndCache(seg, vocab, mergeRank, scratch, cache, seedsByAngleBracket);
    for (let j = 0; j < segIds.length; j++) result.push(segIds[j]);
  }
  return result;
}

// Encode a single segment using MinHeap BPE.
// seeds: sorted array of {str, id, chars[]} for '<'-prefixed vocab entries that have
// no producing merge (SentencePiece user-defined symbols); matched greedily before BPE.
function encodeSegment(seg, vocab, mergeRank, scratch, seeds) {
  const chars = [...seg];
  const n = chars.length;

  ensureScratch(scratch, n);
  const { str, ids, prev, next, ver, alive, heap } = scratch;
  heap.reset();

  // Initialize nodes: one node per Unicode char, except seed tokens which collapse
  // multiple chars into a single node (longest match at each '<' position).
  let nNodes = 0;
  let pos = 0;
  while (pos < n) {
    const node = nNodes++;
    prev[node] = node - 1;
    next[node] = node + 1;
    ver[node]  = 0;
    alive[node] = 1;

    let matched = false;
    if (seeds && chars[pos] === '<') {
      for (const seed of seeds) {
        const sl = seed.chars.length;
        if (pos + sl > n) continue;
        let ok = true;
        for (let k = 0; k < sl; k++) {
          if (chars[pos + k] !== seed.chars[k]) { ok = false; break; }
        }
        if (ok) {
          str[node] = seed.str;
          ids[node] = seed.id;
          pos += sl;
          matched = true;
          break;
        }
      }
    }

    if (!matched) {
      const c = chars[pos];
      if (vocab[c] !== undefined) {
        str[node] = c;
        ids[node] = vocab[c];
      } else {
        const codePoint = c.codePointAt(0);
        const hex = codePoint.toString(16).toUpperCase().padStart(2, '0');
        const byteKey = `<0x${hex}>`;
        if (vocab[byteKey] !== undefined) {
          str[node] = byteKey;
          ids[node] = vocab[byteKey];
        } else {
          str[node] = c;
          ids[node] = vocab['<unk>'] ?? 0;
        }
      }
      pos++;
    }
  }
  next[nNodes - 1] = -1;

  for (let i = 0; i < nNodes - 1; i++) {
    const rank = mergeRank.get(`${str[i]} ${str[i + 1]}`);
    if (rank !== undefined) heap.push(rank, i, i + 1, ver[i], ver[i + 1]);
  }

  while (heap.size > 0) {
    const top = heap.pop();
    if (!top) break;
    const { left, right, verL, verR } = top;

    if (!alive[left] || !alive[right]) continue;
    if (next[left] !== right) continue;
    if (ver[left] !== verL || ver[right] !== verR) continue;

    str[left] = str[left] + str[right];
    ids[left] = vocab[str[left]] ?? vocab['<unk>'] ?? 0;
    ver[left]++;

    alive[right] = 0;
    ver[right]++;

    const nr = next[right];
    next[left] = nr;
    if (nr !== -1) prev[nr] = left;

    const pl = prev[left];
    if (pl !== -1 && alive[pl]) {
      const r = mergeRank.get(`${str[pl]} ${str[left]}`);
      if (r !== undefined) heap.push(r, pl, left, ver[pl], ver[left]);
    }
    const nl = next[left];
    if (nl !== -1 && alive[nl]) {
      const r = mergeRank.get(`${str[left]} ${str[nl]}`);
      if (r !== undefined) heap.push(r, left, nl, ver[left], ver[nl]);
    }
  }

  const result = [];
  let i = 0;
  while (i !== -1) {
    if (alive[i]) result.push(ids[i]);
    i = next[i];
  }
  return result;
}

/**
 * Decode token ids back to string.
 * @param {number[]} ids
 * @param {Object} vocabData
 * @returns {string}
 */
function decodeSPM(ids, vocabData) {
  return decodeSPMPrepared(ids, buildPreparedSPM(vocabData));
}

function decodeSPMPrepared(ids, prepared) {
  if (!ids || ids.length === 0) return '';

  let result = '';
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const str = prepared.idToStr.get(id) ?? '';
    const byteMatch = str.match(/^<0x([0-9A-Fa-f]{2})>$/);
    if (byteMatch) {
      result += String.fromCharCode(parseInt(byteMatch[1], 16));
    } else {
      result += str;
    }
  }

  // Replace ▁ with space, remove leading space
  return result.replace(new RegExp(SPACE_CHAR, 'g'), ' ').replace(/^ /, '');
}

module.exports = { buildPreparedSPM, encodeSPM, decodeSPM, encodeSPMPrepared, decodeSPMPrepared };
