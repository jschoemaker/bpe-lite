'use strict';

/**
 * tiktoken-style BPE encoder.
 * Vocab keys are base64-encoded byte sequences, values are ranks (merge priority).
 * Lower rank = higher priority in merges.
 */

// Fast check: does a pre-tokenized chunk contain any letter or digit?
// Non-word chunks (pure symbols, punctuation, spaces) are eligible for batching.
const WORD_CHAR_RE = /[\p{L}\p{N}]/u;

// Byte → single-byte "binary string" lookup (pre-built at module load)
const BYTE_STRS = (() => {
  const out = new Array(256);
  for (let i = 0; i < 256; i++) out[i] = String.fromCharCode(i);
  return out;
})();

// opt D — shared UTF-8 encode buffer; one allocation for the process lifetime.
// Node.js is single-threaded and all encode paths are synchronous, so this is safe.
const _sb = { buf: Buffer.allocUnsafe(4096), cap: 4096 };

function writeChunk(chunk) {
  const maxNeeded = chunk.length * 4; // max 4 UTF-8 bytes per JS char
  if (maxNeeded > _sb.cap) {
    _sb.cap = maxNeeded * 2;
    _sb.buf = Buffer.allocUnsafe(_sb.cap);
  }
  return _sb.buf.write(chunk, 0, 'utf8');
}

class MinHeap {
  constructor() {
    this.ranks = [];
    this.left  = [];
    this.right = [];
    this.verL  = [];
    this.verR  = [];
  }

  // opt B — reset to empty without de-allocating internal arrays
  reset() {
    this.ranks.length = 0;
    this.left.length  = 0;
    this.right.length = 0;
    this.verL.length  = 0;
    this.verR.length  = 0;
  }

  get size() { return this.ranks.length; }

  push(rank, left, right, verL, verR) {
    const i = this.ranks.length;
    this.ranks.push(rank);
    this.left.push(left);
    this.right.push(right);
    this.verL.push(verL);
    this.verR.push(verR);
    this._siftUp(i);
  }

  pop() {
    const n = this.ranks.length;
    if (n === 0) return null;

    const rank = this.ranks[0], left = this.left[0], right = this.right[0];
    const verL = this.verL[0], verR = this.verR[0];

    const last = n - 1;
    if (last === 0) {
      this.ranks.pop(); this.left.pop(); this.right.pop(); this.verL.pop(); this.verR.pop();
      return { rank, left, right, verL, verR };
    }

    this.ranks[0] = this.ranks[last]; this.left[0] = this.left[last];
    this.right[0] = this.right[last]; this.verL[0] = this.verL[last]; this.verR[0] = this.verR[last];

    this.ranks.pop(); this.left.pop(); this.right.pop(); this.verL.pop(); this.verR.pop();
    this._siftDown(0);
    return { rank, left, right, verL, verR };
  }

  _siftUp(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.ranks[p] < this.ranks[i]) break;
      if (this.ranks[p] === this.ranks[i] && this.left[p] <= this.left[i]) break;
      this._swap(i, p);
      i = p;
    }
  }

  _siftDown(i) {
    const n = this.ranks.length;
    while (true) {
      const l = i * 2 + 1;
      if (l >= n) break;
      const r = l + 1;
      let m = l;
      if (r < n) {
        if (this.ranks[r] < this.ranks[l] ||
            (this.ranks[r] === this.ranks[l] && this.left[r] < this.left[l])) m = r;
      }
      if (this.ranks[i] < this.ranks[m]) break;
      if (this.ranks[i] === this.ranks[m] && this.left[i] <= this.left[m]) break;
      this._swap(i, m);
      i = m;
    }
  }

  _swap(i, j) {
    [this.ranks[i], this.ranks[j]] = [this.ranks[j], this.ranks[i]];
    [this.left[i],  this.left[j]]  = [this.left[j],  this.left[i]];
    [this.right[i], this.right[j]] = [this.right[j], this.right[i]];
    [this.verL[i],  this.verL[j]]  = [this.verL[j],  this.verL[i]];
    [this.verR[i],  this.verR[j]]  = [this.verR[j],  this.verR[i]];
  }
}

function compilePretokenizer(patternStr) {
  if (!patternStr) return { type: 'none' };

  const regexStr = patternStr
    .replace(/\(\?i:/g, '(?:')
    .replace(/\\p\{L\}/g, '\\p{L}')
    .replace(/\\p\{N\}/g, '\\p{N}');

  try {
    return { type: 'regex', re: new RegExp(regexStr, 'guy') };
  } catch {
    try {
      return { type: 'regex', re: new RegExp(regexStr, 'gi') };
    } catch {
      return { type: 'fallback' };
    }
  }
}

function pretokenize(text, compiled) {
  if (!text) return [];
  if (!compiled || compiled.type === 'none') return [text];
  if (compiled.type === 'fallback') return text.match(/\S+|\s+/g) || [text];
  return text.match(compiled.re) || [text];
}

function buildPreparedTiktoken(vocabData) {
  const { vocab, specialTokens = {}, pattern, normalize } = vocabData;

  const vocabBin = new Map();
  let maxId = -1;
  for (const [b64, id] of Object.entries(vocab)) {
    const buf = Buffer.from(b64, 'base64');
    vocabBin.set(buf.toString('latin1'), id);
    if (id > maxId) maxId = id;
  }

  const specials = Object.entries(specialTokens);
  for (const [, id] of specials) {
    if (id > maxId) maxId = id;
  }

  const idToBytes = new Array(maxId + 1);
  for (const [b64, id] of Object.entries(vocab)) {
    idToBytes[id] = Buffer.from(b64, 'base64');
  }
  for (const [str, id] of specials) {
    idToBytes[id] = Buffer.from(str, 'utf8');
  }

  return {
    vocabBin,
    idToBytes,
    specials,
    patternCompiled: compilePretokenizer(pattern),
    normalize: normalize || null,
    // symbolBatch: merge consecutive non-word regex chunks before BPE so that
    // cross-character byte merges can fire (matches Claude's no-regex byte BPE).
    symbolBatch: !!vocabData.symbolBatch,
    // opt A — per-instance chunk cache: chunk string → ids[]
    cache: new Map(),
    // opt B — per-instance grow-only scratch (reused across chunks)
    scratch: { str: null, prev: null, next: null, ver: null, alive: null, cap: 0, heap: new MinHeap() },
  };
}

// opt B — grow scratch arrays only when needed
function ensureScratch(scratch, n) {
  if (n <= scratch.cap) return;
  const cap = n * 2;
  scratch.str   = new Array(cap);
  scratch.prev  = new Int32Array(cap);
  scratch.next  = new Int32Array(cap);
  scratch.ver   = new Int32Array(cap);
  scratch.alive = new Uint8Array(cap);
  scratch.cap   = cap;
}

// opt E — unified encode for one pre-tokenized chunk (replaces bpeEncode + bpeCount)
// Precondition: writeChunk(chunk) was just called and returned n.
// Reads from _sb.buf[0..n-1]. Returns ids[].
function bpeChunk(n, vocabBin, scratch) {
  const buf = _sb.buf;

  // opt C — fast path: single byte
  if (n === 1) {
    const id = vocabBin.get(BYTE_STRS[buf[0]]);
    return id === undefined ? [] : [id];
  }

  // opt C — fast path: two bytes
  if (n === 2) {
    const s0 = BYTE_STRS[buf[0]], s1 = BYTE_STRS[buf[1]];
    const merged = vocabBin.get(s0 + s1);
    if (merged !== undefined) return [merged];
    const r = [];
    const i0 = vocabBin.get(s0); if (i0 !== undefined) r.push(i0);
    const i1 = vocabBin.get(s1); if (i1 !== undefined) r.push(i1);
    return r;
  }

  // General path — reuse scratch arrays (opt B), reuse heap (opt B)
  ensureScratch(scratch, n);
  const { str, prev, next, ver, alive, heap } = scratch;
  heap.reset();

  for (let i = 0; i < n; i++) {
    str[i]   = BYTE_STRS[buf[i]];
    prev[i]  = i - 1;
    next[i]  = i + 1;
    ver[i]   = 0;
    alive[i] = 1;
  }
  next[n - 1] = -1;

  for (let i = 0; i < n - 1; i++) {
    const rank = vocabBin.get(str[i] + str[i + 1]);
    if (rank !== undefined) heap.push(rank, i, i + 1, 0, 0);
  }

  while (heap.size > 0) {
    const top = heap.pop();
    if (!top) break;
    const { left, right, verL, verR } = top;
    if (!alive[left] || !alive[right]) continue;
    if (next[left] !== right) continue;
    if (ver[left] !== verL || ver[right] !== verR) continue;

    str[left] = str[left] + str[right];
    ver[left]++;
    alive[right] = 0;
    ver[right]++;

    const nr = next[right];
    next[left] = nr;
    if (nr !== -1) prev[nr] = left;

    const pl = prev[left];
    if (pl !== -1 && alive[pl]) {
      const rank = vocabBin.get(str[pl] + str[left]);
      if (rank !== undefined) heap.push(rank, pl, left, ver[pl], ver[left]);
    }
    const nl = next[left];
    if (nl !== -1 && alive[nl]) {
      const rank = vocabBin.get(str[left] + str[nl]);
      if (rank !== undefined) heap.push(rank, left, nl, ver[left], ver[nl]);
    }
  }

  const ids = [];
  let i = 0;
  while (i !== -1) {
    if (alive[i]) {
      const id = vocabBin.get(str[i]);
      if (id !== undefined) ids.push(id);
    }
    i = next[i];
  }
  return ids;
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

function splitOnSpecials(text, specials) {
  if (specials.length === 0) return [{ text, isSpecial: false }];

  const result = [];
  let remaining = text;

  while (remaining.length > 0) {
    let bestIdx = -1, bestStr = null, bestId = null;

    for (const [str, id] of specials) {
      const idx = remaining.indexOf(str);
      if (idx === -1) continue;
      if (
        bestIdx === -1 ||
        idx < bestIdx ||
        (idx === bestIdx && bestStr && str.length > bestStr.length)
      ) {
        bestIdx = idx; bestStr = str; bestId = id;
      }
    }

    if (bestIdx === -1) { result.push({ text: remaining, isSpecial: false }); break; }
    if (bestIdx > 0) result.push({ text: remaining.slice(0, bestIdx), isSpecial: false });
    result.push({ isSpecial: true, id: bestId });
    remaining = remaining.slice(bestIdx + bestStr.length);
  }

  return result;
}

// ─── Prepared-object API ──────────────────────────────────────────────────────

function encodeTiktokenPrepared(text, prepared) {
  if (!text) return [];
  if (prepared.normalize) text = text.normalize(prepared.normalize);

  const ids = [];
  const { vocabBin, scratch, cache, patternCompiled, specials } = prepared;
  const pieces = splitOnSpecials(text, specials);

  for (const piece of pieces) {
    if (piece.isSpecial) { ids.push(piece.id); continue; }
    const t = piece.text;

    if (patternCompiled.type === 'regex') {
      // opt — exec loop avoids materialising the full matches array
      const re = patternCompiled.re;
      re.lastIndex = 0;
      let m;
      if (prepared.symbolBatch) {
        // Merge consecutive non-word chunks (symbols, punctuation, spaces) into one
        // BPE input so cross-character byte merges can fire — matches Claude's behavior.
        let symBuf = null;
        while ((m = re.exec(t)) !== null) {
          const chunk = m[0];
          if (!WORD_CHAR_RE.test(chunk)) {
            symBuf = symBuf === null ? chunk : symBuf + chunk;
          } else {
            if (symBuf !== null) {
              let symIds = cache.get(symBuf);
              if (symIds === undefined) { symIds = bpeChunk(writeChunk(symBuf), vocabBin, scratch); cache.set(symBuf, symIds); }
              for (let i = 0; i < symIds.length; i++) ids.push(symIds[i]);
              symBuf = null;
            }
            let chunkIds = cache.get(chunk);
            if (chunkIds === undefined) { chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch); cache.set(chunk, chunkIds); }
            for (let i = 0; i < chunkIds.length; i++) ids.push(chunkIds[i]);
          }
        }
        if (symBuf !== null) {
          let symIds = cache.get(symBuf);
          if (symIds === undefined) { symIds = bpeChunk(writeChunk(symBuf), vocabBin, scratch); cache.set(symBuf, symIds); }
          for (let i = 0; i < symIds.length; i++) ids.push(symIds[i]);
        }
      } else {
        while ((m = re.exec(t)) !== null) {
          const chunk = m[0];
          let chunkIds = cache.get(chunk);
          if (chunkIds === undefined) {
            chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
            cache.set(chunk, chunkIds);
          }
          for (let i = 0; i < chunkIds.length; i++) ids.push(chunkIds[i]);
        }
      }
    } else {
      const chunks = patternCompiled.type === 'none' ? [t] : (t.match(/\S+|\s+/g) || [t]);
      for (let ci = 0; ci < chunks.length; ci++) {
        const chunk = chunks[ci];
        let chunkIds = cache.get(chunk);
        if (chunkIds === undefined) {
          chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
          cache.set(chunk, chunkIds);
        }
        for (let i = 0; i < chunkIds.length; i++) ids.push(chunkIds[i]);
      }
    }
  }

  return ids;
}

function decodeTiktokenPrepared(ids, prepared) {
  if (!ids || ids.length === 0) return '';

  const bufs = [];
  for (let i = 0; i < ids.length; i++) {
    const bytes = prepared.idToBytes[ids[i]];
    if (bytes) bufs.push(bytes);
  }
  return Buffer.concat(bufs).toString('utf8');
}

function countTiktokenPrepared(text, prepared) {
  if (!text) return 0;
  if (prepared.normalize) text = text.normalize(prepared.normalize);

  const { vocabBin, scratch, cache, patternCompiled, specials } = prepared;
  const pieces = splitOnSpecials(text, specials);
  let count = 0;

  for (const piece of pieces) {
    if (piece.isSpecial) { count++; continue; }
    const t = piece.text;

    if (patternCompiled.type === 'regex') {
      const re = patternCompiled.re;
      re.lastIndex = 0;
      let m;
      if (prepared.symbolBatch) {
        let symBuf = null;
        while ((m = re.exec(t)) !== null) {
          const chunk = m[0];
          if (!WORD_CHAR_RE.test(chunk)) {
            symBuf = symBuf === null ? chunk : symBuf + chunk;
          } else {
            if (symBuf !== null) {
              let symIds = cache.get(symBuf);
              if (symIds === undefined) { symIds = bpeChunk(writeChunk(symBuf), vocabBin, scratch); cache.set(symBuf, symIds); }
              count += symIds.length;
              symBuf = null;
            }
            let chunkIds = cache.get(chunk);
            if (chunkIds === undefined) { chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch); cache.set(chunk, chunkIds); }
            count += chunkIds.length;
          }
        }
        if (symBuf !== null) {
          let symIds = cache.get(symBuf);
          if (symIds === undefined) { symIds = bpeChunk(writeChunk(symBuf), vocabBin, scratch); cache.set(symBuf, symIds); }
          count += symIds.length;
        }
      } else {
        while ((m = re.exec(t)) !== null) {
          const chunk = m[0];
          let chunkIds = cache.get(chunk);
          if (chunkIds === undefined) {
            chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
            cache.set(chunk, chunkIds);
          }
          count += chunkIds.length;
        }
      }
    } else {
      const chunks = patternCompiled.type === 'none' ? [t] : (t.match(/\S+|\s+/g) || [t]);
      for (let ci = 0; ci < chunks.length; ci++) {
        const chunk = chunks[ci];
        let chunkIds = cache.get(chunk);
        if (chunkIds === undefined) {
          chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
          cache.set(chunk, chunkIds);
        }
        count += chunkIds.length;
      }
    }
  }

  return count;
}

function countTiktokenUpToPrepared(text, prepared, limit) {
  if (!text) return 0;
  if (prepared.normalize) text = text.normalize(prepared.normalize);

  const { vocabBin, scratch, cache, patternCompiled, specials } = prepared;
  const pieces = splitOnSpecials(text, specials);
  let count = 0;

  outer: for (const piece of pieces) {
    if (piece.isSpecial) {
      if (++count > limit) break;
      continue;
    }
    const t = piece.text;

    if (patternCompiled.type === 'regex') {
      const re = patternCompiled.re;
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(t)) !== null) {
        const chunk = m[0];
        let chunkIds = cache.get(chunk);
        if (chunkIds === undefined) {
          chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
          cache.set(chunk, chunkIds);
        }
        count += chunkIds.length;
        if (count > limit) break outer;
      }
    } else {
      const chunks = patternCompiled.type === 'none' ? [t] : (t.match(/\S+|\s+/g) || [t]);
      for (let ci = 0; ci < chunks.length; ci++) {
        const chunk = chunks[ci];
        let chunkIds = cache.get(chunk);
        if (chunkIds === undefined) {
          chunkIds = bpeChunk(writeChunk(chunk), vocabBin, scratch);
          cache.set(chunk, chunkIds);
        }
        count += chunkIds.length;
        if (count > limit) break outer;
      }
    }
  }

  return count;
}

// ─── Standalone wrappers (build prepared fresh each call — used by tests / direct API) ──

function encodeTiktoken(text, vocabData) {
  return encodeTiktokenPrepared(text, buildPreparedTiktoken(vocabData));
}

function decodeTiktoken(ids, vocabData) {
  return decodeTiktokenPrepared(ids, buildPreparedTiktoken(vocabData));
}

function countTiktokenUpTo(text, vocabData, limit) {
  return countTiktokenUpToPrepared(text, buildPreparedTiktoken(vocabData), limit);
}

module.exports = {
  MinHeap,
  encodeTiktoken,
  decodeTiktoken,
  countTiktokenUpTo,
  buildPreparedTiktoken,
  encodeTiktokenPrepared,
  decodeTiktokenPrepared,
  countTiktokenPrepared,
  countTiktokenUpToPrepared,
};
