'use strict';

const {
  buildPreparedTiktoken,
  encodeTiktokenPrepared,
  decodeTiktokenPrepared,
  countTiktokenPrepared,
  countTiktokenUpToPrepared,
} = require('./bpe');
const { buildPreparedSPM, encodeSPMPrepared, decodeSPMPrepared } = require('./spm');

class Tokenizer {
  constructor(vocabData) {
    this._data = vocabData;
    this._engine = vocabData.engine;
    this._preparedTiktoken = null;
    this._preparedSPM = null;

    if (this._engine !== 'tiktoken' && this._engine !== 'spm') {
      throw new Error(`Unknown tokenizer engine: ${this._engine}`);
    }

    if (this._engine === 'tiktoken') {
      this._preparedTiktoken = buildPreparedTiktoken(vocabData);
    } else {
      this._preparedSPM = buildPreparedSPM(vocabData);
    }
  }

  encode(text) {
    if (this._engine === 'tiktoken') return encodeTiktokenPrepared(text, this._preparedTiktoken);
    return encodeSPMPrepared(text, this._preparedSPM);
  }

  decode(ids) {
    if (this._engine === 'tiktoken') return decodeTiktokenPrepared(ids, this._preparedTiktoken);
    return decodeSPMPrepared(ids, this._preparedSPM);
  }

  count(text) {
    if (this._engine === 'tiktoken') return countTiktokenPrepared(text, this._preparedTiktoken);
    return this.encode(text).length;
  }

  /**
   * Count tokens, stopping as soon as the count exceeds limit.
   * More efficient than encode() for token limit checks on long text.
   * @param {string} text
   * @param {number} limit
   * @returns {number}
   */
  countUpTo(text, limit) {
    if (this._engine === 'tiktoken') return countTiktokenUpToPrepared(text, this._preparedTiktoken, limit);
    // SPM encodes the whole text as one unit — no clean early exit, just encode and count
    return this.encode(text).length;
  }
}

module.exports = { Tokenizer };
