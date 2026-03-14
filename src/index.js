'use strict';

const path = require('path');
const fs = require('fs');
const zlib = require('zlib');
const { Tokenizer } = require('./tokenizer');

const VOCABS_DIR = path.join(__dirname, '..', 'vocabs');

// Lazy-loaded tokenizer instances (created once per provider per process)
const _cache = {};

function loadTokenizer(provider) {
  if (_cache[provider]) return _cache[provider];

  const gzPath   = path.join(VOCABS_DIR, `${provider}.json.gz`);
  const jsonPath  = path.join(VOCABS_DIR, `${provider}.json`);

  let data;
  if (fs.existsSync(gzPath)) {
    data = JSON.parse(zlib.gunzipSync(fs.readFileSync(gzPath)).toString('utf8'));
  } else if (fs.existsSync(jsonPath)) {
    data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  } else {
    throw new Error(
      `Vocab file not found for provider "${provider}".\n` +
      'Run "node scripts/build-vocabs.js" to build vocab files.'
    );
  }

  _cache[provider] = new Tokenizer(data);
  return _cache[provider];
}

/**
 * Count tokens in text for a given provider.
 * @param {string} text
 * @param {'openai'|'anthropic'|'gemini'} provider
 * @returns {number}
 */
function countTokens(text, provider = 'openai') {
  return loadTokenizer(provider).count(text);
}

/**
 * Encode text to token ids.
 * @param {string} text
 * @param {'openai'|'anthropic'|'gemini'} provider
 * @returns {number[]}
 */
function encode(text, provider = 'openai') {
  return loadTokenizer(provider).encode(text);
}

/**
 * Decode token ids back to text.
 * @param {number[]} ids
 * @param {'openai'|'openai-o200k'|'anthropic'|'gemini'} provider
 * @returns {string}
 */
function decode(ids, provider = 'openai') {
  return loadTokenizer(provider).decode(ids);
}

/**
 * Check if text is within a token limit without necessarily encoding the whole string.
 * Returns false if the limit is exceeded, otherwise returns the token count.
 * @param {string} text
 * @param {number} limit
 * @param {'openai'|'openai-o200k'|'anthropic'|'gemini'} provider
 * @returns {number|false}
 */
function isWithinTokenLimit(text, limit, provider = 'openai') {
  const count = loadTokenizer(provider).countUpTo(text, limit);
  return count <= limit ? count : false;
}

/** Get a Tokenizer instance for OpenAI (cl100k_base — GPT-4, GPT-3.5). */
function openai() { return loadTokenizer('openai'); }

/** Get a Tokenizer instance for OpenAI modern models (o200k_base — GPT-4o, o1, o3, o4, GPT-4.1, GPT-5). */
function openaiO200k() { return loadTokenizer('openai-o200k'); }

/** Get a Tokenizer instance for Anthropic (cl100k approximation). */
function anthropic() { return loadTokenizer('anthropic'); }

/** Get a Tokenizer instance for Gemini (Gemma3 vocab). */
function gemini() { return loadTokenizer('gemini'); }

module.exports = { countTokens, encode, decode, isWithinTokenLimit, openai, openaiO200k, anthropic, gemini };
