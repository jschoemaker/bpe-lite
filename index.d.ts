export type Provider = 'openai' | 'openai-o200k' | 'anthropic' | 'gemini';

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
  count(text: string): number;
  countUpTo(text: string, limit: number): number;
}

/**
 * Count the number of tokens in text for the given provider.
 */
export function countTokens(text: string, provider?: Provider): number;

/**
 * Encode text to token ids.
 */
export function encode(text: string, provider?: Provider): number[];

/**
 * Decode token ids back to text.
 */
export function decode(ids: number[], provider?: Provider): string;

/**
 * Check if text is within a token limit.
 * Returns the token count if within the limit, or false if exceeded.
 * More efficient than encode() for long texts since it short-circuits.
 */
export function isWithinTokenLimit(text: string, limit: number, provider?: Provider): number | false;

/** Tokenizer instance for OpenAI cl100k_base (GPT-4, GPT-3.5). */
export function openai(): Tokenizer;

/** Tokenizer instance for OpenAI o200k_base (GPT-4o, o1, o3, o4, GPT-4.1, GPT-5). */
export function openaiO200k(): Tokenizer;

/** Tokenizer instance for Anthropic (cl100k approximation, ~95% accurate). */
export function anthropic(): Tokenizer;

/** Tokenizer instance for Gemini (Gemma 3 vocab, exact). */
export function gemini(): Tokenizer;
