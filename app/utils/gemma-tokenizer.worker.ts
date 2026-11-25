import { Tokenizer } from "@huggingface/tokenizers";
import {
  foldContentsForApi,
  getAttachmentCoverage,
  getMessageContentForApi,
} from "./message-content";
import { countGemmaRequestTokens } from "./gemma-request-budget";
import type {
  TokenizerRequest,
  TokenizerResponse,
  TokenizerResult,
} from "./gemma-tokenizer";

const ASSET_PATH = "/tokenizers/gemma-4-4d7ae498/";
let tokenizer: Promise<Tokenizer> | undefined;

function loadTokenizer(): Promise<Tokenizer> {
  if (!tokenizer) {
    const controller = new AbortController();
    tokenizer = Promise.all(
      ["tokenizer.json", "tokenizer_config.json"].map(async (name) => {
        const response = await fetch(ASSET_PATH + name, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Could not load the tokenizer");
        return response.json();
      }),
    )
      .then(([data, config]) => new Tokenizer(data, config))
      .catch((error) => {
        // One file failed so the other download is no use. Stop it.
        controller.abort();
        tokenizer = undefined;
        throw error;
      });
  }
  return tokenizer;
}

self.onmessage = async ({ data }: MessageEvent<TokenizerRequest>) => {
  let response: TokenizerResponse;
  try {
    const model = await loadTokenizer();
    // The same text is counted several times while a file is cut to fit.
    // Remember the counts but only for this one job.
    const counts = new Map<string, number>();
    const count = (text: string) => {
      const cached = counts.get(text);
      if (cached !== undefined) return cached;
      const tokens = model.encode(text, { add_special_tokens: false }).ids
        .length;
      if (counts.size >= 64) counts.clear();
      counts.set(text, tokens);
      return tokens;
    };
    let result: TokenizerResult;
    switch (data.type) {
      case "fold":
        result = foldContentsForApi(data.messages, data.keepImages, count);
        break;
      case "coverage":
        result = getAttachmentCoverage(data.files, data.pictureTokens, count);
        break;
      case "request":
        result = countGemmaRequestTokens(data.messages, count);
        break;
      case "subtract": {
        const budget = { remaining: data.remaining };
        const seen = new Set(data.seen);
        getMessageContentForApi(data.message, true, seen, budget, count);
        result = { remaining: budget.remaining, seen: Array.from(seen) };
        break;
      }
    }
    response = { id: data.id, result };
  } catch {
    response = { id: data.id, error: "Token counting failed" };
  }
  self.postMessage(response);
};
