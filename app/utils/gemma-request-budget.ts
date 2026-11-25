import type { RequestMessage } from "../client/api";
import { imageCost } from "./message-content";

export const GEMMA_CONTEXT_TOKENS = 32768;
// Room that is always kept free for the answer and its thinking. The answer
// can be longer than this when the request is small.
const GEMMA_REPLY_RESERVE = 4096;
const GEMMA_GATEWAY_RESERVE = 256;
export const GEMMA_INPUT_TOKEN_BUDGET =
  GEMMA_CONTEXT_TOKENS - GEMMA_REPLY_RESERVE - GEMMA_GATEWAY_RESERVE;
export const GEMMA_REQUEST_TOO_LONG =
  "Beskeden, materialerne og chathistorikken fylder for meget tilsammen. " +
  "Forkort beskeden, vælg færre materialer, eller start en ny chat.";

function trimForTemplate(text: string): string {
  // The server trims text the way Python does. That keeps U+FEFF.
  return text.replace(
    /^[\p{White_Space}\u001c-\u001f]+|[\p{White_Space}\u001c-\u001f]+$/gu,
    "",
  );
}

// Counts a whole request the way the server does. It follows the Gemma 4 chat
// template of vLLM 0.26 with thinking on. Checked against the server with
// scripts/test-dbc-token-counts.mjs.
// https://github.com/vllm-project/vllm/blob/v0.26.0/examples/tool_chat_template_gemma4.jinja
// Call it with the final messages. File text added and system prompts merged.
export function countGemmaRequestTokens(
  messages: RequestMessage[],
  count: (text: string) => number,
): number {
  let imageTokens = 0;
  const textPart = (text: string, assistant: boolean) =>
    trimForTemplate(
      assistant
        ? text
            .split("<channel|>")
            .map((part) => part.split("<|channel>")[0])
            .join("")
        : text,
    );
  const content = (message: RequestMessage) => {
    const assistant = message.role === "assistant";
    if (typeof message.content === "string")
      return textPart(message.content, assistant);
    return message.content
      .map((part) => {
        if (part.type === "text") return textPart(part.text ?? "", assistant);
        if (part.type !== "image_url") return "";
        // The two marks around a picture are in the text and counted there.
        // The rest of the picture's cost is added as a number.
        imageTokens += imageCost(part.image_url?.tokens) - 2;
        return "<|image><image|>";
      })
      .join("");
  };
  let serialized = "<bos><|turn>system\n<|think|>\n";
  let start = 0;
  if (messages[0]?.role === "system") {
    const first = messages[0];
    serialized +=
      typeof first.content === "string"
        ? trimForTemplate(first.content)
        : first.content
            .map((part) => trimForTemplate(part.text ?? "") + " ")
            .join("");
    start = 1;
  }
  serialized += "<turn|>\n";
  for (let i = start; i < messages.length; i += 1) {
    const message = messages[i];
    const assistant = message.role === "assistant";
    if (!assistant || i === start || messages[i - 1].role !== "assistant")
      serialized += `<|turn>${assistant ? "model" : message.role}\n`;
    serialized += content(message);
    serialized +=
      assistant && messages[i + 1]?.role === "assistant" ? "\n" : "<turn|>\n";
  }
  return count(serialized + "<|turn>model\n") + imageTokens;
}
