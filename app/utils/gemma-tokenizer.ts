import type {
  FileAttachment,
  MultimodalContent,
  RequestMessage,
} from "../client/api";
import {
  type AttachmentCoverage,
  foldContentsForApi,
  getMessageContentForApi,
  getMessageFiles,
  isAttachmentBudgetFull,
} from "./message-content";
import { imageTokensFromUrl } from "./gemma-image-tokens";
import {
  GEMMA_INPUT_TOKEN_BUDGET,
  GEMMA_REQUEST_TOO_LONG,
} from "./gemma-request-budget";

type Contents = (string | MultimodalContent[])[];
type Jobs = {
  fold: { messages: RequestMessage[]; keepImages: boolean };
  coverage: { files: FileAttachment[]; pictureTokens: number[] };
  request: { messages: RequestMessage[] };
  subtract: { message: RequestMessage; remaining: number; seen: string[] };
};
type Results = {
  fold: Contents;
  coverage: AttachmentCoverage;
  request: number;
  subtract: { remaining: number; seen: string[] };
};

export type TokenizerRequest = {
  [K in keyof Jobs]: { id: number; type: K } & Jobs[K];
}[keyof Jobs];
export type TokenizerResult = Results[keyof Results];
export type TokenizerResponse =
  | { id: number; result: TokenizerResult }
  | { id: number; error: string };

export const TOKENIZER_LOAD_ERROR =
  "Teksten kunne ikke tælles. Tjek din internetforbindelse, og prøv igen.";
let worker: Worker | undefined;
let nextId = 0;
type PendingJob = {
  resolve: (result: TokenizerResult) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
  signal?: AbortSignal;
  abort: () => void;
};
const pending = new Map<number, PendingJob>();

function cleanUp(job: PendingJob) {
  clearTimeout(job.timeout);
  job.signal?.removeEventListener("abort", job.abort);
}

function stopWorker() {
  if (worker) {
    // A stopped worker can still report an error. It must not stop a new one.
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }
  worker = undefined;
  for (const job of pending.values()) {
    cleanUp(job);
    job.reject(new Error(TOKENIZER_LOAD_ERROR));
  }
  pending.clear();
}

function run<K extends keyof Jobs>(
  type: K,
  data: Jobs[K],
  signal?: AbortSignal,
): Promise<Results[K]> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("aborted"));
      return;
    }
    try {
      if (!worker) {
        worker = new Worker(
          new URL("./gemma-tokenizer.worker.ts", import.meta.url),
          { type: "module" },
        );
        worker.onmessage = ({ data }: MessageEvent<TokenizerResponse>) => {
          const job = pending.get(data.id);
          if (!job) return;
          cleanUp(job);
          pending.delete(data.id);
          if ("error" in data) job.reject(new Error(TOKENIZER_LOAD_ERROR));
          else job.resolve(data.result);
        };
        worker.onerror = stopWorker;
        worker.onmessageerror = stopWorker;
      }
      const id = nextId++;
      const abort = () => {
        const job = pending.get(id);
        if (!job) return;
        cleanUp(job);
        pending.delete(id);
        reject(new Error("aborted"));
        // Stop the worker so a new try does not wait behind the cancelled job.
        // But not while another job is still using it.
        if (pending.size === 0) stopWorker();
      };
      pending.set(id, {
        resolve: (result) => resolve(result as Results[K]),
        reject,
        // Give up after a minute so sending never hangs on a dead worker.
        timeout: setTimeout(stopWorker, 60000),
        signal,
        abort,
      });
      signal?.addEventListener("abort", abort, { once: true });
      worker.postMessage({ id, type, ...data });
    } catch {
      stopWorker();
      reject(new Error(TOKENIZER_LOAD_ERROR));
    }
  });
}

function fileForWorker(
  file: FileAttachment,
  imageUrl: (url: string) => string,
): FileAttachment {
  return {
    name: file.name,
    mime: file.mime,
    id: file.id,
    text: file.text,
    imageCount: file.imageCount,
    pdfPageCount: file.pdfPageCount,
    imageTokens: file.images?.map(imageTokensFromUrl) ?? file.imageTokens,
    images: file.images?.map(imageUrl),
  };
}

function messageForWorker(
  message: RequestMessage,
  imageUrl: (url: string) => string,
): RequestMessage {
  return {
    role: message.role,
    content:
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => {
            if (part.type === "file" && part.file) {
              return { type: "file", file: fileForWorker(part.file, imageUrl) };
            }
            if (part.type === "image_url") {
              return {
                type: "image_url",
                image_url: {
                  url: imageUrl(part.image_url?.url ?? ""),
                  tokens: imageTokensFromUrl(part.image_url?.url ?? ""),
                },
              };
            }
            return { type: part.type, text: part.text };
          }),
  };
}

export async function foldContentsWithTokenizer(
  messages: RequestMessage[],
  keepImages: boolean,
  signal?: AbortSignal,
): Promise<Contents> {
  // The worker only needs what each picture costs. The big picture data stays
  // here and is put back on the pictures that fit.
  const urls: string[] = [];
  const prepared = messages.map((message) =>
    messageForWorker(message, (url) => String(urls.push(url) - 1)),
  );
  if (signal?.aborted) throw new Error("aborted");
  const contents = prepared.some(
    (message) => getMessageFiles(message).length > 0,
  )
    ? await run("fold", { messages: prepared, keepImages }, signal)
    : foldContentsForApi(prepared, keepImages);
  return contents.map((content) =>
    typeof content === "string"
      ? content
      : content.map((part) =>
          part.type === "image_url" && part.image_url
            ? {
                ...part,
                image_url: { url: urls[Number(part.image_url.url)] },
              }
            : part,
        ),
  );
}

export async function isAttachmentBudgetFullWithTokenizer(
  files: FileAttachment[],
  pictures: string[],
): Promise<boolean> {
  // Pictures alone need no text tokenizer.
  if (files.length === 0)
    return isAttachmentBudgetFull(files, pictures.map(imageTokensFromUrl));
  return (
    (await getAttachmentCoverageWithTokenizer(files, pictures)).remaining <= 0
  );
}

export async function subtractMessageCost(
  message: RequestMessage,
  budget: { remaining: number },
  seen: Set<string>,
  signal?: AbortSignal,
): Promise<void> {
  const prepared = messageForWorker(message, () => "");
  if (signal?.aborted) throw new Error("aborted");
  if (getMessageFiles(prepared).length === 0) {
    getMessageContentForApi(prepared, true, seen, budget);
    return;
  }
  const result = await run(
    "subtract",
    {
      message: prepared,
      remaining: budget.remaining,
      seen: Array.from(seen),
    },
    signal,
  );
  budget.remaining = result.remaining;
  for (const key of result.seen) seen.add(key);
}

export function getAttachmentCoverageWithTokenizer(
  files: FileAttachment[],
  pictures: string[],
): Promise<AttachmentCoverage> {
  return run("coverage", {
    files: files.map((file) => fileForWorker(file, () => "")),
    pictureTokens: pictures.map(imageTokensFromUrl),
  });
}

export async function assertGemmaRequestFits(
  messages: RequestMessage[],
  signal?: AbortSignal,
): Promise<void> {
  const prepared = messages.map((message) =>
    messageForWorker(message, () => ""),
  );
  const tokens = await run("request", { messages: prepared }, signal);
  if (tokens > GEMMA_INPUT_TOKEN_BUDGET)
    throw new Error(GEMMA_REQUEST_TOO_LONG);
}
