/**
 * Vercel AI SDK adapters: `import { ... } from "@hypergraph/sdk/ai"`.
 *
 * ```ts
 * const model = wrapLanguageModel({
 *   model: openai("gpt-4o-mini"),
 *   middleware: hypergraphMemoryMiddleware({ client, sessionId }),
 * });
 * const result = streamText({
 *   model,
 *   messages,
 *   tools: { recall: hypergraphContextTool(client) },
 *   onFinish: ({ response }) => saveMessages(client, sessionId, [lastUserMessage, ...response.messages]),
 * });
 * ```
 */
import { type LanguageModelMiddleware, type ModelMessage, tool } from "ai";
import { z } from "zod";

import type { HypergraphClient } from "./client.js";
import { type ChatRecord, type Embed, SessionMemory } from "./memory.js";
import { type Role, entitiesOf, sessionNode } from "./session.js";
import type { ScoredEdge } from "./types.js";

type CallOptions = Parameters<NonNullable<LanguageModelMiddleware["transformParams"]>>[0]["params"];
type PromptMessage = CallOptions["prompt"][number];

export const CONTEXT_HEADER = "Relevant facts from temporal memory (validity window, event, participants):";

function isoDate(epoch: number | undefined): string {
  return epoch === undefined ? "?" : new Date(epoch * 1000).toISOString().slice(0, 10);
}

/** Compact one-line-per-fact rendering used in prompts and tool results. */
export function renderFact(edge: ScoredEdge): string {
  const window = `[${isoDate(edge.validFrom)} → ${edge.isOngoing ? "now" : isoDate(edge.validTo)}]`;
  const who = entitiesOf(edge.participants).join(", ");
  return `${window} ${edge.eventType}(${who})${edge.summary ? `: ${edge.summary}` : ""}`;
}

function lastUserText(prompt: PromptMessage[]): string {
  for (let i = prompt.length - 1; i >= 0; i--) {
    const message = prompt[i];
    if (message?.role === "user") {
      return message.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n");
    }
  }
  return "";
}

export interface MemoryMiddlewareOptions {
  client: HypergraphClient;
  sessionId: string;
  /** Embeds the latest user message to rank facts by similarity. */
  embed?: Embed;
  /** Extra seed entities, fixed or derived from the user's message. */
  seeds?: string[] | ((query: string) => string[]);
  /** Facts to inject (default 8). */
  topK?: number;
  /** Traversal depth from the session node (default 2). */
  kHops?: number;
  decayLambda?: number;
}

/**
 * Middleware that injects time-valid facts reachable from the session (and
 * from `seeds`) as a system message before each model call.
 */
export function hypergraphMemoryMiddleware(options: MemoryMiddlewareOptions): LanguageModelMiddleware {
  const memory = new SessionMemory(options.client, options.sessionId, {
    embed: options.embed,
    decayLambda: options.decayLambda,
  });
  return {
    specificationVersion: "v4",
    transformParams: async ({ params }) => {
      const query = lastUserText(params.prompt);
      const seeds = typeof options.seeds === "function" ? options.seeds(query) : (options.seeds ?? []);
      const facts = await memory.relevant(query, options.topK ?? 8, { kHops: options.kHops ?? 2, seeds });
      const lines = facts.filter((f) => f.summary || f.participants.length > 2).map(renderFact);
      if (lines.length === 0) return params;
      const context: PromptMessage = { role: "system", content: `${CONTEXT_HEADER}\n${lines.join("\n")}` };
      const firstNonSystem = params.prompt.findIndex((m) => m.role !== "system");
      const at = firstNonSystem === -1 ? params.prompt.length : firstNonSystem;
      return { ...params, prompt: [...params.prompt.slice(0, at), context, ...params.prompt.slice(at)] };
    },
  };
}

export interface ContextToolOptions {
  sessionId?: string;
  embed?: Embed;
  /** Default traversal depth (default 3). */
  kHops?: number;
  topK?: number;
}

const contextInput = z.object({
  entities: z
    .array(z.string())
    .min(1)
    .describe("Entity ids to start from, e.g. 'org:acme' or 'person:jane_doe'"),
  question: z.string().optional().describe("What you want to know; used to rank facts"),
  asOf: z
    .string()
    .optional()
    .describe("ISO-8601 date the facts must be valid at; defaults to now"),
  hops: z.number().int().min(1).max(4).optional().describe("How many relationship hops to follow"),
});

/** A tool the model can call for multi-hop, point-in-time fact retrieval. */
export function hypergraphContextTool(client: HypergraphClient, options: ContextToolOptions = {}) {
  return tool({
    description:
      "Look up facts about entities in temporal memory, following relationships up to 4 hops. " +
      "Facts are only returned if they were valid at the requested date.",
    inputSchema: contextInput,
    execute: async ({ entities, question, asOf, hops }) => {
      const seeds = options.sessionId ? [sessionNode(options.sessionId), ...entities] : entities;
      const embedding = question && options.embed ? await options.embed(question) : undefined;
      const asOfEpoch = asOf ? Math.floor(Date.parse(asOf) / 1000) : undefined;
      if (asOfEpoch !== undefined && Number.isNaN(asOfEpoch)) {
        throw new Error(`asOf is not an ISO-8601 date: ${asOf}`);
      }
      const facts = await client.traverse(seeds, {
        kHops: hops ?? options.kHops ?? 3,
        asOf: asOfEpoch,
        embedding,
        topK: options.topK ?? 12,
      });
      return { facts: facts.map(renderFact) };
    },
  });
}

function messageText(message: ModelMessage): string {
  if (typeof message.content === "string") return message.content;
  const parts: string[] = [];
  for (const part of message.content) {
    if (part.type === "text") parts.push(part.text);
    else if (part.type === "tool-result") parts.push(JSON.stringify(part.output));
  }
  return parts.join("\n");
}

function messageRole(message: ModelMessage): Role {
  const role = message.role;
  switch (role) {
    case "user":
    case "assistant":
    case "system":
    case "tool":
      return role;
    default: {
      const unhandled: never = role;
      throw new Error(`unknown message role ${String(unhandled)}`);
    }
  }
}

/**
 * Persists messages to the session, e.g. from `streamText`'s `onFinish`.
 * Only text (and tool results as JSON) is stored.
 */
export async function saveMessages(
  client: HypergraphClient,
  sessionId: string,
  messages: ModelMessage[],
  options: { embed?: Embed; entities?: string[] } = {},
): Promise<void> {
  const records: ChatRecord[] = messages.map((message) => ({
    role: messageRole(message),
    content: messageText(message),
    entities: options.entities,
  }));
  await new SessionMemory(client, sessionId, { embed: options.embed }).add(records);
}

/** The stored session as `ModelMessage`s (oldest first), for `streamText({ messages })`. */
export async function loadMessages(
  client: HypergraphClient,
  sessionId: string,
  limit = 0,
): Promise<ModelMessage[]> {
  const history = await new SessionMemory(client, sessionId).history(limit);
  return history.flatMap((record): ModelMessage[] => {
    switch (record.role) {
      case "user":
        return [{ role: "user", content: record.content }];
      case "assistant":
        return [{ role: "assistant", content: record.content }];
      case "system":
        return [{ role: "system", content: record.content }];
      case "tool":
        // Tool results need their originating call id, which is not stored.
        return [];
      default: {
        const unhandled: never = record.role;
        throw new Error(`unknown role ${String(unhandled)}`);
      }
    }
  });
}
