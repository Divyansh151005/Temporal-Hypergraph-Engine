import { generateText, wrapLanguageModel } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it } from "vitest";

import {
  CONTEXT_HEADER,
  hypergraphContextTool,
  hypergraphMemoryMiddleware,
  loadMessages,
  renderFact,
  saveMessages,
} from "../src/ai-sdk.js";
import { HypergraphClient, recordNode, sessionNode } from "../src/index.js";
import { FakeTransport } from "./fake-transport.js";

const reply = (text: string) => ({
  content: [{ type: "text" as const, text }],
  finishReason: { unified: "stop" as const, raw: undefined },
  usage: {
    inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: { total: 1, text: 1, reasoning: undefined },
  },
  warnings: [],
});

async function seeded() {
  const client = new HypergraphClient({ transport: new FakeTransport() });
  await client.insert({
    eventType: "PURCHASE",
    participants: [sessionNode("s1"), "customer:ada", "product:x1", recordNode()],
    validFrom: 1_700_000_000,
    summary: "Ada bought an X1 laptop",
  });
  await client.insert({
    eventType: "WARRANTY",
    participants: ["product:x1", "plan:gold"],
    validFrom: 1_700_000_000,
    summary: "X1 laptops carry the gold warranty",
  });
  return client;
}

describe("hypergraphMemoryMiddleware", () => {
  it("injects multi-hop context as a system message", async () => {
    const client = await seeded();
    const model = new MockLanguageModelV4({ doGenerate: reply("covered") });
    const { text } = await generateText({
      model: wrapLanguageModel({ model, middleware: hypergraphMemoryMiddleware({ client, sessionId: "s1" }) }),
      system: "You are support.",
      prompt: "Is my laptop under warranty?",
    });
    expect(text).toBe("covered");
    const prompt = model.doGenerateCalls[0]!.prompt;
    expect(prompt.map((m) => m.role)).toEqual(["system", "system", "user"]);
    const context = prompt[1]!.content as string;
    expect(context.startsWith(CONTEXT_HEADER)).toBe(true);
    expect(context).toContain("Ada bought an X1 laptop");
    expect(context).toContain("X1 laptops carry the gold warranty");
  });

  it("leaves the prompt alone when nothing is known", async () => {
    const client = new HypergraphClient({ transport: new FakeTransport() });
    const model = new MockLanguageModelV4({ doGenerate: reply("hi") });
    await generateText({
      model: wrapLanguageModel({ model, middleware: hypergraphMemoryMiddleware({ client, sessionId: "empty" }) }),
      prompt: "hello",
    });
    expect(model.doGenerateCalls[0]!.prompt.map((m) => m.role)).toEqual(["user"]);
  });
});

describe("hypergraphContextTool", () => {
  it("returns facts valid at the requested date", async () => {
    const client = await seeded();
    const recall = hypergraphContextTool(client);
    const options = { toolCallId: "1", messages: [], context: {} } as never;
    const now = await recall.execute!({ entities: ["customer:ada"], hops: 2 }, options);
    expect((now as { facts: string[] }).facts).toHaveLength(2);
    expect((now as { facts: string[] }).facts[0]).toMatch(/^\[2023-11-14 → now\] /);
    const before = await recall.execute!({ entities: ["customer:ada"], asOf: "2020-01-01" }, options);
    expect((before as { facts: string[] }).facts).toEqual([]);
    await expect(recall.execute!({ entities: ["x"], asOf: "not a date" }, options)).rejects.toThrow(/ISO-8601/);
  });
});

describe("saveMessages / loadMessages", () => {
  it("round-trips text messages", async () => {
    const client = new HypergraphClient({ transport: new FakeTransport() });
    await saveMessages(client, "s9", [
      { role: "user", content: [{ type: "text", text: "where is order 7?" }] },
      { role: "assistant", content: "It ships today." },
    ]);
    expect(await loadMessages(client, "s9")).toEqual([
      { role: "user", content: "where is order 7?" },
      { role: "assistant", content: "It ships today." },
    ]);
  });

  it("renders facts compactly", () => {
    expect(
      renderFact({
        id: "1",
        eventType: "CEO",
        participants: ["session:s", "person:bob", "org:beta"],
        validFrom: 0,
        validTo: 86_400,
        isOngoing: false,
        summary: "Bob ran Beta",
        confidence: 1,
        score: 1,
      }),
    ).toBe("[1970-01-01 → 1970-01-02] CEO(person:bob, org:beta): Bob ran Beta");
  });
});
