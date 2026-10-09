import { status } from "@grpc/grpc-js";
import { describe, expect, it } from "vitest";

import { TraversalOrder } from "../src/generated/hypergraph_service.js";
import {
  AuthenticationError,
  DEFAULT_SESSION,
  HypergraphClient,
  HypergraphError,
  SessionMemory,
  TemporalHypergraph,
  entitiesOf,
  fromF32Bytes,
  recordNode,
  resolveEndpoint,
  sessionNode,
  toF32Bytes,
} from "../src/index.js";
import { toHypergraphError } from "../src/transport.js";
import { FakeTransport } from "./fake-transport.js";

describe("helpers", () => {
  it("packs float32 little-endian", () => {
    const bytes = toF32Bytes([1, -0.5]);
    expect([...bytes]).toEqual([0, 0, 128, 63, 0, 0, 0, 191]);
    expect([...fromF32Bytes(bytes)!]).toEqual([1, -0.5]);
    expect(toF32Bytes(undefined).length).toBe(0);
    expect(fromF32Bytes(new Uint8Array())).toBeUndefined();
  });

  it("builds session and record nodes", () => {
    expect(sessionNode("42")).toBe("session:42");
    expect(() => sessionNode("")).toThrow();
    const ids = Array.from({ length: 50 }, recordNode);
    expect([...ids].sort()).toEqual(ids);
    expect(ids[0]).toMatch(/^record:[0-9a-f]{16}-[0-9a-f]{12}$/);
    expect(entitiesOf(["session:1", ids[0]!, "org:acme"])).toEqual(["org:acme"]);
  });

  it("resolves endpoints", () => {
    expect(resolveEndpoint("tcp://h:1")).toBe("h:1");
    expect(resolveEndpoint("unix:///tmp/x.sock")).toBe("unix:///tmp/x.sock");
  });

  it("maps gRPC errors", () => {
    expect(toHypergraphError({ code: status.UNAUTHENTICATED, details: "bad key" })).toBeInstanceOf(
      AuthenticationError,
    );
    const other = toHypergraphError({ code: status.INVALID_ARGUMENT, details: "k_hops" });
    expect(other).toBeInstanceOf(HypergraphError);
    expect(other).not.toBeInstanceOf(AuthenticationError);
    expect((other as HypergraphError).code).toBe(status.INVALID_ARGUMENT);
  });
});

describe("HypergraphClient over a mock transport", () => {
  it("builds requests and maps responses", async () => {
    const transport = new FakeTransport();
    const client = new HypergraphClient({ transport });
    const id = await client.insert({
      eventType: "LINK",
      participants: ["a", "b"],
      validFrom: 100,
      validTo: 200,
      summary: "a->b",
      embedding: [1, 0],
    });
    const stored = transport.edges.get(id)!;
    expect(stored.temporalBoundary).toMatchObject({ validFromEpoch: 100, validToEpoch: 200, isOngoing: false });
    expect(stored.confidenceScore).toBe(1);

    const [hit] = await client.traverse(["a"], { asOf: 150, order: "chronological", embedding: [1, 0] });
    expect(hit).toMatchObject({ id, summary: "a->b", participants: ["a", "b"], validFrom: 100, validTo: 200 });
    expect([...hit!.embedding!]).toEqual([1, 0]);
    const request = transport.calls.find((c) => c.method === "traverse")!.request as { order: number; kHops: number; topK: number };
    expect(request).toMatchObject({ order: TraversalOrder.TRAVERSAL_ORDER_CHRONOLOGICAL, kHops: 1, topK: 10 });

    expect(await client.traverse(["a"], { asOf: 250 })).toEqual([]);
    expect((await client.query([1, 0], { asOf: 150 })).map((e) => e.id)).toEqual([id]);
    const streamed = [];
    for await (const edge of client.streamContext([1, 0], { asOf: 150 })) streamed.push(edge.id);
    expect(streamed).toEqual([id]);

    expect(await client.delete([])).toBe(0);
    expect(await client.delete([id, id])).toBe(1);
    expect(await client.health()).toBe(true);
  });

  it("stores and reads a session in order", async () => {
    const client = new HypergraphClient({ transport: new FakeTransport() });
    const memory = new SessionMemory(client, "s1");
    await memory.add([
      { role: "user", content: "hi", entities: ["order:7"] },
      { role: "assistant", content: "hello" },
      { role: "user", content: "" },
    ]);
    await new SessionMemory(client, "s2").add([{ role: "user", content: "other" }]);
    const history = await memory.history();
    expect(history.map((r) => [r.role, r.content])).toEqual([
      ["user", "hi"],
      ["assistant", "hello"],
    ]);
    expect(history[0]!.entities).toEqual(["order:7"]);
    expect(await memory.clear()).toBe(2);
    expect(await memory.history()).toEqual([]);
  });

  it("TemporalHypergraph is a one-line session memory", async () => {
    const transport = new FakeTransport();
    const memory = new TemporalHypergraph({ apiKey: "k", transport });
    expect(memory.sessionId).toBe(DEFAULT_SESSION);
    expect(memory.node).toBe(sessionNode(DEFAULT_SESSION));
    await memory.add([{ role: "user", content: "remember me" }]);
    const shared = new SessionMemory(new HypergraphClient({ transport }), DEFAULT_SESSION);
    expect((await shared.history()).map((r) => r.content)).toEqual(["remember me"]);
    expect(new TemporalHypergraph({ transport, sessionId: "s9" }).node).toBe("session:s9");
  });
});
