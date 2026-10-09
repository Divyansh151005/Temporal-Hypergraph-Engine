import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { AuthenticationError, HypergraphClient, SessionMemory, recordNode } from "../src/index.js";

const ROOT = resolve(import.meta.dirname, "../../..");

function serverBinary(): string | undefined {
  const explicit = process.env.HYPERGRAPH_SERVER_BIN;
  if (explicit) return existsSync(explicit) ? explicit : undefined;
  const builds = ["release", "debug"]
    .map((profile) => join(ROOT, "target", profile, "hypergraph-server"))
    .filter((path) => existsSync(path));
  return builds.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)[0];
}

function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.once("error", fail);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      probe.close(() => done(typeof address === "object" && address ? address.port : 0));
    });
  });
}

async function startServer(binary: string, apiKey?: string) {
  const port = await freePort();
  const dataDir = mkdtempSync(join(tmpdir(), "hg-ts-"));
  const args = ["--data-dir", dataDir, "--listen", `tcp://127.0.0.1:${port}`, "--sync", "buffered"];
  const env = { ...process.env };
  delete env.HYPERGRAPH_API_KEY;
  const child = spawn(binary, apiKey ? [...args, "--api-key", apiKey] : args, { env, stdio: "ignore" });
  const endpoint = `127.0.0.1:${port}`;
  const probe = new HypergraphClient({ endpoint, apiKey, timeoutMs: 500 });
  for (let attempt = 0; ; attempt++) {
    try {
      if (await probe.health()) break;
    } catch (error) {
      if (attempt > 100 || child.exitCode !== null) throw error;
      await new Promise((r) => setTimeout(r, 50));
    }
  }
  probe.close();
  return {
    endpoint,
    stop: async () => {
      if (child.exitCode === null) {
        const exited = new Promise((r) => child.once("exit", r));
        child.kill("SIGINT");
        await exited;
      }
      rmSync(dataDir, { recursive: true, force: true });
    },
  };
}

const binary = serverBinary();
const unique = (prefix: string) => `${prefix}:${Math.random().toString(16).slice(2, 10)}`;

describe.skipIf(!binary)("against hypergraph-server", () => {
  let server: Awaited<ReturnType<typeof startServer>>;
  let client: HypergraphClient;

  beforeAll(async () => {
    server = await startServer(binary!);
    client = new HypergraphClient({ endpoint: `tcp://${server.endpoint}` });
  });

  afterAll(async () => {
    client?.close();
    await server?.stop();
  });

  it("inserts, traverses with time travel, streams and deletes", async () => {
    const [acme, beta, alice, bob] = [unique("org"), unique("org"), unique("person"), unique("person")];
    const acquired = await client.insert({
      eventType: "ACQUIRED",
      participants: [acme, beta, recordNode()],
      validFrom: 50,
      summary: "Acme acquired Beta",
      embedding: [1, 0],
    });
    const old = await client.insert({
      eventType: "CEO",
      participants: [beta, alice, recordNode()],
      validFrom: 100,
      validTo: 500,
      summary: "Alice was CEO of Beta",
      embedding: [0, 1],
    });
    const current = await client.insert({
      eventType: "CEO",
      participants: [beta, bob, recordNode()],
      validFrom: 500,
      summary: "Bob is CEO of Beta",
      embedding: [0.6, 0.8],
    });

    const at300 = await client.traverse([acme], { kHops: 2, asOf: 300, topK: 0 });
    expect(new Set(at300.map((e) => e.id))).toEqual(new Set([acquired, old]));
    const at600 = await client.traverse([acme], { kHops: 2, asOf: 600, embedding: [0, 1], topK: 1 });
    expect(at600.map((e) => e.summary)).toEqual(["Bob is CEO of Beta"]);
    expect(at600[0]!.isOngoing).toBe(true);

    const streamed = [];
    for await (const edge of client.streamContext([1, 0], { asOf: 600, topK: 2 })) streamed.push(edge);
    expect(streamed.length).toBeGreaterThan(0);

    const batch = await client.insertBatch([
      { eventType: "E", participants: [unique("x")], validFrom: 1 },
      { eventType: "E", participants: [unique("y")], validFrom: 10, validTo: 5 },
    ]);
    expect(batch.processed).toBe(2);
    expect(batch.errors).toHaveLength(1);

    expect(await client.delete([current, current])).toBe(1);
    expect(await client.traverse([bob], { asOf: 600 })).toEqual([]);
    await expect(client.traverse([acme], { kHops: 9 })).rejects.toThrow(/k_hops/);
  });

  it("shares session storage with the Python integrations' scheme", async () => {
    const memory = new SessionMemory(client, unique("s").replace(":", "-"));
    await memory.add([
      { role: "user", content: "one" },
      { role: "assistant", content: "two" },
      { role: "user", content: "three" },
    ]);
    expect((await memory.history()).map((r) => r.content)).toEqual(["one", "two", "three"]);
    expect((await memory.history(1)).map((r) => r.content)).toEqual(["three"]);
    expect(await memory.clear()).toBe(3);
  });

  it("enforces the API key", async () => {
    const secured = await startServer(binary!, "s3cret");
    try {
      const good = new HypergraphClient({ endpoint: secured.endpoint, apiKey: "s3cret" });
      expect(await good.health()).toBe(true);
      good.close();
      const bad = new HypergraphClient({ endpoint: secured.endpoint, apiKey: "wrong" });
      await expect(bad.health()).rejects.toBeInstanceOf(AuthenticationError);
      bad.close();
    } finally {
      await secured.stop();
    }
  });
});
