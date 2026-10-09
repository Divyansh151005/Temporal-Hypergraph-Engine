# @hypergraph/sdk

TypeScript client and Vercel AI SDK memory for the Temporal Hypergraph Engine.

```bash
npm install @hypergraph/sdk ai zod
```

```ts
import { TemporalHypergraph } from "@hypergraph/sdk";

const memory = new TemporalHypergraph({ apiKey: "your_key" }); // session "default"; pass sessionId to change
```

```ts
import { HypergraphClient } from "@hypergraph/sdk";
import { hypergraphMemoryMiddleware, hypergraphContextTool, saveMessages } from "@hypergraph/sdk/ai";
import { streamText, wrapLanguageModel } from "ai";

const client = new HypergraphClient({ apiKey: process.env.HYPERGRAPH_API_KEY });

const result = streamText({
  model: wrapLanguageModel({
    model: openai("gpt-4o-mini"),
    middleware: hypergraphMemoryMiddleware({ client, sessionId }),
  }),
  messages,
  tools: { recall: hypergraphContextTool(client, { sessionId }) },
  onFinish: ({ response }) => saveMessages(client, sessionId, [userMessage, ...response.messages]),
});
```

- `hypergraphMemoryMiddleware` injects facts reachable from the session (and
  optional `seeds`) within `kHops`, valid now, as a system message. Pass
  `embed` to rank them by similarity to the latest user message.
- `hypergraphContextTool` lets the model run multi-hop, point-in-time lookups
  (`entities`, `question`, `asOf` as ISO-8601, `hops`).
- `saveMessages` / `loadMessages` persist and restore a session's text
  messages. Sessions use the same storage scheme as the Python
  `langchain-hypergraph` and `llama-index-hypergraph` packages.

## Client

`HypergraphClient({ endpoint, apiKey, timeoutMs, tls })`: `insert`,
`insertBatch`, `query`, `streamContext` (an `AsyncIterable`), `traverse`,
`delete`, `health`. `endpoint` defaults to `$HYPERGRAPH_ENDPOINT` or
`localhost:50051` and accepts `host:port`, `tcp://host:port` and
`unix:///path`; `apiKey` defaults to `$HYPERGRAPH_API_KEY`.

The client speaks gRPC through `@grpc/grpc-js`, so it runs on Node.js (18+).
The Edge runtime is not supported: it has no HTTP/2 client and the server has
no REST gateway. The `ai` subpath requires `ai` 7, which needs Node.js 22+.

## Development

```bash
npm ci
npm run gen         # regenerate src/generated from proto/ (needs grpcio-tools)
npm run build       # ESM + CJS + .d.ts into dist/
npx tsc --noEmit
npm test            # live tests run when target/{release,debug}/hypergraph-server exists
```
