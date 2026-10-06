import {
  type HyperedgePayload,
  type InsertHyperedgeRequest,
  type QueryTemporalRequest,
  type QueryTemporalResponse,
  TraversalOrder as WireOrder,
} from "./generated/hypergraph_service.js";
import { GrpcTransport, type GrpcTransportOptions, type Transport } from "./transport.js";
import {
  type BatchResult,
  type Hyperedge,
  type QueryOptions,
  type ScoredEdge,
  type TraversalOrder,
  type TraverseOptions,
  nowEpoch,
} from "./types.js";

export interface HypergraphClientOptions extends GrpcTransportOptions {
  /** Overrides the gRPC transport (e.g. a fake in tests). */
  transport?: Transport;
}

export function toF32Bytes(vector: ArrayLike<number> | undefined): Buffer {
  if (!vector || vector.length === 0) return Buffer.alloc(0);
  const bytes = Buffer.alloc(vector.length * 4);
  for (let i = 0; i < vector.length; i++) bytes.writeFloatLE(vector[i] ?? 0, i * 4);
  return bytes;
}

export function fromF32Bytes(raw: Uint8Array | undefined): Float32Array | undefined {
  if (!raw || raw.length === 0) return undefined;
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Float32Array(raw.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

function wireOrder(order: TraversalOrder): WireOrder {
  switch (order) {
    case "relevance":
      return WireOrder.TRAVERSAL_ORDER_RELEVANCE;
    case "chronological":
      return WireOrder.TRAVERSAL_ORDER_CHRONOLOGICAL;
    default: {
      const unhandled: never = order;
      throw new Error(`unknown traversal order ${String(unhandled)}`);
    }
  }
}

export function toPayload(edge: Hyperedge): HyperedgePayload {
  return {
    id: "",
    eventType: edge.eventType,
    summary: edge.summary ?? "",
    participants: edge.participants.map((entityId) => ({ entityId, role: "" })),
    temporalBoundary: {
      validFromEpoch: edge.validFrom,
      validToEpoch: edge.validTo,
      isOngoing: edge.validTo === undefined,
      rawExpression: "",
    },
    confidenceScore: edge.confidence ?? 1,
    embeddingRaw: toF32Bytes(edge.embedding),
  };
}

export function fromPayload(payload: HyperedgePayload, score = 0): ScoredEdge {
  const boundary = payload.temporalBoundary;
  return {
    id: payload.id,
    eventType: payload.eventType,
    participants: payload.participants.map((p) => p.entityId),
    validFrom: boundary?.validFromEpoch,
    validTo: boundary?.validToEpoch,
    isOngoing: boundary?.isOngoing ?? false,
    summary: payload.summary,
    confidence: payload.confidenceScore,
    score,
    embedding: fromF32Bytes(payload.embeddingRaw),
  };
}

function scored(response: QueryTemporalResponse): ScoredEdge[] {
  return response.matchedEdges.map((edge, i) => fromPayload(edge, response.decayScores[i] ?? 0));
}

/**
 * Client for `hypergraph.v1.HypergraphEngineService`.
 *
 * ```ts
 * const client = new HypergraphClient({ endpoint: "localhost:50051", apiKey: "..." });
 * const facts = await client.traverse(["org:acme"], { kHops: 3 });
 * ```
 */
export class HypergraphClient {
  readonly transport: Transport;

  constructor(options: HypergraphClientOptions = {}) {
    this.transport = options.transport ?? new GrpcTransport(options);
  }

  private insertRequest(edge: Hyperedge, sessionId: string): InsertHyperedgeRequest {
    return { sessionId, hyperedge: toPayload(edge) };
  }

  private queryRequest(embedding: ArrayLike<number>, options: QueryOptions): QueryTemporalRequest {
    return {
      sessionId: "",
      queryEmbeddingRaw: toF32Bytes(embedding),
      targetTimestampEpoch: options.asOf ?? nowEpoch(),
      decayLambda: options.decayLambda ?? 0,
      topK: options.topK ?? 10,
    };
  }

  /** Inserts one hyperedge and returns its id. */
  async insert(edge: Hyperedge, sessionId = ""): Promise<string> {
    return (await this.transport.insert(this.insertRequest(edge, sessionId))).hyperedgeId;
  }

  /** Streams hyperedges in one call; the server group-commits them. */
  async insertBatch(edges: Hyperedge[], sessionId = ""): Promise<BatchResult> {
    const response = await this.transport.insertBatch(
      edges.map((edge) => this.insertRequest(edge, sessionId)),
    );
    return { processed: response.processedCount, errors: response.errors, ok: response.errors.length === 0 };
  }

  /** Global similarity search ranked by `cosine * confidence * decay`. */
  async query(embedding: ArrayLike<number>, options: QueryOptions = {}): Promise<ScoredEdge[]> {
    return scored(await this.transport.query(this.queryRequest(embedding, options)));
  }

  /** Like {@link query}, streamed edge by edge (without scores). */
  async *streamContext(
    embedding: ArrayLike<number>,
    options: QueryOptions = {},
  ): AsyncIterable<ScoredEdge> {
    for await (const payload of this.transport.stream(this.queryRequest(embedding, options))) {
      yield fromPayload(payload);
    }
  }

  /** Hyperedges within `kHops` of `seeds`, valid at `asOf`. */
  async traverse(seeds: string[], options: TraverseOptions = {}): Promise<ScoredEdge[]> {
    const response = await this.transport.traverse({
      seedEntityIds: seeds,
      kHops: options.kHops ?? 1,
      asOfEpoch: options.asOf ?? 0,
      queryEmbeddingRaw: toF32Bytes(options.embedding),
      decayLambda: options.decayLambda ?? 0,
      topK: options.topK ?? 10,
      order: wireOrder(options.order ?? "relevance"),
      eventTypes: options.eventTypes ?? [],
    });
    return scored(response);
  }

  /** Soft-deletes hyperedges; returns how many were live. */
  async delete(ids: string[]): Promise<number> {
    if (ids.length === 0) return 0;
    return (await this.transport.delete({ hyperedgeIds: ids })).deletedCount;
  }

  async health(): Promise<boolean> {
    return (await this.transport.health()).healthy;
  }

  close(): void {
    this.transport.close();
  }
}
