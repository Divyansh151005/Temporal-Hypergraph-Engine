import type {
  BatchResponse,
  DeleteHyperedgesRequest,
  DeleteHyperedgesResponse,
  HealthCheckResponse,
  HyperedgePayload,
  InsertHyperedgeRequest,
  InsertHyperedgeResponse,
  QueryTemporalRequest,
  QueryTemporalResponse,
  TraverseTemporalRequest,
} from "../src/generated/hypergraph_service.js";
import { TraversalOrder } from "../src/generated/hypergraph_service.js";
import { fromF32Bytes } from "../src/client.js";
import type { Transport } from "../src/transport.js";

/** In-memory transport mirroring the engine's traversal semantics (no conflict resolution or decay). */
export class FakeTransport implements Transport {
  readonly edges = new Map<string, HyperedgePayload>();
  readonly calls: { method: string; request: unknown }[] = [];
  private next = 1;

  private valid(edge: HyperedgePayload, asOf: number): boolean {
    const b = edge.temporalBoundary;
    const from = b?.validFromEpoch ?? 0;
    return from <= asOf && (b?.validToEpoch === undefined || asOf < b.validToEpoch);
  }

  private score(edge: HyperedgePayload, query: Uint8Array): number {
    const q = fromF32Bytes(query);
    const e = fromF32Bytes(edge.embeddingRaw);
    if (!q) return edge.confidenceScore;
    if (!e || e.length !== q.length) return 0;
    let dot = 0;
    let nq = 0;
    let ne = 0;
    for (let i = 0; i < q.length; i++) {
      dot += (q[i] ?? 0) * (e[i] ?? 0);
      nq += (q[i] ?? 0) ** 2;
      ne += (e[i] ?? 0) ** 2;
    }
    return nq && ne ? (edge.confidenceScore * dot) / Math.sqrt(nq * ne) : 0;
  }

  async insert(request: InsertHyperedgeRequest): Promise<InsertHyperedgeResponse> {
    this.calls.push({ method: "insert", request });
    const id = `fake-${this.next++}`;
    this.edges.set(id, { ...request.hyperedge!, id });
    return { success: true, hyperedgeId: id, sequenceId: this.next };
  }

  async insertBatch(requests: InsertHyperedgeRequest[]): Promise<BatchResponse> {
    this.calls.push({ method: "insertBatch", request: requests });
    for (const request of requests) {
      const id = `fake-${this.next++}`;
      this.edges.set(id, { ...request.hyperedge!, id });
    }
    return { processedCount: requests.length, allSucceeded: true, errors: [] };
  }

  async query(request: QueryTemporalRequest): Promise<QueryTemporalResponse> {
    this.calls.push({ method: "query", request });
    const hits = [...this.edges.values()]
      .filter((e) => this.valid(e, request.targetTimestampEpoch))
      .map((e) => ({ e, s: this.score(e, request.queryEmbeddingRaw) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, request.topK);
    return { matchedEdges: hits.map((h) => h.e), decayScores: hits.map((h) => h.s), queryLatencyUs: 1 };
  }

  async *stream(request: QueryTemporalRequest): AsyncIterable<HyperedgePayload> {
    yield* (await this.query(request)).matchedEdges;
  }

  async traverse(request: TraverseTemporalRequest): Promise<QueryTemporalResponse> {
    this.calls.push({ method: "traverse", request });
    const asOf = request.asOfEpoch || Math.floor(Date.now() / 1000);
    const live = [...this.edges.values()].filter((e) => this.valid(e, asOf));
    let frontier = new Set(request.seedEntityIds);
    const seen = new Set(frontier);
    const found: HyperedgePayload[] = [];
    for (let hop = 0; hop < Math.max(request.kHops, 1); hop++) {
      const reached = live.filter(
        (e) => !found.includes(e) && e.participants.some((p) => frontier.has(p.entityId)),
      );
      found.push(...reached);
      frontier = new Set(reached.flatMap((e) => e.participants.map((p) => p.entityId)).filter((p) => !seen.has(p)));
      frontier.forEach((p) => seen.add(p));
    }
    const typed = request.eventTypes.length
      ? found.filter((e) => request.eventTypes.includes(e.eventType))
      : found;
    const scored = typed.map((e) => ({ e, s: this.score(e, request.queryEmbeddingRaw) }));
    if (request.order === TraversalOrder.TRAVERSAL_ORDER_CHRONOLOGICAL) {
      scored.sort((a, b) => (b.e.temporalBoundary?.validFromEpoch ?? 0) - (a.e.temporalBoundary?.validFromEpoch ?? 0));
    } else {
      scored.sort((a, b) => b.s - a.s);
    }
    const top = request.topK ? scored.slice(0, request.topK) : scored;
    return { matchedEdges: top.map((h) => h.e), decayScores: top.map((h) => h.s), queryLatencyUs: 1 };
  }

  async delete(request: DeleteHyperedgesRequest): Promise<DeleteHyperedgesResponse> {
    this.calls.push({ method: "delete", request });
    let deletedCount = 0;
    for (const id of new Set(request.hyperedgeIds)) if (this.edges.delete(id)) deletedCount++;
    return { deletedCount };
  }

  async health(): Promise<HealthCheckResponse> {
    return { healthy: true, uptimeSeconds: 1, activeConnections: 0 };
  }

  close(): void {}
}
