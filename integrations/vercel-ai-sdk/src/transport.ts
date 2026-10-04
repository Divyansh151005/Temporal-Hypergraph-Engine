import {
  type CallOptions,
  credentials,
  Metadata,
  type ServiceError,
  status,
} from "@grpc/grpc-js";

import {
  type BatchResponse,
  type DeleteHyperedgesRequest,
  type DeleteHyperedgesResponse,
  type HealthCheckResponse,
  type HyperedgePayload,
  HypergraphEngineServiceClient,
  type InsertHyperedgeRequest,
  type InsertHyperedgeResponse,
  type QueryTemporalRequest,
  type QueryTemporalResponse,
  type TraverseTemporalRequest,
} from "./generated/hypergraph_service.js";
import { AuthenticationError, HypergraphError } from "./types.js";

/** Wire-level operations the client needs; swap in a fake for tests. */
export interface Transport {
  insert(request: InsertHyperedgeRequest): Promise<InsertHyperedgeResponse>;
  insertBatch(requests: InsertHyperedgeRequest[]): Promise<BatchResponse>;
  query(request: QueryTemporalRequest): Promise<QueryTemporalResponse>;
  stream(request: QueryTemporalRequest): AsyncIterable<HyperedgePayload>;
  traverse(request: TraverseTemporalRequest): Promise<QueryTemporalResponse>;
  delete(request: DeleteHyperedgesRequest): Promise<DeleteHyperedgesResponse>;
  health(): Promise<HealthCheckResponse>;
  close(): void;
}

export const DEFAULT_ENDPOINT = "localhost:50051";
export const API_KEY_HEADER = "x-api-key";
const MAX_MESSAGE_BYTES = 16 * 1024 * 1024;

export interface GrpcTransportOptions {
  /** `host:port`, `tcp://host:port` or `unix:///path` (default `$HYPERGRAPH_ENDPOINT` or `localhost:50051`). */
  endpoint?: string;
  /** Sent as `x-api-key` (default `$HYPERGRAPH_API_KEY`). */
  apiKey?: string;
  /** Per-call deadline (default 5000). */
  timeoutMs?: number;
  /** Use TLS with the system roots instead of plaintext. */
  tls?: boolean;
}

export function resolveEndpoint(endpoint?: string): string {
  const target = endpoint || process.env.HYPERGRAPH_ENDPOINT || DEFAULT_ENDPOINT;
  return target.startsWith("tcp://") ? target.slice("tcp://".length) : target;
}

export function toHypergraphError(error: unknown): Error {
  const service = error as Partial<ServiceError>;
  if (typeof service.code !== "number") {
    return error instanceof Error ? error : new Error(String(error));
  }
  const details = service.details ?? service.message ?? "";
  return service.code === status.UNAUTHENTICATED
    ? new AuthenticationError(service.code, details)
    : new HypergraphError(service.code, details);
}

type Unary<Req, Res> = (
  request: Req,
  metadata: Metadata,
  options: Partial<CallOptions>,
  callback: (error: ServiceError | null, response: Res) => void,
) => unknown;

/** `@grpc/grpc-js` transport (Node.js only; the Edge runtime has no HTTP/2 client). */
export class GrpcTransport implements Transport {
  readonly endpoint: string;
  private readonly client: HypergraphEngineServiceClient;
  private readonly metadata = new Metadata();
  private readonly timeoutMs: number;

  constructor(options: GrpcTransportOptions = {}) {
    this.endpoint = resolveEndpoint(options.endpoint);
    this.timeoutMs = options.timeoutMs ?? 5000;
    const apiKey = options.apiKey || process.env.HYPERGRAPH_API_KEY;
    if (apiKey) this.metadata.set(API_KEY_HEADER, apiKey);
    const channelOptions: Record<string, string | number> = {
      "grpc.max_send_message_length": MAX_MESSAGE_BYTES,
      "grpc.max_receive_message_length": MAX_MESSAGE_BYTES,
    };
    if (this.endpoint.startsWith("unix:")) channelOptions["grpc.default_authority"] = "localhost";
    this.client = new HypergraphEngineServiceClient(
      this.endpoint,
      options.tls ? credentials.createSsl() : credentials.createInsecure(),
      channelOptions,
    );
  }

  private options(): Partial<CallOptions> {
    return { deadline: Date.now() + this.timeoutMs };
  }

  private unary<Req, Res>(method: Unary<Req, Res>, request: Req): Promise<Res> {
    return new Promise((resolve, reject) => {
      method.call(this.client, request, this.metadata, this.options(), (error, response) =>
        error ? reject(toHypergraphError(error)) : resolve(response),
      );
    });
  }

  insert(request: InsertHyperedgeRequest): Promise<InsertHyperedgeResponse> {
    return this.unary(this.client.insertHyperedge, request);
  }

  insertBatch(requests: InsertHyperedgeRequest[]): Promise<BatchResponse> {
    return new Promise((resolve, reject) => {
      const call = this.client.insertHyperedgeBatch(this.metadata, this.options(), (error, response) =>
        error ? reject(toHypergraphError(error)) : resolve(response),
      );
      for (const request of requests) call.write(request);
      call.end();
    });
  }

  query(request: QueryTemporalRequest): Promise<QueryTemporalResponse> {
    return this.unary(this.client.queryTemporalContext, request);
  }

  async *stream(request: QueryTemporalRequest): AsyncIterable<HyperedgePayload> {
    const call = this.client.streamTemporalContext(request, this.metadata, this.options());
    try {
      for await (const payload of call) yield payload as HyperedgePayload;
    } catch (error) {
      throw toHypergraphError(error);
    } finally {
      call.cancel();
    }
  }

  traverse(request: TraverseTemporalRequest): Promise<QueryTemporalResponse> {
    return this.unary(this.client.traverseTemporalContext, request);
  }

  delete(request: DeleteHyperedgesRequest): Promise<DeleteHyperedgesResponse> {
    return this.unary(this.client.deleteHyperedges, request);
  }

  health(): Promise<HealthCheckResponse> {
    return this.unary(this.client.healthCheck, {});
  }

  close(): void {
    this.client.close();
  }
}
