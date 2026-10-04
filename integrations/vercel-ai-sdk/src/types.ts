/** Domain types of the SDK. Times are epoch seconds. */

/** A hyperedge to insert. */
export interface Hyperedge {
  eventType: string;
  participants: string[];
  validFrom: number;
  /** End of validity (exclusive); omit for an ongoing fact. */
  validTo?: number;
  summary?: string;
  /** In `[0, 1]`; scales retrieval scores. Defaults to 1. */
  confidence?: number;
  embedding?: ArrayLike<number>;
}

/** A hyperedge returned by a query or traversal. */
export interface ScoredEdge {
  id: string;
  eventType: string;
  participants: string[];
  validFrom?: number;
  validTo?: number;
  isOngoing: boolean;
  summary: string;
  confidence: number;
  /** `confidence * decay`, times cosine similarity when an embedding was given. */
  score: number;
  embedding?: Float32Array;
}

export type TraversalOrder = "relevance" | "chronological";

export interface TraverseOptions {
  /** Traversal depth, 1 to 4 (default 1). */
  kHops?: number;
  /** Only hyperedges valid at this instant (default now). */
  asOf?: number;
  /** Re-ranks results by cosine similarity to this vector. */
  embedding?: ArrayLike<number>;
  /** Results to return; 0 returns all (default 10). */
  topK?: number;
  /** `relevance` (default) or `chronological` (newest first). */
  order?: TraversalOrder;
  /** Keep only these event types. */
  eventTypes?: string[];
  decayLambda?: number;
}

export interface QueryOptions {
  asOf?: number;
  topK?: number;
  decayLambda?: number;
}

export interface BatchResult {
  processed: number;
  errors: string[];
  ok: boolean;
}

/** A failed engine call; `code` is the gRPC status code. */
export class HypergraphError extends Error {
  constructor(
    readonly code: number,
    readonly details: string,
  ) {
    super(`gRPC status ${code}: ${details}`);
    this.name = "HypergraphError";
  }
}

/** The server rejected the API key. */
export class AuthenticationError extends HypergraphError {
  constructor(code: number, details: string) {
    super(code, details);
    this.name = "AuthenticationError";
  }
}

export const nowEpoch = (): number => Math.floor(Date.now() / 1000);
