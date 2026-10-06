import { HypergraphClient, type HypergraphClientOptions } from "./client.js";
import {
  MESSAGE_EVENT_TYPES,
  type Role,
  entitiesOf,
  eventTypeRole,
  recordNode,
  recordOf,
  roleEventType,
  sessionNode,
} from "./session.js";
import { type ScoredEdge, nowEpoch } from "./types.js";

export type Embed = (text: string) => Promise<ArrayLike<number>>;

export interface ChatRecord {
  role: Role;
  content: string;
  entities?: string[];
  timestamp?: number;
  id?: string;
}

export interface SessionMemoryOptions {
  /** Embeds message text on write and queries on {@link SessionMemory.relevant}. */
  embed?: Embed;
  decayLambda?: number;
}

function byTimeThenRecord(a: ScoredEdge, b: ScoredEdge): number {
  const time = (a.validFrom ?? 0) - (b.validFrom ?? 0);
  if (time !== 0) return time;
  const ra = recordOf(a.participants) ?? "";
  const rb = recordOf(b.participants) ?? "";
  return ra < rb ? -1 : ra > rb ? 1 : 0;
}

/** Messages of one session, stored as hyperedges around `session:<id>`. */
export class SessionMemory {
  readonly node: string;

  constructor(
    readonly client: HypergraphClient,
    readonly sessionId: string,
    private readonly options: SessionMemoryOptions = {},
  ) {
    this.node = sessionNode(sessionId);
  }

  async add(records: ChatRecord[]): Promise<void> {
    const kept = records.filter((r) => r.content.length > 0);
    if (kept.length === 0) return;
    const embed = this.options.embed;
    const vectors = embed ? await Promise.all(kept.map((r) => embed(r.content))) : [];
    const timestamp = nowEpoch();
    const result = await this.client.insertBatch(
      kept.map((record, i) => ({
        eventType: roleEventType(record.role),
        participants: [...new Set([this.node, recordNode(), ...(record.entities ?? [])])],
        validFrom: record.timestamp ?? timestamp,
        summary: record.content,
        embedding: vectors[i],
      })),
      this.sessionId,
    );
    if (!result.ok) throw new Error(`failed to store messages: ${result.errors.join("; ")}`);
  }

  /** Messages oldest first; `limit` keeps only the newest `limit` (0 = all). */
  async history(limit = 0): Promise<ChatRecord[]> {
    const edges = await this.client.traverse([this.node], {
      kHops: 1,
      topK: limit,
      order: "chronological",
      eventTypes: [...MESSAGE_EVENT_TYPES],
      asOf: nowEpoch() + 1,
    });
    return edges.sort(byTimeThenRecord).flatMap((edge) => {
      const role = eventTypeRole(edge.eventType);
      if (!role) return [];
      return [
        {
          role,
          content: edge.summary,
          entities: entitiesOf(edge.participants),
          timestamp: edge.validFrom,
          id: edge.id,
        },
      ];
    });
  }

  /** Session facts most relevant to `query`, reaching `kHops` through linked entities. */
  async relevant(
    query: string,
    k = 4,
    options: { kHops?: number; seeds?: string[] } = {},
  ): Promise<ScoredEdge[]> {
    const embedding = this.options.embed ? await this.options.embed(query) : undefined;
    return this.client.traverse([this.node, ...(options.seeds ?? [])], {
      kHops: options.kHops ?? 1,
      embedding,
      topK: k,
      decayLambda: this.options.decayLambda ?? 0,
      asOf: nowEpoch() + 1,
    });
  }

  /** Deletes every message of the session; returns how many were removed. */
  async clear(): Promise<number> {
    const ids = (await this.history()).flatMap((r) => (r.id ? [r.id] : []));
    return this.client.delete(ids);
  }
}

export const DEFAULT_SESSION = "default";

export interface TemporalHypergraphOptions extends HypergraphClientOptions, SessionMemoryOptions {
  /** Session to read and write (default `"default"`). */
  sessionId?: string;
  /** A preconfigured client (overrides the connection options). */
  client?: HypergraphClient;
}

/** One-line entry point: `const memory = new TemporalHypergraph({ apiKey: "..." })`. */
export class TemporalHypergraph extends SessionMemory {
  constructor(options: TemporalHypergraphOptions = {}) {
    const { sessionId, client, embed, decayLambda, ...connection } = options;
    super(client ?? new HypergraphClient(connection), sessionId ?? DEFAULT_SESSION, { embed, decayLambda });
  }
}
