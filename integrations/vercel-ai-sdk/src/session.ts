/**
 * Graph-topology conventions shared with the Python integrations.
 *
 * A session is an entity node (`session:<id>`) that participates in every
 * hyperedge written for it, so reading a session is a 1-hop traversal. Each
 * message also gets its own record node (`record:<hex time_ns>-<uuid>`) that
 * keeps messages out of the engine's temporal conflict resolution and orders
 * messages written within the same second.
 */
import { randomUUID } from "node:crypto";

export const SESSION_PREFIX = "session:";
export const RECORD_PREFIX = "record:";

export type Role = "user" | "assistant" | "system" | "tool";

export const MESSAGE_EVENT_TYPES = [
  "MESSAGE_HUMAN",
  "MESSAGE_AI",
  "MESSAGE_SYSTEM",
  "MESSAGE_TOOL",
] as const;
export type MessageEventType = (typeof MESSAGE_EVENT_TYPES)[number];

export function sessionNode(sessionId: string): string {
  if (!sessionId) throw new Error("sessionId must not be empty");
  return `${SESSION_PREFIX}${sessionId}`;
}

let lastStamp = 0n;

/** A fresh record entity id; ids created later in this process sort later. */
export function recordNode(): string {
  const nowNs = BigInt(Date.now()) * 1_000_000n;
  lastStamp = nowNs > lastStamp ? nowNs : lastStamp + 1n;
  const stamp = lastStamp.toString(16).padStart(16, "0");
  return `${RECORD_PREFIX}${stamp}-${randomUUID().replaceAll("-", "").slice(0, 12)}`;
}

export function recordOf(participants: readonly string[]): string | undefined {
  return participants.find((p) => p.startsWith(RECORD_PREFIX));
}

/** Participants without session and record nodes. */
export function entitiesOf(participants: readonly string[]): string[] {
  return participants.filter((p) => !p.startsWith(SESSION_PREFIX) && !p.startsWith(RECORD_PREFIX));
}

export function roleEventType(role: Role): MessageEventType {
  switch (role) {
    case "user":
      return "MESSAGE_HUMAN";
    case "assistant":
      return "MESSAGE_AI";
    case "system":
      return "MESSAGE_SYSTEM";
    case "tool":
      return "MESSAGE_TOOL";
    default: {
      const unhandled: never = role;
      throw new Error(`unknown role ${String(unhandled)}`);
    }
  }
}

export function eventTypeRole(eventType: string): Role | undefined {
  switch (eventType) {
    case "MESSAGE_HUMAN":
      return "user";
    case "MESSAGE_AI":
      return "assistant";
    case "MESSAGE_SYSTEM":
      return "system";
    case "MESSAGE_TOOL":
      return "tool";
    default:
      return undefined;
  }
}
