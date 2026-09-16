"""Graph-topology conventions shared by every framework integration.

A session is an entity node (``session:<id>``) that participates in every
hyperedge written for that session, so reading a session is a 1-hop traversal
seeded at that node.

Each chat message or document also gets its own record node
(``record:<hex time_ns>-<uuid>``). It keeps records out of the engine's
temporal conflict resolution, which would otherwise treat two messages with
the same event type and session as one evolving fact and truncate the older
one. The hex timestamp also orders records written within the same second.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable, Sequence

SESSION_PREFIX = "session:"
RECORD_PREFIX = "record:"

_stamp_lock = threading.Lock()
_last_stamp = 0


def session_node(session_id: str) -> str:
    """Entity id of a session."""
    if not session_id:
        raise ValueError("session_id must not be empty")
    return f"{SESSION_PREFIX}{session_id}"


def record_node() -> str:
    """A fresh record entity id; ids created later in this process sort later."""
    global _last_stamp
    with _stamp_lock:
        _last_stamp = max(time.time_ns(), _last_stamp + 1)
        stamp = _last_stamp
    return f"{RECORD_PREFIX}{stamp:016x}-{uuid.uuid4().hex[:12]}"


def scoped_participants(session_id: str | None, participants: Iterable[str]) -> list[str]:
    """``participants`` prefixed by the session node (when given), deduplicated in order."""
    seen: dict[str, None] = {}
    if session_id:
        seen[session_node(session_id)] = None
    for participant in participants:
        if participant:
            seen.setdefault(participant, None)
    return list(seen)


def record_of(participants: Sequence[str]) -> str | None:
    """The record node among ``participants``, if any."""
    return next((p for p in participants if p.startswith(RECORD_PREFIX)), None)


def entities_of(participants: Sequence[str]) -> list[str]:
    """``participants`` without session and record nodes."""
    return [
        p for p in participants if not p.startswith((SESSION_PREFIX, RECORD_PREFIX))
    ]
