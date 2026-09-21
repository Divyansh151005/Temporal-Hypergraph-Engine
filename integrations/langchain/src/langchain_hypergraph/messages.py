"""Conversions between LangChain messages and stored chat records."""

from __future__ import annotations

from collections.abc import Sequence

from hypergraph_client import ChatRecord, Role
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from typing_extensions import assert_never

ENTITIES_KWARG = "entities"
"""``additional_kwargs`` key holding entity ids to link a message to."""

_ROLES = {
    "human": Role.HUMAN,
    "chat": Role.HUMAN,
    "ai": Role.AI,
    "system": Role.SYSTEM,
    "tool": Role.TOOL,
    "function": Role.TOOL,
}


def message_text(message: BaseMessage) -> str:
    """The text of ``message``; non-text content blocks are dropped."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def to_record(message: BaseMessage) -> ChatRecord:
    role = _ROLES.get(message.type)
    if role is None:
        raise ValueError(f"unsupported message type {message.type!r}")
    entities = message.additional_kwargs.get(ENTITIES_KWARG) or ()
    return ChatRecord(role, message_text(message), entities=tuple(str(e) for e in entities))


def to_message(record: ChatRecord) -> BaseMessage:
    kwargs = {ENTITIES_KWARG: list(record.entities)} if record.entities else {}
    match record.role:
        case Role.HUMAN:
            return HumanMessage(record.content, id=record.id, additional_kwargs=kwargs)
        case Role.AI:
            return AIMessage(record.content, id=record.id, additional_kwargs=kwargs)
        case Role.SYSTEM:
            return SystemMessage(record.content, id=record.id, additional_kwargs=kwargs)
        case Role.TOOL:
            # The engine stores only the text, so the originating tool call id is lost.
            return ToolMessage(record.content, tool_call_id="", id=record.id, additional_kwargs=kwargs)
        case _:
            assert_never(record.role)


def to_records(messages: Sequence[BaseMessage]) -> list[ChatRecord]:
    return [to_record(m) for m in messages]


def to_messages(records: Sequence[ChatRecord]) -> list[BaseMessage]:
    return [to_message(r) for r in records]
