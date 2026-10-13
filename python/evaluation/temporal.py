"""Temporal normalization: ISO-8601 strings or epoch seconds to epoch seconds."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_YEAR = re.compile(r"^\d{4}$")
_YEAR_MONTH = re.compile(r"^\d{4}-\d{2}$")


def to_epoch(value: str | int | float | None) -> int | None:
    """Epoch seconds for ``value``.

    Accepts ``None``, epoch seconds (int, float or numeric string) and ISO-8601
    dates or datetimes, including reduced precision (``2023``, ``2023-10``),
    ``Z`` and numeric offsets. Naive datetimes are taken as UTC.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"not a timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text) and not _YEAR.match(text):
        return int(float(text))
    if _YEAR.match(text):
        text = f"{text}-01-01"
    elif _YEAR_MONTH.match(text):
        text = f"{text}-01"
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def to_iso(epoch: int | None) -> str | None:
    """``YYYY-MM-DDTHH:MM:SSZ`` for ``epoch`` (``None`` stays ``None``)."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_date(epoch: int | None) -> str:
    """``YYYY-MM-DD`` for ``epoch``, ``?`` when unknown."""
    if epoch is None:
        return "?"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def epoch_of(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def add_months(epoch: int, months: int) -> int:
    """Midnight UTC on the first of the month ``months`` after ``epoch``'s month."""
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    index = moment.year * 12 + moment.month - 1 + months
    return epoch_of(index // 12, index % 12 + 1)


def month_phrase(epoch: int) -> str:
    """``March 2019``."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%B %Y")


def day_phrase(epoch: int) -> str:
    """``March 5, 2019``."""
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{moment.strftime('%B')} {moment.day}, {moment.year}"
