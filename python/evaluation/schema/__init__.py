"""Draft-07 JSON schemas for datasets and LLM outputs."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

SCHEMA_DIR = Path(__file__).resolve().parent


@cache
def load(name: str) -> dict[str, Any]:
    """``schema/<name>.schema.json`` as a dict."""
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


@cache
def validator(name: str) -> Draft7Validator:
    schema = load(name)
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


@cache
def graph_validator() -> Draft7Validator:
    """Validates a bare ``{nodes, hyperedges}`` graph (e.g. a prediction)."""
    definitions = load("hypergraph")["definitions"]
    return Draft7Validator({"$ref": "#/definitions/graph", "definitions": definitions})


def _messages(validator_: Draft7Validator, instance: Any) -> list[str]:
    return [
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
        for e in sorted(validator_.iter_errors(instance), key=lambda e: list(e.absolute_path))
    ]


def errors(name: str, instance: Any) -> list[str]:
    """Human-readable validation errors (empty when valid)."""
    return _messages(validator(name), instance)


def graph_errors(graph: Any) -> list[str]:
    return _messages(graph_validator(), graph)


def validate(name: str, instance: Any) -> None:
    problems = errors(name, instance)
    if problems:
        raise ValueError(f"{name} schema violations: " + "; ".join(problems[:10]))
