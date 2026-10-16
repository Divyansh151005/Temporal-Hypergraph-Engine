#!/usr/bin/env python3
"""Runs the temporal hypergraph extraction prompt over a dataset (the pipeline under test).

Uses the system prompt and strict schema from
docs/MyUnderstanding/Schema_SystemPrompts.txt and writes one prediction per
document, normalized to the ground-truth graph shape (epoch-second bounds):

    {"dataset_id": ..., "model": ..., "graph": {"nodes": [...], "hyperedges": [...]}, "raw": {...}}

    python python/evaluation/extract_graph.py --in datasets/core_100_golden_dataset.json --out results/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hypergraph_evaluation.llm import LLM, LLMError
from hypergraph_evaluation.schema import graph_errors
from hypergraph_evaluation.synthetic_generator import load_records
from hypergraph_evaluation.temporal import to_epoch, to_iso

log = logging.getLogger("extract_graph")

SYSTEM_PROMPT = """SYSTEM PROMPT: Temporal Hypergraph Extraction Engine

ROLE & PURPOSE:
You are an expert Information Extraction Engine specialized in building Temporal Hypergraphs. Your goal is to convert raw, unstructured text into N-ary hyperedges and bound them within precise temporal intervals.

CURRENT CONTEXT:
The current execution timestamp is: {current_timestamp} (ISO-8601). Use this to resolve relative temporal references (e.g., "yesterday", "last quarter").

CRITICAL INSTRUCTIONS:

1. N-ARY HYPEREDGE BINDING (No Binary Fragmentation)
   - Traditional knowledge graphs use binary triples (Node A -> Node B). You MUST NOT break complex events into fragmented pairs.
   - If an event involves multiple entities (e.g., "Company X acquired Company Y for $1B with approval from Regulator Z"), group ALL participants into ONE Hyperedge array under `participants`.
   - Assign semantic roles (e.g., "ACQUIRER", "TARGET", "REGULATOR") rather than generic structural nodes.

2. TEMPORAL BOUNDARY RESOLUTION
   - Point-in-time events (e.g., "launched product on May 1, 2024"): Set `valid_from` and `valid_to` to the exact same ISO-8601 timestamp. `is_ongoing` = false.
   - Interval states (e.g., "served as CEO from 2018 to 2021"): Set `valid_from` to start date and `valid_to` to end date. `is_ongoing` = false.
   - Open-ended / Active states (e.g., "appointed CEO in Nov 2023" or "is currently the CFO"): Set `valid_from` to the start date, set `valid_to` to `null`. `is_ongoing` = true.
   - DO NOT FABRICATE DATES: If temporal boundaries are not explicitly stated or cleanly deducible from context, you MUST set them to `null`.

3. ENTITY NORMALIZATION
   - Assign unique, canonical IDs using format `type:snake_case_name` (e.g., `person:sam_altman`, `org:microsoft`).
   - Deduplicate entities: If "Apple Inc." and "Apple" are mentioned, create exactly one entity in the `entities` array and list both under `aliases`.

4. EXPLICIT NULL HANDLING
   - If a specific field is unknown, output `null`. Do not invent plausible approximations."""


def _epoch_or_none(value: Any, field: str, edge_id: str) -> int | None:
    try:
        return to_epoch(value)
    except ValueError:
        log.warning("edge %s: unparseable %s %r treated as unknown", edge_id, field, value)
        return None


def normalize(extraction: dict[str, Any]) -> dict[str, Any]:
    """Maps the extraction schema onto the ``{nodes, hyperedges}`` evaluation shape."""
    nodes = [
        {"id": e["id"], "label": e["name"], "type": e["type"], "aliases": list(e.get("aliases", []))}
        for e in extraction["entities"]
    ]
    hyperedges = []
    for edge in extraction["hyperedges"]:
        bounds = edge["temporal_boundary"]
        hyperedges.append(
            {
                "id": edge["id"],
                "event_type": edge["event_type"],
                "participants": [p["entity_id"] for p in edge["participants"]],
                "roles": [p["role"] for p in edge["participants"]],
                "valid_from": _epoch_or_none(bounds["valid_from"], "valid_from", edge["id"]),
                "valid_to": _epoch_or_none(bounds["valid_to"], "valid_to", edge["id"]),
                "summary": edge["summary"],
                "confidence": max(0.0, min(1.0, float(edge["confidence_score"]))),
            }
        )
    return {"nodes": nodes, "hyperedges": hyperedges}


def extract(record: dict[str, Any], llm: LLM) -> dict[str, Any]:
    reference = to_iso(to_epoch(record.get("reference_time"))) or "unknown"
    system = SYSTEM_PROMPT.format(current_timestamp=reference)
    raw, model = llm.complete_json_with_source(system, record["raw_text"], "extraction")
    graph = normalize(raw)
    problems = graph_errors(graph)
    if problems:
        log.warning("%s: prediction has schema issues: %s", record["dataset_id"], problems[:3])
    return {"dataset_id": record["dataset_id"], "model": model, "graph": graph, "raw": raw}


def run(records: Sequence[dict[str, Any]], llm: LLM, workers: int) -> list[dict[str, Any]]:
    def one(record: dict[str, Any]) -> dict[str, Any]:
        try:
            return extract(record, llm)
        except LLMError as error:
            log.error("%s: extraction failed: %s", record["dataset_id"], error)
            return {
                "dataset_id": record["dataset_id"],
                "model": llm.label,
                "graph": {"nodes": [], "hyperedges": []},
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, records))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", type=Path, required=True, help="dataset (.json array or .jsonl)")
    parser.add_argument("--out", type=Path, required=True, help="predictions .jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N documents")
    parser.add_argument("--llm", help="provider chain (default: $EXTRACT_LLM, else the built-in extract chain)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = load_records(args.inp)
    if args.limit:
        records = records[: args.limit]
    llm = LLM.for_role("extract", args.llm)
    log.info("extracting with %s", llm.label)
    predictions = run(records, llm, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    failed = sum("error" in p for p in predictions)
    print(f"wrote {len(predictions)} predictions to {args.out} ({failed} failed)")
    return 1 if failed == len(predictions) else 0


if __name__ == "__main__":
    sys.exit(main())
