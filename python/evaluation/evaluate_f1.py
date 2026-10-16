#!/usr/bin/env python3
"""Graph-alignment F1 for temporal hypergraph extraction (docs/05-extraction-benchmarks.txt section 3).

Per document:

1. Node alignment: cosine similarity between node names (label plus aliases;
   a pair scores the best name pair), Hungarian assignment on ``1 - cos``,
   kept when ``cos > --node-threshold`` (0.90).
2. Edge alignment: predicted participants are remapped to truth ids, every
   (predicted, truth) pair is scored ``0.3*semantic + 0.5*jaccard +
   0.2*temporal``, and ``linear_sum_assignment(maximize=True)`` picks the
   optimal pairing. Pairs scoring above ``--edge-threshold`` (0.80) are true
   positives; unpaired predictions are false positives, unpaired truth edges
   false negatives.
3. Temporal IoU on half-open ``[valid_from, valid_to)`` intervals. Unknown
   bounds are clamped to the document horizon; two all-unknown intervals score
   1. Point events (``valid_from == valid_to``) are compared by start date
   within ``--point-tolerance-days``.

Reported micro (pooled counts) and macro (mean per document): node P/R/F1,
edge P/R/F1, participant P/R/F1 over assigned edge pairs, mean participant
Jaccard and Temporal Boundary Accuracy (mean IoU over true-positive edges).

    python python/evaluation/evaluate_f1.py --pred results/predictions.jsonl \\
        --truth datasets/core_100_golden_dataset.json --out-csv results/f1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openai import OpenAI
from scipy.optimize import linear_sum_assignment

from hypergraph_evaluation.synthetic_generator import load_records
from hypergraph_evaluation.temporal import to_epoch

DAY = 86_400
SEMANTIC_WEIGHT, PARTICIPANT_WEIGHT, TEMPORAL_WEIGHT = 0.3, 0.5, 0.2

# Minimum acceptable values from docs/05-extraction-benchmarks.txt section 5.1.
TARGETS = {
    "node_precision": 0.85,
    "node_recall": 0.80,
    "participant_f1": 0.75,
    "temporal_accuracy": 0.70,
}


# --- similarity ---------------------------------------------------------------


class Embedder(Protocol):
    def similarity(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        """Cosine similarity matrix ``len(left) x len(right)``."""
        ...


def normalize_name(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower().replace("_", " ")).split())


class TfidfEmbedder:
    """Character 3-gram TF-IDF (sublinear tf, smoothed idf fitted on the compared names)."""

    def _grams(self, text: str) -> Counter[str]:
        padded = f"#{normalize_name(text)}#"
        return Counter(padded[i : i + 3] for i in range(max(1, len(padded) - 2)))

    def similarity(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        if not left or not right:
            return np.zeros((len(left), len(right)))
        grams = [self._grams(t) for t in (*left, *right)]
        vocab = {g: i for i, g in enumerate(sorted({g for c in grams for g in c}))}
        df = np.zeros(len(vocab))
        for counts in grams:
            for g in counts:
                df[vocab[g]] += 1
        idf = np.log((1 + len(grams)) / (1 + df)) + 1
        matrix = np.zeros((len(grams), len(vocab)))
        for row, counts in enumerate(grams):
            for g, c in counts.items():
                matrix[row, vocab[g]] = (1 + math.log(c)) * idf[vocab[g]]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        return np.clip(matrix[: len(left)] @ matrix[len(left) :].T, 0.0, 1.0)


class OpenAIEmbedder:
    """``text-embedding-3-small`` (or ``$EMBEDDING_MODEL``) through the LLM router's endpoint."""

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        if client is None:
            client = OpenAI(
                api_key=os.getenv("LLM_API_KEY", "sk-local-dummy"),
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            )
        self.client = client
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._cache: dict[str, np.ndarray] = {}

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            response = self.client.embeddings.create(model=self.model, input=todo)
            for text, item in zip(todo, response.data):
                vector = np.asarray(item.embedding, dtype=float)
                self._cache[text] = vector / (np.linalg.norm(vector) or 1.0)
        return np.stack([self._cache[t] for t in texts])

    def similarity(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        if not left or not right:
            return np.zeros((len(left), len(right)))
        return np.clip(self._vectors(left) @ self._vectors(right).T, 0.0, 1.0)


# --- scoring primitives -------------------------------------------------------


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Horizon:
    start: int
    end: int

    @classmethod
    def of(cls, *graphs: dict[str, Any], reference_time: int | None = None) -> Horizon:
        times = [
            t
            for graph in graphs
            for edge in graph.get("hyperedges", [])
            for t in (to_epoch(edge.get("valid_from")), to_epoch(edge.get("valid_to")))
            if t is not None
        ]
        if reference_time is not None:
            times.append(reference_time)
        if not times:
            return cls(0, DAY)
        start, end = min(times), max(times)
        return cls(start, max(end, start) + DAY)


def temporal_score(
    pred: tuple[int | None, int | None],
    truth: tuple[int | None, int | None],
    horizon: Horizon,
    point_tolerance: int = 31 * DAY,
) -> float:
    """IoU of half-open intervals; see the module docstring for unknown bounds and points."""
    if pred == (None, None) and truth == (None, None):
        return 1.0
    p_point = pred[0] is not None and pred[0] == pred[1]
    t_point = truth[0] is not None and truth[0] == truth[1]
    if p_point or t_point:
        if pred[0] is None or truth[0] is None:
            return 0.0
        return 1.0 if abs(pred[0] - truth[0]) <= point_tolerance else 0.0

    def clamp(interval: tuple[int | None, int | None]) -> tuple[int, int]:
        start = horizon.start if interval[0] is None else interval[0]
        end = horizon.end if interval[1] is None else interval[1]
        return start, max(start, end)

    (ps, pe), (ts, te) = clamp(pred), clamp(truth)
    if pe == ps and te == ts:
        return 1.0 if ps == ts else 0.0
    intersection = max(0, min(pe, te) - max(ps, ts))
    union = max(pe, te) - min(ps, ts)
    return intersection / union if union > 0 else 0.0


def prf(tp: float, pred: float, truth: float) -> tuple[float, float, float]:
    """Precision, recall, F1; empty prediction and empty truth count as perfect."""
    precision = tp / pred if pred else (1.0 if not truth else 0.0)
    recall = tp / truth if truth else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# --- alignment ----------------------------------------------------------------


def _names(node: dict[str, Any]) -> list[str]:
    names = [node.get("label") or node.get("name") or node["id"].split(":", 1)[-1]]
    names += [a for a in node.get("aliases", []) if a]
    return names


def align_nodes(
    pred_nodes: Sequence[dict[str, Any]],
    truth_nodes: Sequence[dict[str, Any]],
    embedder: Embedder,
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Predicted node id to (truth node id, cosine) for accepted Hungarian pairs."""
    if not pred_nodes or not truth_nodes:
        return {}
    pred_names = [_names(n) for n in pred_nodes]
    truth_names = [_names(n) for n in truth_nodes]
    flat = embedder.similarity(
        [x for names in pred_names for x in names], [x for names in truth_names for x in names]
    )
    rows = np.cumsum([0] + [len(n) for n in pred_names])
    cols = np.cumsum([0] + [len(n) for n in truth_names])
    sim = np.array(
        [
            [flat[rows[i] : rows[i + 1], cols[j] : cols[j + 1]].max() for j in range(len(truth_nodes))]
            for i in range(len(pred_nodes))
        ]
    )
    pred_idx, truth_idx = linear_sum_assignment(1.0 - sim)
    return {
        pred_nodes[i]["id"]: (truth_nodes[j]["id"], float(sim[i, j]))
        for i, j in zip(pred_idx, truth_idx)
        if sim[i, j] > threshold
    }


def _bounds(edge: dict[str, Any]) -> tuple[int | None, int | None]:
    return to_epoch(edge.get("valid_from")), to_epoch(edge.get("valid_to"))


@dataclass
class DocScore:
    dataset_id: str
    node_tp: int = 0
    node_pred: int = 0
    node_truth: int = 0
    edge_tp: int = 0
    edge_pred: int = 0
    edge_truth: int = 0
    participant_overlap: int = 0
    participant_pred: int = 0
    participant_truth: int = 0
    jaccards: list[float] = field(default_factory=list)
    tp_ious: list[float] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        np_, nr, nf = prf(self.node_tp, self.node_pred, self.node_truth)
        ep, er, ef = prf(self.edge_tp, self.edge_pred, self.edge_truth)
        pp, pr, pf = prf(self.participant_overlap, self.participant_pred, self.participant_truth)
        return {
            "node_precision": np_,
            "node_recall": nr,
            "node_f1": nf,
            "edge_precision": ep,
            "edge_recall": er,
            "edge_f1": ef,
            "participant_precision": pp,
            "participant_recall": pr,
            "participant_f1": pf,
            "participant_jaccard": float(np.mean(self.jaccards)) if self.jaccards else float(not self.edge_truth),
            "temporal_accuracy": float(np.mean(self.tp_ious)) if self.tp_ious else float(not self.edge_truth),
        }


@dataclass(frozen=True)
class Config:
    node_threshold: float = 0.90
    edge_threshold: float = 0.80
    point_tolerance_days: int = 31


def score_document(
    dataset_id: str,
    pred: dict[str, Any],
    truth: dict[str, Any],
    embedder: Embedder,
    config: Config = Config(),
    reference_time: int | None = None,
) -> DocScore:
    pred_nodes, truth_nodes = pred.get("nodes", []), truth.get("nodes", [])
    pred_edges, truth_edges = pred.get("hyperedges", []), truth.get("hyperedges", [])
    mapping = align_nodes(pred_nodes, truth_nodes, embedder, config.node_threshold)
    score = DocScore(
        dataset_id,
        node_tp=len(mapping),
        node_pred=len(pred_nodes),
        node_truth=len(truth_nodes),
        edge_pred=len(pred_edges),
        edge_truth=len(truth_edges),
        participant_pred=sum(len(set(e["participants"])) for e in pred_edges),
        participant_truth=sum(len(set(e["participants"])) for e in truth_edges),
    )
    if not pred_edges or not truth_edges:
        return score

    remapped = [
        {mapping[p][0] if p in mapping else f"unmatched:{p}" for p in e["participants"]} for e in pred_edges
    ]
    truth_sets = [set(e["participants"]) for e in truth_edges]
    semantic = embedder.similarity(
        [normalize_name(e["event_type"]) for e in pred_edges],
        [normalize_name(e["event_type"]) for e in truth_edges],
    )
    pred_types = [normalize_name(e["event_type"]) for e in pred_edges]
    truth_types = [normalize_name(e["event_type"]) for e in truth_edges]
    horizon = Horizon.of(pred, truth, reference_time=reference_time)
    tolerance = config.point_tolerance_days * DAY

    total = np.zeros((len(pred_edges), len(truth_edges)))
    jac = np.zeros_like(total)
    iou = np.zeros_like(total)
    for i, pred_edge in enumerate(pred_edges):
        for j, truth_edge in enumerate(truth_edges):
            sem = 1.0 if pred_types[i] == truth_types[j] else float(semantic[i, j])
            jac[i, j] = jaccard(remapped[i], truth_sets[j])
            iou[i, j] = temporal_score(_bounds(pred_edge), _bounds(truth_edge), horizon, tolerance)
            total[i, j] = SEMANTIC_WEIGHT * sem + PARTICIPANT_WEIGHT * jac[i, j] + TEMPORAL_WEIGHT * iou[i, j]

    for i, j in zip(*linear_sum_assignment(total, maximize=True)):
        score.participant_overlap += len(remapped[i] & truth_sets[j])
        score.jaccards.append(float(jac[i, j]))
        if total[i, j] > config.edge_threshold:
            score.edge_tp += 1
            score.tp_ious.append(float(iou[i, j]))
    return score


# --- dataset level ------------------------------------------------------------


def _graph(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("graph") or record.get("ground_truth_graph") or {"nodes": [], "hyperedges": []}


def evaluate(
    predictions: Sequence[dict[str, Any]],
    truths: Sequence[dict[str, Any]],
    embedder: Embedder | None = None,
    config: Config = Config(),
) -> tuple[list[DocScore], dict[str, dict[str, float]]]:
    """Scores every truth document (missing predictions score as empty graphs)."""
    embedder = embedder or TfidfEmbedder()
    by_id = {p["dataset_id"]: p for p in predictions}
    docs = [
        score_document(
            truth["dataset_id"],
            _graph(by_id.get(truth["dataset_id"], {})),
            _graph(truth),
            embedder,
            config,
            to_epoch(truth.get("reference_time")),
        )
        for truth in truths
    ]
    return docs, aggregate(docs)


def aggregate(docs: Sequence[DocScore]) -> dict[str, dict[str, float]]:
    pooled = DocScore("micro")
    for doc in docs:
        for name in (
            "node_tp", "node_pred", "node_truth", "edge_tp", "edge_pred", "edge_truth",
            "participant_overlap", "participant_pred", "participant_truth",
        ):
            setattr(pooled, name, getattr(pooled, name) + getattr(doc, name))
        pooled.jaccards += doc.jaccards
        pooled.tp_ious += doc.tp_ious
    per_doc = [d.metrics() for d in docs]
    macro = {k: float(np.mean([m[k] for m in per_doc])) for k in per_doc[0]} if per_doc else {}
    counts = {k: v for k, v in asdict(pooled).items() if isinstance(v, int)}
    return {"micro": pooled.metrics(), "macro": macro, "counts": counts, "documents": {"n": len(docs)}}


def write_csv(path: Path, docs: Sequence[DocScore], summary: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(summary["micro"])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_id", *metric_names])
        for doc in docs:
            metrics = doc.metrics()
            writer.writerow([doc.dataset_id, *(f"{metrics[m]:.4f}" for m in metric_names)])
        for scope in ("micro", "macro"):
            writer.writerow([f"__{scope}__", *(f"{summary[scope].get(m, 0.0):.4f}" for m in metric_names)])


def regressions(summary: dict, baseline: dict, max_drop: float) -> list[str]:
    """Micro metrics that fell by more than ``max_drop`` versus ``baseline``."""
    problems = []
    for metric in ("node_f1", "edge_f1", "participant_f1", "temporal_accuracy"):
        before = baseline.get("micro", {}).get(metric)
        after = summary["micro"][metric]
        if before is not None and after < before - max_drop:
            problems.append(f"{metric} dropped {before:.4f} -> {after:.4f} (max drop {max_drop})")
    return problems


def render(summary: dict[str, dict[str, float]]) -> str:
    lines = [f"{'metric':<24}{'micro':>8}{'macro':>8}  target"]
    for metric, value in summary["micro"].items():
        target = TARGETS.get(metric)
        flag = "" if target is None else f"> {target:.2f} {'ok' if value > target else 'BELOW'}"
        lines.append(f"{metric:<24}{value:>8.3f}{summary['macro'].get(metric, 0.0):>8.3f}  {flag}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="predictions (.jsonl or .json)")
    parser.add_argument("--truth", type=Path, required=True, help="ground-truth dataset (.json or .jsonl)")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--embedder", choices=("tfidf", "openai"), default="tfidf")
    parser.add_argument("--node-threshold", type=float, default=0.90)
    parser.add_argument("--edge-threshold", type=float, default=0.80)
    parser.add_argument("--point-tolerance-days", type=int, default=31)
    parser.add_argument("--baseline", type=Path, help="previous --out-json to compare against")
    parser.add_argument("--max-drop", type=float, default=0.02, help="allowed micro F1 drop versus --baseline")
    parser.add_argument("--min-recall", type=float, help="fail if micro node or edge recall is below this")
    args = parser.parse_args(argv)

    embedder: Embedder = OpenAIEmbedder() if args.embedder == "openai" else TfidfEmbedder()
    config = Config(args.node_threshold, args.edge_threshold, args.point_tolerance_days)
    docs, summary = evaluate(load_records(args.pred), load_records(args.truth), embedder, config)
    print(render(summary))
    if args.out_csv:
        write_csv(args.out_csv, docs, summary)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {**summary, "config": asdict(config), "per_document": {d.dataset_id: d.metrics() for d in docs}}
        args.out_json.write_text(json.dumps(payload, indent=1) + "\n")

    failures: list[str] = []
    if args.baseline:
        failures += regressions(summary, json.loads(args.baseline.read_text()), args.max_drop)
    if args.min_recall is not None:
        for metric in ("node_recall", "edge_recall"):
            if summary["micro"][metric] < args.min_recall:
                failures.append(f"{metric} {summary['micro'][metric]:.4f} < {args.min_recall}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
