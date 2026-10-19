#!/usr/bin/env python3
"""Hypergraph k-hop retrieval vs chunked vector RAG on synthetic multi-hop temporal QA.

Reports context-token reduction (fixed k and recall-matched k), multi-hop hallucination
reduction (LLM answer + judge, with LLM-free coverage/staleness proxies) and retrieval
latency percentiles. Writes ``results.json``, ``results.csv`` and ``summary.md`` to
``<out-dir>/<timestamp>/``.

    python benchmarks/run_benchmark.py --offline-corpus --skip-llm
    python benchmarks/run_benchmark.py --dataset datasets/synthetic/corp.jsonl --hops 3 --grpc
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from hypergraph_client import HashingEmbedder
from hypergraph_client.server import find_server_binary
from hypergraph_evaluation.llm import LLM, LLMError
from hypergraph_evaluation.synthetic_generator import generate_dataset, load_records
from hypergraph_evaluation.temporal import to_iso

from baseline_vector import Chunk, VectorBaseline, chunk_document, render_chunks
from hypergraph_runner import EntityLinker, Fact, GrpcRunner, NativeRunner, graph_edges, render_facts
from metrics import (
    ContenderStats,
    Entity,
    Question,
    TokenCounter,
    coverage,
    exact_verdict,
    percentiles,
    reduction,
    stale_contaminated,
)

log = logging.getLogger("benchmark")

REPO = Path(__file__).resolve().parent.parent
TARGETS = {"token_reduction": 0.80, "latency_p95_ms": 2.0}

ANSWER_SYSTEM = """You answer questions strictly from the provided context.
Reply with the name of the single entity that answers the question as of the date the question asks about.
Facts may have changed over time; use only what held at that date.
If the context does not determine the answer, reply UNKNOWN.
Respond with JSON: {"answer": "<entity name or UNKNOWN>"}."""

JUDGE_SYSTEM = """You grade an answer to a time-sensitive question.
verdict is "correct" if the answer names the gold entity (aliases count), "stale" if it names the outdated entity,
"abstained" if it declines to answer, and "wrong" otherwise.
Respond with JSON: {"verdict": "correct" | "stale" | "wrong" | "abstained"}."""


@dataclass
class Trial:
    """Everything observed for one question."""

    question: Question
    query: np.ndarray
    seeds: list[str]
    baseline_hits: list[Chunk]
    facts: list[Fact]
    contexts: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    matched_k: int = 0


def entity_table(graph: dict) -> dict[str, Entity]:
    return {n["id"]: Entity(n["label"], tuple(n.get("aliases") or ())) for n in graph["nodes"]}


def questions_of(record: dict) -> list[Question]:
    entities = entity_table(record["ground_truth_graph"])
    edges = {e["id"]: e for e in record["ground_truth_graph"]["hyperedges"]}
    return [
        Question(
            id=qa["id"],
            dataset_id=record["dataset_id"],
            text=qa["question"],
            as_of=int(qa["as_of"]),
            hops=int(qa["hops"]),
            answer=entities.get(qa.get("answer_id"), Entity(qa["answer"])),
            stale=(entities.get(qa.get("stale_answer_id")) or Entity(qa["stale_answer"])) if qa.get("stale_answer") else None,
            gold_edges=[[entities[p] for p in edges[i]["participants"]] for i in qa["gold_edge_ids"]],
            seed_labels=[entities[s].label for s in qa["seeds"]],
        )
        for qa in record["qa"]
    ]


def load_corpus(args: argparse.Namespace) -> list[dict]:
    if args.dataset:
        records = load_records(args.dataset)
    else:
        records = generate_dataset(args.n, seed=args.seed, noise=args.noise, workers=4)
    return records[: args.limit] if args.limit else records


def load_graphs(args: argparse.Namespace, records: Sequence[dict]) -> dict[str, dict]:
    truth = {r["dataset_id"]: r["ground_truth_graph"] for r in records}
    if not args.graphs:
        return truth
    predicted = {r["dataset_id"]: r["graph"] for r in load_records(args.graphs)}
    missing = sorted(set(truth) - set(predicted))
    if missing:
        log.warning("%d documents have no extracted graph; they contribute no facts", len(missing))
    return {d: predicted.get(d, {"nodes": [], "hyperedges": []}) for d in truth}


def time_calls(call, items: Sequence[Any], warmup: int, repeats: int) -> list[int]:
    for item in items[:warmup]:
        call(item)
    return [call(item) for _ in range(repeats) for item in items]


def criterion_numbers() -> dict[str, float]:
    """Mean ns per iteration for the engine's own 3-hop Criterion benches, if they have been run."""
    found = {}
    for path in sorted((REPO / "target" / "criterion").glob("traverse_3hop*/new/estimates.json")):
        estimate = json.loads(path.read_text())["mean"]["point_estimate"]
        found[path.parent.parent.name] = estimate / 1e6
    return found


@dataclass(frozen=True)
class Graded:
    verdict: str
    answered_by: str | None = None
    judged_by: str | None = None


def answer_and_grade(answerer: LLM, judge: LLM, trial: Trial, contender: str) -> Graded:
    """Answers from the contender's context, grading by name match and falling back to the judge."""
    question = trial.question
    try:
        reply, answered_by = answerer.complete_json_with_source(
            ANSWER_SYSTEM, f"Context:\n{trial.contexts[contender]}\n\nQuestion: {question.text}", "answer"
        )
        verdict = exact_verdict(reply["answer"], question)
        if verdict is not None:
            return Graded(verdict, answered_by)
        stale = ", ".join(question.stale.names) if question.stale else "(none)"
        judged, judged_by = judge.complete_json_with_source(
            JUDGE_SYSTEM,
            f"Question: {question.text}\nGold entity: {', '.join(question.answer.names)}\n"
            f"Outdated entity: {stale}\nAnswer: {reply['answer']}",
            "judge",
        )
        return Graded(judged["verdict"], answered_by, judged_by)
    except LLMError as error:
        log.warning("%s/%s: %s", question.id, contender, error)
        return Graded("abstained")


def fmt(value: float | None, pct: bool = False, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.1f}%" if pct else f"{value:.{digits}f}"


def render_summary(results: dict) -> str:
    c = results["contenders"]
    r = results["reductions"]
    lat = results["latency"]
    hg, fixed, matched = c["hypergraph"], c["baseline_fixed_k"], c["baseline_recall_matched"]
    cfg = results["config"]
    lines = [
        "# Hypergraph vs vector RAG",
        "",
        f"{results['corpus']['documents']} documents, {results['corpus']['questions']} multi-hop questions, "
        f"{results['corpus']['chunks']} chunks, {results['corpus']['hyperedges']} hyperedges; "
        f"{cfg['hops']}-hop traversal, baseline k={cfg['k']}, tokenizer `{cfg['tokenizer']}`, "
        f"embedder `{cfg['embedder']}`, baseline backend `{cfg['baseline_backend']}`.",
        "",
        "| Metric | Hypergraph | Vector RAG (k fixed) | Vector RAG (recall-matched k) |",
        "|---|---|---|---|",
        f"| Mean context tokens | {fmt(hg['mean_tokens'], digits=1)} | {fmt(fixed['mean_tokens'], digits=1)} | {fmt(matched['mean_tokens'], digits=1)} |",
        f"| Gold-fact coverage | {fmt(hg['gold_fact_coverage'], True)} | {fmt(fixed['gold_fact_coverage'], True)} | {fmt(matched['gold_fact_coverage'], True)} |",
        f"| Stale-fact contamination | {fmt(hg['stale_contamination'], True)} | {fmt(fixed['stale_contamination'], True)} | {fmt(matched['stale_contamination'], True)} |",
        f"| Hallucination rate (LLM) | {fmt(hg['hallucination_rate'], True)} | {fmt(fixed['hallucination_rate'], True)} | n/a |",
        "",
        "| Reduction vs vector RAG | Value | Target |",
        "|---|---|---|",
        f"| Context tokens (k fixed) | {fmt(r['token_fixed_k'], True)} | ≈{TARGETS['token_reduction'] * 100:.0f}% |",
        f"| Context tokens (recall-matched k) | {fmt(r['token_recall_matched'], True)} | |",
        f"| Stale-fact contamination | {fmt(r['stale_contamination'], True)} | |",
        f"| Multi-hop hallucinations | {fmt(r['hallucination'], True)} | > 0 |",
        "",
        "| Retrieval latency | p50 ms | p95 ms | p99 ms | n |",
        "|---|---|---|---|---|",
    ]
    for name, stats in lat.items():
        lines.append(f"| {name} | {fmt(stats['p50_ms'])} | {fmt(stats['p95_ms'])} | {fmt(stats['p99_ms'])} | {stats['n']} |")
    lines.append("")
    lines.append(f"Latency target: {cfg['hops']}-hop p95 under {TARGETS['latency_p95_ms']} ms (in-process).")
    if cfg["llm"]:
        served = ", ".join(f"{model} ({n})" for model, n in cfg["llm"]["answered_by"].items()) or "none"
        judged = ", ".join(f"{model} ({n})" for model, n in cfg["llm"]["judged_by"].items()) or "none"
        lines += ["", f"Answers served by {served}; judgements by {judged}; "
                  f"{cfg['llm']['failed']} answer calls failed and count as abstentions."]
    if results["criterion_ms"]:
        lines += ["", "Engine-only Criterion means (`cargo bench -p engine-core --bench traversal`):", ""]
        lines += [f"- `{name}`: {ms:.3f} ms" for name, ms in results["criterion_ms"].items()]
    lines += ["", f"Seed linking recall: {fmt(results['linking']['seed_recall'], True)}; "
              f"questions with no linked seed: {results['linking']['unlinked']}."]
    if results["targets_missed"]:
        lines += ["", "Targets missed: " + "; ".join(results["targets_missed"])]
    return "\n".join(lines) + "\n"


def check_targets(results: dict) -> list[str]:
    missed = []
    token = results["reductions"]["token_fixed_k"]
    if token is None or token < TARGETS["token_reduction"]:
        missed.append(f"token reduction {fmt(token, True)} < {TARGETS['token_reduction'] * 100:.0f}%")
    p95 = results["latency"]["hypergraph_pyo3"]["p95_ms"]
    if not p95 < TARGETS["latency_p95_ms"]:
        missed.append(f"in-process p95 {p95:.3f} ms >= {TARGETS['latency_p95_ms']} ms")
    hallucination = results["reductions"]["hallucination"]
    if results["config"]["llm"] and (hallucination is None or hallucination <= 0):
        missed.append(f"hallucination reduction {fmt(hallucination, True)} is not positive")
    return missed


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="synthetic dataset (JSON array or JSONL)")
    source.add_argument("--offline-corpus", action="store_true", help="generate template articles in-process (no LLM)")
    parser.add_argument("--graphs", type=Path, help="extracted graphs (extract_graph.py output) instead of gold graphs")
    parser.add_argument("--n", type=int, default=200, help="documents for --offline-corpus")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.0, help="noise rate for --offline-corpus")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N documents")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--k", type=int, default=5, help="baseline top-k chunks")
    parser.add_argument("--max-k", type=int, default=20, help="largest k tried for recall matching")
    parser.add_argument("--top-k", type=int, default=8, help="hypergraph facts kept after re-ranking")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256, help="hashing embedder dimension")
    parser.add_argument("--baseline-backend", choices=("qdrant", "numpy"), default="qdrant")
    parser.add_argument("--grpc", action="store_true", help="also time traversal over gRPC/UDS")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-llm", action="store_true", help="report LLM-free proxies only")
    parser.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    parser.add_argument("--answer-llm", help="answerer provider chain (default: $ANSWER_LLM, else built-in)")
    parser.add_argument("--judge-llm", help="judge provider chain (default: $JUDGE_LLM, else built-in)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results")
    parser.add_argument("--assert-targets", action="store_true", help="exit 1 if a target is missed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    counter = TokenCounter()
    embedder = HashingEmbedder(args.dim)
    embed = embedder.embed

    records = load_corpus(args)
    graphs = load_graphs(args, records)
    questions = [q for r in records for q in questions_of(r)]
    log.info("%d documents, %d questions", len(records), len(questions))

    chunks = [c for r in records for c in chunk_document(r["dataset_id"], r["raw_text"], counter, args.chunk_tokens)]
    baseline = VectorBaseline(chunks, embed, backend=args.baseline_backend)
    native = NativeRunner(embed, args.dim)
    edges = [e for g in graphs.values() for e in graph_edges(g)]
    native.ingest(edges)
    linker = EntityLinker(e for g in graphs.values() for e in entity_table(g).values())
    log.info("indexed %d chunks and %d hyperedges", len(chunks), len(edges))

    trials = []
    for question in questions:
        query = embed(question.text)
        seeds = linker.link(question.text)
        facts, _ = native.retrieve(seeds, question.as_of, query, args.hops, args.top_k)
        trial = Trial(question, query, seeds, baseline.search(query, args.max_k), facts)
        trial.contexts["hypergraph"] = render_facts(facts)
        trial.contexts["baseline"] = render_chunks(trial.baseline_hits[: args.k])
        target = coverage(trial.contexts["hypergraph"], question)
        trial.matched_k = next(
            (k for k in range(1, len(trial.baseline_hits) + 1)
             if coverage(render_chunks(trial.baseline_hits[:k]), question) >= target),
            len(trial.baseline_hits),
        )
        trial.contexts["baseline_matched"] = render_chunks(trial.baseline_hits[: trial.matched_k])
        trial.tokens = {name: counter.count(text) for name, text in trial.contexts.items()}
        trials.append(trial)

    llm_info: dict[str, Any] | None = None
    if not args.skip_llm:
        answerer = LLM.for_role("answer", args.answer_llm)
        judge = LLM.for_role("judge", args.judge_llm)
        log.info("answering with %s; judging with %s", answerer.label, judge.label)
        jobs = [(t, c) for t in trials for c in ("hypergraph", "baseline")]
        with ThreadPoolExecutor(args.workers) as pool:
            graded = list(pool.map(lambda job: answer_and_grade(answerer, judge, *job), jobs))
        for (trial, contender), grade in zip(jobs, graded):
            trial.verdicts[contender] = grade.verdict
        llm_info = {
            "answer_chain": answerer.label,
            "judge_chain": judge.label,
            "answered_by": dict(Counter(g.answered_by for g in graded if g.answered_by)),
            "judged_by": dict(Counter(g.judged_by for g in graded if g.judged_by)),
            "failed": sum(g.answered_by is None for g in graded),
        }

    stats = {name: ContenderStats() for name in ("hypergraph", "baseline_fixed_k", "baseline_recall_matched")}
    for trial in trials:
        for name, context in (("hypergraph", "hypergraph"), ("baseline_fixed_k", "baseline"),
                              ("baseline_recall_matched", "baseline_matched")):
            stats[name].add(trial.contexts[context], trial.tokens[context], trial.question)
        if trial.verdicts:
            stats["hypergraph"].verdicts.append(trial.verdicts["hypergraph"])
            stats["baseline_fixed_k"].verdicts.append(trial.verdicts["baseline"])

    probes = [(t.seeds, t.question.as_of, t.query) for t in trials]
    latency = {
        "hypergraph_pyo3": percentiles(time_calls(
            lambda p: native.retrieve(p[0], p[1], p[2], args.hops, args.top_k)[1], probes, args.warmup, args.repeats)),
        f"vector_{args.baseline_backend}": percentiles(time_calls(
            lambda p: baseline.timed_search(p[2], args.k)[1], probes, args.warmup, args.repeats)),
    }
    if args.grpc:
        if find_server_binary() is None:
            log.warning("--grpc requested but hypergraph-server is not built; skipping")
        else:
            grpc_runner = GrpcRunner(embed, args.dim)
            try:
                grpc_runner.ingest(edges)
                latency["hypergraph_grpc_uds"] = percentiles(time_calls(
                    lambda p: grpc_runner.retrieve(p[0], p[1], p[2], args.hops, args.top_k)[1],
                    probes, args.warmup, args.repeats))
            finally:
                grpc_runner.close()

    summaries = {name: s.summary() for name, s in stats.items()}
    hg, fixed, matched = summaries["hypergraph"], summaries["baseline_fixed_k"], summaries["baseline_recall_matched"]
    linked = [set(t.question.seed_labels) <= set(t.seeds) for t in trials]
    results: dict[str, Any] = {
        "created": to_iso(int(time.time())),
        "config": {
            "source": str(args.dataset) if args.dataset else f"offline-corpus n={args.n} seed={args.seed} noise={args.noise}",
            "graphs": str(args.graphs) if args.graphs else "gold",
            "hops": args.hops, "k": args.k, "max_k": args.max_k, "top_k": args.top_k,
            "chunk_tokens": args.chunk_tokens, "tokenizer": counter.name, "embedder": f"hashing-{args.dim}",
            "baseline_backend": args.baseline_backend, "warmup": args.warmup, "repeats": args.repeats,
            "llm": llm_info,
        },
        "corpus": {"documents": len(records), "questions": len(trials), "chunks": len(chunks), "hyperedges": len(edges)},
        "contenders": summaries,
        "reductions": {
            "token_fixed_k": reduction(fixed["mean_tokens"], hg["mean_tokens"]),
            "token_recall_matched": reduction(matched["mean_tokens"], hg["mean_tokens"]),
            "stale_contamination": reduction(fixed["stale_contamination"], hg["stale_contamination"]),
            "hallucination": reduction(fixed["hallucination_rate"], hg["hallucination_rate"]),
        },
        "latency": latency,
        "criterion_ms": criterion_numbers(),
        "linking": {"seed_recall": sum(linked) / len(linked) if linked else 0.0,
                    "unlinked": sum(not t.seeds for t in trials)},
        "targets": TARGETS,
    }
    results["targets_missed"] = check_targets(results)

    out = args.out_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "dataset_id", "contender", "tokens", "gold_fact_coverage",
                         "stale_contaminated", "verdict", "k"])
        for t in trials:
            for name, context, k in (("hypergraph", "hypergraph", len(t.facts)), ("baseline_fixed_k", "baseline", args.k),
                                     ("baseline_recall_matched", "baseline_matched", t.matched_k)):
                text = t.contexts[context]
                writer.writerow([t.question.id, t.question.dataset_id, name, t.tokens[context],
                                 round(coverage(text, t.question), 4),
                                 int(stale_contaminated(text, t.question)),
                                 t.verdicts.get(context, ""), k])
    summary = render_summary(results)
    (out / "summary.md").write_text(summary)
    print(summary)
    print(f"wrote {out}")
    native.close()
    if args.assert_targets and results["targets_missed"]:
        print("targets missed: " + "; ".join(results["targets_missed"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
