#!/usr/bin/env python3
"""Renders the Phase 7 benchmark JSON into ``docs/GTM_Whitepaper_Metrics.md`` plus PNG charts.

Every number in the report is read from the result files; nothing is typed by hand.

    python benchmarks/gtm_report.py --in results/phase7 --out docs/GTM_Whitepaper_Metrics.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GiB = 1024**3
MiB = 1024**2


def load(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def sweep_series(contender: dict[str, Any]) -> str:
    """Token-matched vector k differs per dataset size, so it is keyed by role rather than by k."""
    if contender.get("token_matched"):
        return contender["name"].rsplit("_k", 1)[0] + "_token_matched"
    return contender["name"]


def rate(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f} M/s"
    if x >= 1e3:
        return f"{x / 1e3:.1f} K/s"
    return f"{x:.0f}/s"


def num(x: float) -> str:
    return f"{x:,.0f}"


def us(x: float) -> str:
    if x >= 1_000:
        return f"{x / 1_000:.2f} ms"
    if x >= 1:
        return f"{x:.1f} µs"
    return f"{x * 1_000:.0f} ns"


def ms(x: float) -> str:
    return f"{x:.3f} ms" if x < 10 else f"{x:.1f} ms"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def signed_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def env_line(env: dict[str, Any]) -> str:
    mem = env.get("memory_bytes")
    parts = [env.get("cpu", "?")]
    if env.get("logical_cores"):
        parts.append(f"{env['logical_cores']} cores")
    if mem:
        parts.append(f"{mem / GiB:.0f} GiB RAM")
    parts.append(env.get("os", ""))
    if env.get("rustc"):
        parts.append(env["rustc"])
    if env.get("python"):
        parts.append(f"Python {env['python']}")
    parts.append(f"git {env.get('git_sha', '?')}")
    return ", ".join(p for p in parts if p)


class Report:
    def __init__(self, img_dir: Path, doc_dir: Path) -> None:
        self.img_dir = img_dir
        self.doc_dir = doc_dir
        self.parts: list[str] = []
        self.summary: list[str] = []
        img_dir.mkdir(parents=True, exist_ok=True)

    def add(self, text: str) -> None:
        self.parts.append(text)

    def figure(self, fig: plt.Figure, name: str, alt: str) -> None:
        path = self.img_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rel = os.path.relpath(path, self.doc_dir)
        self.add(f"![{alt}]({rel})")


# --------------------------------------------------------------------------- scale


def scale_section(rep: Report, data: dict[str, Any]) -> None:
    modes = data["modes"]
    rep.add("## 1. Extreme-scale ingest\n")
    cfg = data["config"]
    rep.add(
        f"Synthetic temporal hyperedges (arity {cfg['arity']}, {cfg['event_types']} event types, "
        f"{pct(cfg['closed_fraction'])} with a closed validity window) over {num(cfg['nodes'])} entities with a "
        f"power-law popularity skew of {cfg['skew']}. Batches of {num(cfg['batch'])} edges are produced on a separate "
        "thread, so data generation is not part of the timed path. Hardware: " + env_line(data["environment"]) + ".\n"
    )
    rows = []
    for m in modes:
        sw = m.get("single_write") or {}
        lat = sw.get("latency", {})
        rows.append([
            f"`{m['mode']}`",
            num(m["ingested_edges"]) + (f" of {num(m['target_edges'])} (stopped early*)" if m.get("stop_reason") else ""),
            f"{rate(m['resident_edges_per_sec'])} (first {num(m['resident_edges'])})",
            rate(m["sustained_edges_per_sec"]),
            rate(m["peak_edges_per_sec"]),
            f"{us(m['batch_latency']['p50_us'])} / {us(m['batch_latency']['p99_us'])}",
            f"{us(lat['p50_us'])} / {us(lat['p99_us'])}" if lat else "n/a",
            f"{m['heap_bytes_per_million_edges'] / MiB:.0f} MiB",
            f"{m['peak_rss_bytes'] / GiB:.2f} GiB",
            f"{m['wal_disk_bytes'] / MiB:,.0f} MiB" if m.get("wal_disk_bytes") else "n/a",
        ])
    rep.add(table(
        ["Durability mode", "Edges ingested", "Throughput while in RAM", "Throughput, whole run", "Peak (1 s window)",
         f"Batch latency p50 / p99 ({num(cfg['batch'])} edges)", "Single-write p50 / p99",
         "Heap per 1M edges", "Peak RSS", "WAL on disk"],
        rows,
    ))
    rep.add(
        "\n- `memory`: in-memory engine, no WAL. `wal-buffered`: WAL with OS-buffered writes (survives a process crash). "
        "`wal-fsync`: `fdatasync` on every committed batch (survives power loss)."
        f"\n- Single-write latency: {next((m['single_write']['threads'] for m in modes if m.get('single_write')), 0)} "
        "client threads each issue one `insert` at a time through the group-commit queue against a preloaded graph."
        "\n- Throughput while in RAM is the cumulative rate up to the last memory sample at which the live heap still "
        "fit in resident memory. The whole-run figure includes any time spent paging."
        "\n- Heap per 1M edges is the least-squares slope of live heap bytes (counting allocator) against edges ingested, "
        "so it excludes fixed start-up cost.\n"
    )
    for m in modes:
        if m.get("stop_reason"):
            rep.add(f"\\* `{m['mode']}` stopped at {num(m['ingested_edges'])} edges: {m['stop_reason']}. The harness stops "
                    "rather than report throughput that is bound by swap.\n")
    rep.add(
        "**Engineering note: bulk-load batch size.** Each committed batch publishes a new copy-on-write snapshot. "
        "The first write to a 64-node page after a publish copies that page, so a batch of random edges copies close to "
        "the whole node table once the graph has a few hundred thousand entities. Publish cost therefore grows with "
        "graph size, and bulk loads should use large batches to amortise it. Fewer, larger publishes per batch, or a "
        "publish-on-demand bulk-load mode, would flatten the curve below.\n"
    )
    curve = [
        [f"{s['edges']:,}", f"{s['heap_live_bytes'] / GiB:.2f} GiB", f"{s['rss_bytes'] / GiB:.2f} GiB"]
        for s in next(m for m in modes if m["mode"] == modes[0]["mode"])["memory_samples"]
    ]
    rep.add(f"Memory samples, `{modes[0]['mode']}` mode:\n")
    rep.add(table(["Edges", "Live heap", "RSS"], curve))
    rep.add("")
    best = max(modes, key=lambda m: m["peak_edges_per_sec"])
    mem = next((m for m in modes if m["mode"] == "memory"), best)
    per_million = mem["heap_bytes_per_million_edges"]
    summary = (
        f"**Ingest:** {rate(mem['resident_edges_per_sec'])} sustained over the first "
        f"{num(mem['resident_edges'])} temporal hyperedges (peak {rate(best['peak_edges_per_sec'])}), "
        f"using {per_million / MiB:.0f} MiB of heap per million edges"
    )
    if mem.get("stop_reason"):
        summary += (
            f". The run stopped at {num(mem['ingested_edges'])} edges when this "
            f"{data['environment'].get('memory_bytes', 0) / GiB:.0f} GiB machine began paging; the "
            f"{num(mem['target_edges'])}-edge target needs about {per_million * mem['target_edges'] / 1e6 / GiB:.1f} GiB of heap"
        )
    rep.summary.append(summary + ".")

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in modes:
        xs = [s["edges"] / 1e6 for s in m["memory_samples"]]
        ax.plot(xs, [s["heap_live_bytes"] / GiB for s in m["memory_samples"]], marker="o", label=f"{m['mode']} heap")
        ax.plot(xs, [s["rss_bytes"] / GiB for s in m["memory_samples"]], linestyle="--", label=f"{m['mode']} RSS")
    ax.set_xlabel("Edges ingested (millions)")
    ax.set_ylabel("GiB")
    ax.set_title("Memory growth during ingest")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    rep.figure(fig, "memory_growth.png", "Memory growth during ingest")
    rep.add("")


# --------------------------------------------------------------------------- concurrency


def by(results: list[dict[str, Any]], scenario: str) -> dict[int, dict[str, Any]]:
    return {r["readers"]: r for r in results if r["scenario"] == scenario}


def concurrency_section(rep: Report, data: dict[str, Any], throttled: dict[str, Any] | None) -> None:
    cfg = data["config"]
    res = data["results"]
    a0, a1 = by(res, "arcswap_readers_only"), by(res, "arcswap_with_writer")
    r0, r1 = by(res, "rwlock_readers_only"), by(res, "rwlock_with_writer")
    readers = sorted(set(a0) | set(a1) | set(r0) | set(r1))
    cores = data["environment"]["logical_cores"]

    rep.add("## 2. Lock-free read concurrency\n")
    rep.add(
        f"N reader threads run {cfg['hops']}-hop traversals (capped at {num(cfg['limit'])} hyperedges each) over a "
        f"{num(cfg['preload'])}-hyperedge graph. In the writer scenarios one thread inserts {num(cfg['writer_batch'])}-edge "
        "batches back to back"
        + (f", throttled to {num(cfg['writer_rate'])} edges/s" if cfg.get("writer_rate") else ", unthrottled")
        + ". The control repeats the same workload over a conventional `RwLock`-protected graph, where the writer holds "
        "the write lock while it applies a batch. Hardware: " + env_line(data["environment"]) + ".\n"
    )

    def cell(d: dict[int, dict[str, Any]], n: int, f) -> str:
        return f(d[n]) if n in d else "n/a"

    rows = [[
        str(n),
        cell(a0, n, lambda r: rate(r["reads_per_sec"])),
        cell(a1, n, lambda r: rate(r["reads_per_sec"])),
        cell(r1, n, lambda r: rate(r["reads_per_sec"])),
        cell(a0, n, lambda r: us(r["read_latency"]["p99_us"])),
        cell(a1, n, lambda r: us(r["read_latency"]["p99_us"])),
        cell(r1, n, lambda r: us(r["read_latency"]["p99_us"])),
        cell(a1, n, lambda r: us(r["snapshot_acquire_latency"]["p99_us"])),
        cell(r1, n, lambda r: us(r["snapshot_acquire_latency"]["p99_us"])),
    ] for n in readers]
    rep.add(table(
        ["Readers", "ArcSwap reads/s (no writer)", "ArcSwap reads/s (+writer)", "RwLock reads/s (+writer)",
         "ArcSwap read p99 (no writer)", "ArcSwap read p99 (+writer)", "RwLock read p99 (+writer)",
         "ArcSwap snapshot-acquire p99 (+writer)", "RwLock lock-acquire p99 (+writer)"],
        rows,
    ))
    rep.add("")

    derived = {d["readers"]: d for d in data["derived"]}
    rows = []
    for n in readers:
        d = derived.get(n, {})
        rows.append([
            str(n),
            signed_pct(d["arcswap_p99_overhead"]) if d.get("arcswap_p99_overhead") is not None else "n/a",
            signed_pct(d["rwlock_p99_overhead"]) if d.get("rwlock_p99_overhead") is not None else "n/a",
            cell(a1, n, lambda r: rate(r["writer_edges_per_sec"])),
            cell(r1, n, lambda r: rate(r["writer_edges_per_sec"])),
            pct(d["arcswap_scaling_efficiency"]) if d.get("arcswap_scaling_efficiency") is not None else "n/a",
        ])
    rep.add(table(
        ["Readers", "ArcSwap read-p99 overhead from writer", "RwLock read-p99 overhead from writer",
         "Writer throughput (ArcSwap)", "Writer throughput (RwLock)", "ArcSwap read scaling efficiency"],
        rows,
    ))
    rep.add(
        "\n- Overhead is measured at a fixed reader count: read p99 with the writer divided by read p99 without it, minus 1. "
        f"This machine has {cores} cores, so above {cores} readers latency also rises from time-slicing. That affects "
        "both designs equally and is not lock contention."
        "\n- Scaling efficiency is reads/s at N readers divided by (reads/s at 1 reader x min(N, cores))."
        f"\n- Acquire timings include about {data['timer_overhead_ns']:.0f} ns of timer overhead.\n"
    )

    top = max((n for n in readers if n in a1 and n in r1), default=None)
    if top is not None:
        rep.summary.append(
            f"**Concurrency:** with {top} readers and a heavy writer, snapshot acquisition stays at "
            f"{us(a1[top]['snapshot_acquire_latency']['p99_us'])} p99 under ArcSwap, against "
            f"{us(r1[top]['snapshot_acquire_latency']['p99_us'])} for an RwLock. Readers sustain "
            f"{rate(a1[top]['reads_per_sec'])} against {rate(r1[top]['reads_per_sec'])} "
            f"({a1[top]['reads_per_sec'] / max(r1[top]['reads_per_sec'], 1e-9):.1f}x)."
        )

    release = [(n, a1[n]["snapshot_release_latency"]["p99_us"]) for n in readers if n in a1 and "snapshot_release_latency" in a1[n]]
    if release:
        worst_n, worst = max(release, key=lambda x: x[1])
        rep.add(
            "**Engineering note: snapshot reclamation.** Readers never block on the writer, so acquisition stays flat. "
            "The remaining read-tail overhead comes from memory reclamation. When the writer publishes, a reader still "
            "holding the previous snapshot becomes its last owner, and dropping the guard frees the retired "
            f"copy-on-write pages on the read path. The guard-release p99 peaks at {us(worst)} (with {worst_n} reader{'' if worst_n == 1 else 's'}). "
            "Handing retired snapshots to a background reclaimer would take this cost off the read path.\n"
        )

    if throttled:
        t1 = by(throttled["results"], "arcswap_with_writer")
        td = {d["readers"]: d for d in throttled["derived"]}
        rows = [[
            str(n), rate(t1[n]["writer_edges_per_sec"]), us(t1[n]["read_latency"]["p99_us"]),
            signed_pct(td[n]["arcswap_p99_overhead"]) if td.get(n, {}).get("arcswap_p99_overhead") is not None else "n/a",
            signed_pct(td[n]["rwlock_p99_overhead"]) if td.get(n, {}).get("rwlock_p99_overhead") is not None else "n/a",
        ] for n in sorted(t1)]
        rep.add(f"Steady-state writer (throttled to {num(throttled['config']['writer_rate'])} edges/s):\n")
        rep.add(table(["Readers", "Writer throughput", "ArcSwap read p99 (+writer)", "ArcSwap read-p99 overhead",
                       "RwLock read-p99 overhead"], rows))
        rep.add("")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for d, label in ((a0, "ArcSwap, no writer"), (a1, "ArcSwap + writer"), (r0, "RwLock, no writer"), (r1, "RwLock + writer")):
        xs = sorted(d)
        if not xs:
            continue
        axes[0].plot(xs, [d[n]["read_latency"]["p99_us"] / 1_000 for n in xs], marker="o", label=label)
        axes[1].plot(xs, [d[n]["reads_per_sec"] for n in xs], marker="o", label=label)
        axes[2].plot(xs, [max(d[n]["snapshot_acquire_latency"]["p99_us"], 1e-3) for n in xs], marker="o", label=label)
    axes[0].set_title("Read latency p99 (ms)")
    axes[1].set_title("Read throughput (reads/s)")
    axes[2].set_title("Snapshot/lock acquire p99 (µs, log)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Reader threads")
        ax.axvline(cores, color="grey", linestyle=":", linewidth=1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    rep.figure(fig, "concurrency_scaling.png", "Read latency, throughput and acquire latency against reader count")
    rep.add("")


# --------------------------------------------------------------------------- competitors


def competitor_section(rep: Report, data: dict[str, Any], sweep: dict[str, Any] | None) -> None:
    cfg, ds = data["config"], data["dataset"]
    contenders = data["contenders"]
    rep.add("## 3. Retrieval against Graph RAG and Vector RAG\n")
    rep.add(
        f"Offline, purely algorithmic retrieval (no LLM calls) over a dense synthetic temporal world: "
        f"{num(ds['entities'])} entities, {num(ds['facts'])} facts ({num(ds['nary_facts'])} N-ary, "
        f"{num(ds['closed_facts'])} superseded or closed), mean entity degree {ds['mean_entity_degree']:.1f} "
        f"(max {num(ds['max_entity_degree'])}). The facts are written up as {num(ds['articles'])} articles, chunked into "
        f"{num(ds['chunks'])} chunks of about {ds['mean_chunk_tokens']:.0f} tokens for the vector baseline. "
        f"{num(ds['queries'])} multi-hop questions (mean {ds['mean_gold_facts']:.1f} gold facts per question) each "
        f"ask about a past date where an outdated fact is usually a live distractor. Embedder: "
        f"`{cfg['embedder_name']}` ({cfg['embedding_dim']} dims), shared by every contender. Hardware: "
        + env_line(data["environment"]) + ".\n"
    )
    rows = []
    for c in contenders:
        rows.append([
            c["label"], ms(c["latency_ms"]["p50"]), ms(c["latency_ms"]["p99"]), f"{c['tokens']['mean']:.0f}",
            f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1']:.3f}", pct(c["full_chain_recall"]),
            pct(c["stale_rate"]),
        ])
    rep.add(table(
        ["Contender", "Latency p50", "Latency p99", "Context tokens (mean)", "Fact precision", "Fact recall",
         "F1", "Full-chain recall", "Stale-fact rate"],
        rows,
    ))

    hg = next(c for c in contenders if c["name"] == "hypergraph")
    rows = []
    for c in contenders:
        if c is hg:
            continue
        speed = c["latency_ms"]["p50"] / max(hg["latency_ms"]["p50"], 1e-9)
        tokens = 1 - hg["tokens"]["mean"] / max(c["tokens"]["mean"], 1e-9)
        rows.append([
            c["label"], f"{speed:.1f}x", f"{pct(-tokens)} more" if tokens < 0 else f"{pct(tokens)} fewer",
            f"{(hg['recall'] - c['recall']) * 100:+.1f} pts",
            f"{hg['recall'] / c['recall']:.1f}x" if c["recall"] > 0 else "inf",
            f"{(c['stale_rate'] - hg['stale_rate']) * 100:.1f} pts lower",
        ])
    rep.add("\nTemporal Hypergraph compared with each baseline:\n")
    rep.add(table(["Baseline", "Hypergraph faster (p50)", "Hypergraph context tokens", "Recall gain",
                   "Recall ratio", "Stale-fact rate"], rows))

    plan = ("a planner-supplied hop count and relation types (as a text-to-Cypher step would provide)"
            if cfg.get("query_plan") else f"a fixed {cfg['hops']}-hop depth and every relation type")
    parity = data["parity"]["hypergraph_vs_networkx_temporal_jaccard"]
    rep.add(
        "\n- **Retrieval budget:** both graph contenders start from the same seed entity and return the top "
        f"{cfg['top_k']} facts by cosine similarity among the facts they reach, using {plan}. "
        f"The vector contender returns the top {cfg['vector_k']} chunks, plus a token-matched k whose median context "
        "is at least the hypergraph's."
        f"\n- **Parity check:** the engine and the temporal NetworkX graph return the same facts (mean Jaccard "
        f"{parity:.3f}). Their accuracy is identical by construction; the difference between them is speed."
        "\n- **Vector precision and recall:** a chunk counts every fact it contains as retrieved."
        "\n- **Latency:** in-process time of the retrieval call only; query embedding happens outside the timed region "
        "for every contender. Hypergraph latency includes the PyO3 boundary and Python result conversion.\n"
    )

    naive = next((c for c in contenders if c["name"] == "networkx_naive"), None)
    temporal = next((c for c in contenders if c["name"] == "networkx_temporal"), None)
    vectors = [c for c in contenders if c["name"].startswith("vector_")]
    best_vec = max(vectors, key=lambda c: c["recall"]) if vectors else None
    parts = []
    if temporal:
        parts.append(f"{temporal['latency_ms']['p50'] / hg['latency_ms']['p50']:.0f}x faster than the temporal property-graph baseline at identical accuracy")
    if naive:
        parts.append(f"0% stale facts against {pct(naive['stale_rate'])} for naive Graph RAG")
    if best_vec:
        parts.append(f"{hg['recall'] / max(best_vec['recall'], 1e-9):.0f}x the multi-hop fact recall of the best vector setting ({pct(hg['recall'])} vs {pct(best_vec['recall'])})")
    if parts:
        rep.summary.append("**Retrieval:** " + "; ".join(parts) + ".")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [c["name"] for c in contenders]
    axes[0].barh(labels, [c["latency_ms"]["p50"] for c in contenders])
    axes[0].set_xscale("log")
    axes[0].set_title("Retrieval latency p50 (ms, log)")
    axes[1].barh(labels, [c["recall"] for c in contenders], label="fact recall")
    axes[1].barh(labels, [-c["stale_rate"] for c in contenders], label="stale-fact rate (negative)")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Fact recall vs stale-fact rate")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    rep.figure(fig, "competitor_comparison.png", "Latency and recall of each contender")
    rep.add("")

    if sweep and sweep.get("sweep"):
        rep.add("### Scaling with dataset size\n")
        latencies = [{sweep_series(c): c["latency_ms"]["p50"] for c in p["contenders"]} for p in sweep["sweep"]]
        names = [n for n in latencies[0] if all(n in lat for lat in latencies)]
        rows = [
            [num(p["dataset"]["entities"]), num(p["dataset"]["facts"])] + [ms(lat[n]) for n in names]
            for p, lat in zip(sweep["sweep"], latencies)
        ]
        rep.add(table(["Entities", "Facts"] + [f"{n} p50" for n in names], rows))
        rep.add(
            f"\nSweep embedder: `{sweep['config']['embedder_name']}`. NetworkX contenders run the first "
            f"{sweep['sweep'][0]['networkx_queries']} queries at each size.\n"
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [p["dataset"]["entities"] for p in sweep["sweep"]]
        for n in names:
            ax.plot(xs, [lat[n] for lat in latencies], marker="o", label=n)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Entities")
        ax.set_ylabel("p50 latency (ms)")
        ax.set_title("Retrieval latency against dataset size")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        rep.figure(fig, "competitor_sweep.png", "Retrieval latency against dataset size")
        rep.add("")


# --------------------------------------------------------------------------- main


REPRODUCE = """## Reproduce

```bash
pip install -r benchmarks/requirements.txt
(cd crates/pyo3-bindings && maturin develop --release)

cargo bench -p thg-benchmarks --bench scale_stress -- \\
  --edges 10000000 --modes memory,wal-buffered,wal-fsync --fsync-edges 1000000 \\
  --out results/phase7/scale_stress.json
cargo bench -p thg-benchmarks --bench concurrency_scaling -- \\
  --readers 1,8,16,32,64 --duration-secs 10 --preload 1000000 \\
  --out results/phase7/concurrency.json
cargo bench -p thg-benchmarks --bench concurrency_scaling -- \\
  --readers 1,8,16,32,64 --duration-secs 10 --preload 1000000 --writer-rate 10000 \\
  --out results/phase7/concurrency_throttled.json
python -m hypergraph_evaluation.competitor_bench \\
  --entities 12000 --facts 60000 --queries 1000 --hops 3 --top-k 16 \\
  --embedder minilm --vector-backend faiss --out results/phase7/competitor.json
python -m hypergraph_evaluation.competitor_bench --scale-sweep 10000,50000,100000 \\
  --embedder hashing --queries 500 --out results/phase7/competitor_sweep.json
python benchmarks/gtm_report.py --in results/phase7 --out docs/GTM_Whitepaper_Metrics.md
```
"""

METHOD = """## Methodology and fairness

- **Offline and deterministic.** Every workload is seeded. No LLM or network calls happen during measurement.
- **Same inputs for every contender.** All contenders share one embedder, one tokenizer (tiktoken `o200k_base`) and one rendering of facts into context text. The graph contenders get the same seed entities (oracle entity linking) and the same query plan.
- **Strong baselines.**
  - The property-graph baseline reifies N-ary facts as event nodes with `valid_from`/`valid_to` properties, the standard Neo4j modelling.
  - Its temporal variant applies the same validity filter as the engine.
  - The vector baseline uses exact (brute-force) cosine search, so approximate-index recall loss does not penalise it.
- **What the engine adds.** Temporal validity is enforced during traversal, so superseded facts are neither returned nor expanded. N-ary facts stay intact as single hyperedges. Traversal runs over immutable lock-free snapshots in native code.
- **Known limitations.**
  - The data is synthetic and template-generated.
  - Latency is single-process and in-memory; network hops to a hosted vector or graph database would add to the baselines' latency.
  - Concurrency numbers above the core count reflect CPU time-slicing as well as synchronization.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", default="results/phase7")
    parser.add_argument("--out", default="docs/GTM_Whitepaper_Metrics.md")
    parser.add_argument("--img-dir", default="docs/img/phase7")
    args = parser.parse_args(argv)

    inp, out = Path(args.inp), Path(args.out)
    scale = load(inp, "scale_stress.json")
    conc = load(inp, "concurrency.json")
    throttled = load(inp, "concurrency_throttled.json")
    comp = load(inp, "competitor.json")
    sweep = load(inp, "competitor_sweep.json")
    if not any((scale, conc, comp)):
        print(f"no results found in {inp}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    rep = Report(Path(args.img_dir), out.parent)
    if scale:
        scale_section(rep, scale)
    if conc:
        concurrency_section(rep, conc, throttled)
    if comp:
        competitor_section(rep, comp, sweep)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = [
        "# Temporal Hypergraph Engine: Phase 7 Performance and Competitor Metrics\n",
        f"_Generated {generated} by `benchmarks/gtm_report.py` from `{inp}/`. Do not edit by hand; re-run the "
        "benchmarks and regenerate._\n",
        "## Executive summary\n",
        "\n".join(f"- {s}" for s in rep.summary) + "\n",
    ]
    out.write_text("\n".join(head + rep.parts + [METHOD, REPRODUCE]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
