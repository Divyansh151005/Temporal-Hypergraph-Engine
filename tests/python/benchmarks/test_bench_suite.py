"""Benchmark building blocks plus a small end-to-end run (numpy baseline, scripted LLM)."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from hypergraph_client import HashingEmbedder
from hypergraph_evaluation.llm import LLM, LLMConfig

import run_benchmark
from baseline_vector import Chunk, VectorBaseline, chunk_document
from hypergraph_runner import EntityLinker, Fact, NativeRunner, graph_edges, render_fact
from metrics import Entity, Question, TokenCounter, coverage, exact_verdict, percentiles, reduction, stale_contaminated

COUNTER = TokenCounter()
EMBED = HashingEmbedder(64).embed

ADA = Entity("Ada Lovelace", ("Lovelace",))
BO = Entity("Bo Chen", ("Chen",))
ACME = Entity("Acme Corp", ("Acme",))
QUESTION = Question("q", "d", "As of 2020, who was CEO of Acme Corp?", 1577836800, 1, ADA, BO, [[ADA, ACME]], ["Acme Corp"])


def test_chunks_respect_the_token_budget():
    text = "\n\n".join(" ".join(f"Sentence {i}-{j} talks about Acme Corp." for j in range(12)) for i in range(4))
    text += "\n\n" + "word " * 400
    chunks = chunk_document("d", text, COUNTER, 64)
    assert all(c.tokens <= 64 for c in chunks)
    assert len(chunks) > 8
    assert "Sentence 0-0" in chunks[0].text


def test_linker_prefers_longest_phrase_and_aliases():
    linker = EntityLinker([ADA, ACME, Entity("Acme Corp Labs"), Entity("Paris")])
    assert linker.link("Did Acme Corp Labs hire Lovelace?") == ["Acme Corp Labs", "Ada Lovelace"]
    assert linker.link("Acme's CEO") == ["Acme Corp"]
    assert linker.link("nothing here") == []


def test_render_fact():
    fact = Fact("EXECUTIVE_ROLE", ("Ada Lovelace", "Acme Corp"), 1577836800, None, "Ada Lovelace became CEO of Acme Corp.")
    assert render_fact(fact) == (
        "[2020-01-01 → present] EXECUTIVE_ROLE(Ada Lovelace, Acme Corp): Ada Lovelace became CEO of Acme Corp."
    )
    assert render_fact(Fact("HQ", ("A", "B"), None, 1609459200, "s")).startswith("[? → 2021-01-01]")


def test_coverage_staleness_and_grading():
    assert coverage("Lovelace runs Acme.", QUESTION) == 1.0
    assert coverage("Lovelace runs things.", QUESTION) == 0.0
    assert stale_contaminated("Bo Chen ran Acme Corp", QUESTION)
    assert not stale_contaminated("Chenille", QUESTION)
    assert exact_verdict("ada lovelace", QUESTION) == "correct"
    assert exact_verdict("Chen", QUESTION) == "stale"
    assert exact_verdict("UNKNOWN", QUESTION) == "abstained"
    assert exact_verdict("Grace Hopper", QUESTION) is None
    assert reduction(0.5, 0.1) == pytest.approx(0.8) and reduction(0.0, 0.1) is None
    stats = percentiles([1_000_000, 2_000_000, 3_000_000])
    assert stats["p50_ms"] == pytest.approx(2.0) and stats["n"] == 3


@pytest.mark.parametrize("backend", ["numpy", "qdrant"])
def test_vector_backends_agree(backend):
    chunks = [Chunk("d", t, 1) for t in ("Acme Corp hired a CEO", "Paris is a city", "weather report", "Acme earnings")]
    index = VectorBaseline(chunks, EMBED, backend=backend)
    hits = index.search(EMBED("Who is the CEO of Acme Corp"), 2)
    assert hits[0].text == "Acme Corp hired a CEO" and len(hits) == 2
    assert len(index.search(EMBED("x"), 10)) == 4


def test_native_runner_traverses_at_as_of():
    runner = NativeRunner(EMBED, 64)
    graph = {
        "nodes": [{"id": "a", "label": "Ada"}, {"id": "b", "label": "Bo"}, {"id": "c", "label": "Acme"}, {"id": "p", "label": "Paris"}],
        "hyperedges": [
            {"event_type": "CEO", "participants": ["a", "c"], "valid_from": 100, "valid_to": 200, "summary": "Ada ran Acme."},
            {"event_type": "CEO", "participants": ["b", "c"], "valid_from": 200, "valid_to": None, "summary": "Bo runs Acme."},
            {"event_type": "HQ", "participants": ["c", "p"], "valid_from": None, "valid_to": None, "summary": "Acme is in Paris."},
        ],
    }
    assert runner.ingest(graph_edges(graph)) == 3
    facts, elapsed = runner.retrieve(["Paris"], 150, EMBED("who ran Acme"), 2, 8)
    assert {f.summary for f in facts} == {"Ada ran Acme.", "Acme is in Paris."} and elapsed > 0
    facts, _ = runner.retrieve(["Paris"], 250, EMBED("who ran Acme"), 1, 8)
    assert [f.summary for f in facts] == ["Acme is in Paris."]


def test_point_events_are_widened_to_a_day_and_ingest():
    graph = {
        "nodes": [{"id": "a", "label": "Acme"}, {"id": "b", "label": "Bolt"}],
        "hyperedges": [{"event_type": "ACQUISITION", "participants": ["a", "b"], "valid_from": 86_400,
                        "valid_to": 86_400, "summary": "Acme bought Bolt."}],
    }
    edges = graph_edges(graph)
    assert (edges[0].valid_from, edges[0].valid_to) == (86_400, 2 * 86_400)
    runner = NativeRunner(EMBED, 64)
    assert runner.ingest(edges) == 1
    facts, _ = runner.retrieve(["Bolt"], 86_400 + 3_600, EMBED("who bought Bolt"), 1, 8)
    assert [f.summary for f in facts] == ["Acme bought Bolt."]


class ScriptedCompletions:
    """Answers with the first person-like name in the context, mimicking a reader that trusts its context."""

    def create(self, **kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        if "grade" in system:
            content = json.dumps({"verdict": "wrong"})
        else:
            context = user.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
            content = json.dumps({"answer": context.split(" ")[3] if context.strip() else "UNKNOWN"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_end_to_end_offline_run(tmp_path, monkeypatch):
    client = SimpleNamespace(chat=SimpleNamespace(completions=ScriptedCompletions()))
    roles = []

    def for_role(role, spec=None):
        roles.append((role, spec))
        return LLM(LLMConfig(model=f"{role}-model", cache_dir=None), client)

    monkeypatch.setattr(run_benchmark, "LLM", SimpleNamespace(for_role=for_role))
    code = run_benchmark.main([
        "--offline-corpus", "--n", "12", "--baseline-backend", "numpy", "--warmup", "2", "--repeats", "1",
        "--workers", "2", "--out-dir", str(tmp_path), "--judge-llm", "groq:qwen/qwen3.8-27b",
    ])
    assert code == 0
    assert roles == [("answer", None), ("judge", "groq:qwen/qwen3.8-27b")]
    (out,) = tmp_path.iterdir()
    results = json.loads((out / "results.json").read_text())
    assert results["corpus"]["documents"] == 12 and results["corpus"]["questions"] == 24
    hg, base = results["contenders"]["hypergraph"], results["contenders"]["baseline_fixed_k"]
    assert hg["stale_contamination"] <= base["stale_contamination"]
    assert hg["gold_fact_coverage"] >= base["gold_fact_coverage"]
    assert hg["hallucination_rate"] is not None and sum(hg[f"{v}_rate"] for v in ("correct", "stale", "wrong", "abstained")) == pytest.approx(1)
    assert results["linking"]["seed_recall"] == 1.0
    llm_info = results["config"]["llm"]
    assert llm_info["answer_chain"] == "openai:answer-model" and llm_info["failed"] == 0
    assert sum(llm_info["answered_by"].values()) == 48
    assert "Answers served by openai:answer-model (48)" in (out / "summary.md").read_text()
    assert results["latency"]["hypergraph_pyo3"]["n"] == 24
    rows = list(csv.DictReader((out / "results.csv").open()))
    assert len(rows) == 72 and {r["contender"] for r in rows} == {"hypergraph", "baseline_fixed_k", "baseline_recall_matched"}
    assert "| Mean context tokens |" in (out / "summary.md").read_text()

    assert run_benchmark.main([
        "--offline-corpus", "--n", "4", "--baseline-backend", "numpy", "--skip-llm", "--warmup", "0",
        "--repeats", "1", "--k", "1", "--out-dir", str(tmp_path / "strict"), "--assert-targets",
    ]) == 1
    assert np.isfinite(results["reductions"]["token_fixed_k"])
