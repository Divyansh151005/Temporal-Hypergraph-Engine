"""Synthetic generator, schemas, temporal normalization and the LLM router (fake client, no network)."""

import json
import random
from types import SimpleNamespace

import openai
import pytest

from hypergraph_evaluation import schema
from hypergraph_evaluation.extract_graph import extract, normalize
from hypergraph_evaluation.llm import LLM, LLMConfig, LLMError
from hypergraph_evaluation.synthetic_generator import (
    DOMAINS,
    generate_dataset,
    generate_record,
    inject_noise,
    load_records,
    missing_mentions,
    write_records,
)
from hypergraph_evaluation.temporal import add_months, to_epoch, to_iso


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply(kwargs) if callable(reply) else reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_llm(replies, tmp_path=None):
    completions = FakeCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig(cache_dir=tmp_path, max_retries=3)
    return LLM(config, client), completions


def test_temporal_normalization():
    assert to_epoch(None) is None and to_epoch("null") is None
    assert to_epoch(1696118400) == 1696118400 and to_epoch("1696118400") == 1696118400
    assert to_epoch("2023-10-01T00:00:00Z") == 1696118400
    assert to_epoch("2023-10-01") == 1696118400
    assert to_epoch("2023-10-01T02:00:00+02:00") == 1696118400
    assert to_epoch("2023-10") == 1696118400
    assert to_epoch("2023") == to_epoch("2023-01-01")
    assert to_iso(1696118400) == "2023-10-01T00:00:00Z" and to_iso(None) is None
    assert to_iso(add_months(to_epoch("2023-11-15"), 3)) == "2024-02-01T00:00:00Z"
    assert to_iso(add_months(to_epoch("2023-01-15"), -1)) == "2022-12-01T00:00:00Z"
    with pytest.raises(ValueError):
        to_epoch("last Tuesday")


def test_schemas_are_valid_draft7():
    for name in ("hypergraph", "extraction", "article", "answer", "judge"):
        schema.validator(name)
    assert schema.errors("article", {"article": 3})
    assert schema.graph_errors({"nodes": [{"id": "bad", "label": "x"}], "hyperedges": []})


@pytest.mark.parametrize("domain", DOMAINS)
def test_offline_records_are_consistent(domain):
    for index in range(DOMAINS.index(domain), 40, len(DOMAINS)):
        record = generate_record(index, seed=11)
        assert record["domain"] == domain
        graph = record["ground_truth_graph"]
        nodes = {n["id"]: n for n in graph["nodes"]}
        edges = {e["id"]: e for e in graph["hyperedges"]}
        assert missing_mentions(record["raw_text"], [_entity(n) for n in nodes.values()]) == []
        for edge in edges.values():
            assert 2 <= len(edge["participants"]) <= 5
            assert set(edge["participants"]) <= set(nodes)
            assert edge["valid_to"] is None or edge["valid_from"] is None or edge["valid_from"] < edge["valid_to"]
        assert len(record["qa"]) == 2
        for qa in record["qa"]:
            assert qa["as_of"] < record["reference_time"]
            gold = [edges[i] for i in qa["gold_edge_ids"]]
            assert all(_valid(e, qa["as_of"]) for e in gold), qa
            assert qa["answer_id"] in gold[-1]["participants"]
            stale = [e for e in edges.values() if qa["stale_answer_id"] in e["participants"]
                     and set(e["participants"]) & set(gold[-1]["participants"]) - {qa["answer_id"]}]
            assert stale and not any(_valid(e, qa["as_of"]) for e in stale)
            assert _reachable(qa["seeds"], edges.values(), qa["as_of"], qa["hops"]) >= set(qa["gold_edge_ids"])


def _entity(node):
    return SimpleNamespace(label=node["label"], aliases=tuple(node["aliases"]))


def _valid(edge, at):
    return (edge["valid_from"] is None or edge["valid_from"] <= at) and (edge["valid_to"] is None or at < edge["valid_to"])


def _reachable(seeds, edges, at, hops):
    frontier, seen, found = set(seeds), set(seeds), set()
    for _ in range(hops):
        reached = [e for e in edges if _valid(e, at) and e["id"] not in found and frontier & set(e["participants"])]
        found |= {e["id"] for e in reached}
        frontier = {p for e in reached for p in e["participants"]} - seen
        seen |= frontier
    return found


def test_labels_are_unique_across_a_dataset_and_deterministic(tmp_path):
    records = generate_dataset(200, seed=5, workers=4)
    owners = {}
    for record in records:
        for node in record["ground_truth_graph"]["nodes"]:
            if node["id"].startswith(("person:", "org:", "product:", "team:", "ticket:")):
                assert owners.setdefault(node["label"], record["dataset_id"]) == record["dataset_id"], node
    again = generate_dataset(200, seed=5, workers=1)
    assert again == records
    path = tmp_path / "d.jsonl"
    write_records(records[:3], path, as_array=False)
    assert load_records(path) == records[:3]
    write_records(records[:3], tmp_path / "d.json", as_array=True)
    assert load_records(tmp_path / "d.json") == records[:3]


def test_noise_injection():
    record = generate_record(0, seed=3)
    entities = [_entity(n) for n in record["ground_truth_graph"]["nodes"]]
    noisy = inject_noise(record["raw_text"], entities, 0.3, random.Random(1))
    assert noisy != record["raw_text"]
    assert len(noisy.split("\n\n")) > len(record["raw_text"].split("\n\n"))
    assert inject_noise("text", entities, 0.0, random.Random(1)) == "text"
    noisy_record = generate_record(0, seed=3, noise=0.3)
    assert noisy_record["noise"] == 0.3 and noisy_record["ground_truth_graph"] == record["ground_truth_graph"]


def test_llm_article_retries_until_entities_are_mentioned(tmp_path):
    offline = generate_record(1, seed=9)
    first_draft = json.dumps({"article": "A vague story that names nobody."})
    final = json.dumps({"article": offline["raw_text"]})
    llm, completions = fake_llm([first_draft, final], tmp_path)
    record = generate_record(1, seed=9, llm=llm)
    assert record["raw_text"] == offline["raw_text"]
    assert record["generator"]["attempts"] == 2
    assert "did not mention" in completions.calls[1]["messages"][1]["content"]
    assert completions.calls[0]["response_format"]["type"] == "json_schema"

    llm, _ = fake_llm([first_draft] * 3)
    with pytest.raises(LLMError):
        generate_record(1, seed=9, llm=llm, max_attempts=3)


def test_llm_router_validates_retries_caches_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    request = SimpleNamespace(method="POST", url="https://x", headers={})
    transient = openai.APIConnectionError(request=request)
    llm, completions = fake_llm([transient, "not json", json.dumps({"answer": 4}), json.dumps({"answer": "Ada"})], tmp_path)
    assert llm.complete_json("sys", "q", "answer") == {"answer": "Ada"}
    assert len(completions.calls) == 4
    assert "invalid" in completions.calls[-1]["messages"][-1]["content"]
    # Cached: no further calls.
    assert llm.complete_json("sys", "q", "answer") == {"answer": "Ada"}
    assert len(completions.calls) == 4

    response = SimpleNamespace(status_code=400, headers={}, request=request)
    rejected = openai.BadRequestError("response_format json_schema is not supported", response=response, body=None)
    llm, completions = fake_llm([rejected, json.dumps({"answer": "Bo"})])
    llm.config.cache_dir = None
    assert llm.complete_json("sys", "q", "answer") == {"answer": "Bo"}
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in completions.calls[1]["messages"][0]["content"]

    llm, _ = fake_llm(["{}"] * 4)
    llm.config.cache_dir = None
    with pytest.raises(LLMError):
        llm.complete_json("sys", "q", "answer")


def test_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    config = LLMConfig.from_env()
    assert (config.api_key, config.base_url, config.model) == ("sk-local-dummy", "http://localhost:11434/v1", "qwen2.5:7b")


EXTRACTION = {
    "entities": [
        {"id": "org:nvidia", "name": "Nvidia", "type": "ORGANIZATION", "aliases": []},
        {"id": "loc:texas", "name": "Texas", "type": "LOCATION", "aliases": ["TX"]},
    ],
    "hyperedges": [
        {
            "id": "edge:jv",
            "event_type": "JOINT_VENTURE",
            "summary": "Nvidia builds in Texas.",
            "participants": [{"entity_id": "org:nvidia", "role": "PARTNER"}, {"entity_id": "loc:texas", "role": "LOCATION"}],
            "temporal_boundary": {"valid_from": "2024-03-15T00:00:00Z", "valid_to": "Q3 2025",
                                  "is_ongoing": True, "raw_temporal_expression": "March 15, 2024"},
            "confidence_score": 1.3,
        }
    ],
}


def test_extraction_normalizes_to_graph_shape(tmp_path):
    graph = normalize(EXTRACTION)
    assert graph["nodes"][1] == {"id": "loc:texas", "label": "Texas", "type": "LOCATION", "aliases": ["TX"]}
    edge = graph["hyperedges"][0]
    assert edge["valid_from"] == to_epoch("2024-03-15") and edge["valid_to"] is None
    assert edge["participants"] == ["org:nvidia", "loc:texas"] and edge["confidence"] == 1.0
    assert schema.graph_errors(graph) == []

    llm, completions = fake_llm([json.dumps(EXTRACTION)], tmp_path)
    prediction = extract({"dataset_id": "d", "raw_text": "text", "reference_time": 1700000000}, llm)
    assert prediction["graph"] == graph
    assert "2023-11-14T22:13:20Z" in completions.calls[0]["messages"][0]["content"]


def test_single_provider_waits_out_a_long_retry_delay(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    response = SimpleNamespace(status_code=429, headers={}, request=SimpleNamespace(method="POST", url="https://x"))
    quota = openai.RateLimitError("Rate limit reached. Please try again in 42.5s.", response=response, body=None)
    llm, completions = fake_llm([quota, '{"answer": 7}', '{"answer": "Ada"}'])
    llm.config.cache_dir = None
    assert llm.complete_json("be terse", "who?", "answer") == {"answer": "Ada"}
    assert sleeps[0] >= 42.5 and len(completions.calls) == 3

    unauthorized = openai.AuthenticationError(
        "bad key", response=SimpleNamespace(status_code=401, headers={}, request=response.request), body=None
    )
    llm, _ = fake_llm([unauthorized])
    llm.config.cache_dir = None
    with pytest.raises(openai.AuthenticationError):
        llm.complete_json("s", "u", "answer")


def test_provider_selection(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "x-key")
    monkeypatch.setenv("GROQ_API_KEY", "q-key")
    config = LLMConfig.from_env(default_provider="xai")
    assert (config.provider, config.model, config.api_key, config.base_url) == (
        "xai", "grok-4.7", "x-key", "https://api.x.ai/v1",
    )
    assert LLMConfig.from_env().provider == "openai"
    groq = LLMConfig.from_env("groq")
    assert (groq.model, groq.api_key, groq.base_url) == (
        "qwen/qwen3.8-27b", "q-key", "https://api.groq.com/openai/v1",
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert LLMConfig.from_env(default_provider="xai").provider == "openai"
    with pytest.raises(ValueError, match="unknown provider"):
        LLMConfig.from_env("gemini")
