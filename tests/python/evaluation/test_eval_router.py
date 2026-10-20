"""Provider chains, failover, the circuit breaker and request shapes (fake clients, no network)."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from hypergraph_evaluation.llm import (
    DEFAULT_CHAINS,
    LLM,
    ROLE_ENV,
    LLMConfig,
    LLMError,
    parse_chain,
    retry_after,
)

KEY_VARS = ("XAI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_PROVIDER",
            "LLM_MODEL", *ROLE_ENV.values())
REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_CACHE_DIR", "")
    monkeypatch.setattr("time.sleep", lambda _s: None)


class ChatFake:
    """An OpenAI-compatible client whose replies are scripted."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def status_error(cls, status, message):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


def rate_limited(message="Rate limit reached. Please try again in 42s."):
    return status_error(openai.RateLimitError, 429, message)


def config(provider, model, **extra):
    return LLMConfig(provider=provider, api_key="k", model=model, cache_dir=None, **extra)


GROK = config("xai", "grok-4.7", base_url="https://api.x.ai/v1")
QWEN = config("groq", "qwen/qwen3.8-27b", base_url="https://api.groq.com/openai/v1")
ADA = json.dumps({"answer": "Ada"})


def test_parse_chain():
    assert parse_chain(" xai , groq:qwen/qwen3.8-27b") == [
        ("xai", "grok-4.7"),
        ("groq", "qwen/qwen3.8-27b"),
    ]
    with pytest.raises(ValueError, match="unknown provider"):
        parse_chain("gemini:gemini-flash-latest")
    with pytest.raises(ValueError, match="empty"):
        parse_chain(" , ")
    for spec in DEFAULT_CHAINS.values():
        parse_chain(spec)


def test_default_chains():
    assert DEFAULT_CHAINS["extract"] == DEFAULT_CHAINS["generator"] == "xai:grok-4.7,groq:qwen/qwen3.8-27b"
    assert DEFAULT_CHAINS["answer"] == DEFAULT_CHAINS["judge"] == (
        "groq:qwen/qwen3.8-27b,xai:grok-4.7"
    )


def test_retry_after_formats():
    assert retry_after(Exception("Please try again in 7.5s")) == 7.5
    assert retry_after(Exception("Please try again in 1m2.5s")) == 62.5
    assert retry_after(Exception("Quota exceeded. Please retry in 42s.")) == 42.0
    assert retry_after(Exception("'retryDelay': '30s'")) == 30.0
    assert retry_after(Exception("try again in 9m0s")) == 120.0
    assert retry_after(Exception("slow down")) is None


def test_role_chains_skip_providers_without_keys(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    chain = LLMConfig.chain_for_role("extract")
    assert [c.label for c in chain] == ["groq:qwen/qwen3.8-27b"]
    assert chain[0].base_url == "https://api.groq.com/openai/v1" and chain[0].api_key == "q"
    assert "xai:grok-4.7" in caplog.text

    monkeypatch.setenv("XAI_API_KEY", "x")
    chain = LLMConfig.chain_for_role("generator")
    assert [c.label for c in chain] == ["xai:grok-4.7", "groq:qwen/qwen3.8-27b"]
    assert chain[0].base_url == "https://api.x.ai/v1" and chain[0].api_key == "x"
    assert [c.label for c in LLMConfig.chain_for_role("judge")] == [
        "groq:qwen/qwen3.8-27b", "xai:grok-4.7",
    ]

    monkeypatch.setenv("ANSWER_LLM", "xai:grok-4.6")
    assert [c.label for c in LLMConfig.chain_for_role("answer")] == ["xai:grok-4.6"]
    assert [c.label for c in LLMConfig.chain_for_role("answer", "groq")] == ["groq:qwen/qwen3.8-27b"]

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen3.8-27b")
    assert [c.label for c in LLMConfig.chain_for_role("extract")] == ["groq:qwen/qwen3.8-27b"]


def test_role_chain_without_any_key_fails_fast():
    with pytest.raises(LLMError, match="no provider in the answer chain"):
        LLMConfig.chain_for_role("answer")
    with pytest.raises(LLMError):
        LLMConfig.chain_for_role("extract", "openai:gpt-4o-mini")


def test_grok_quota_fails_over_to_groq_and_the_breaker_skips_until_cooldown():
    grok, qwen = ChatFake(rate_limited("Quota exceeded. Please retry in 42s."), ADA), ChatFake(ADA, ADA)
    llm = LLM(chain=[GROK, QWEN], clients=[grok, qwen])
    now = [1000.0]
    llm._clock = lambda: now[0]

    assert llm.complete_json_with_source("s", "u", "answer") == ({"answer": "Ada"}, "groq:qwen/qwen3.8-27b")
    assert llm.complete_json_with_source("s", "u2", "answer")[1] == "groq:qwen/qwen3.8-27b"
    assert len(grok.calls) == 1 and len(qwen.calls) == 2

    now[0] += 43
    assert llm.complete_json_with_source("s", "u3", "answer") == ({"answer": "Ada"}, "xai:grok-4.7")
    assert len(grok.calls) == 2


def test_groq_rate_limit_fails_over_to_grok():
    qwen, grok = ChatFake(rate_limited("Rate limit reached for model `qwen/qwen3.8-27b` on tokens per "
                                       "minute (TPM). Please try again in 1m12.5s.")), ChatFake(ADA)
    llm = LLM(chain=[QWEN, GROK], clients=[qwen, grok])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "xai:grok-4.7"
    assert grok.calls[0]["model"] == "grok-4.7"


def test_short_rate_limit_retries_in_place():
    grok, qwen = ChatFake(rate_limited("Please try again in 2.5s"), ADA), ChatFake()
    llm = LLM(chain=[GROK, QWEN], clients=[grok, qwen])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "xai:grok-4.7"
    assert len(grok.calls) == 2 and qwen.calls == []


def test_every_provider_failing_reports_each():
    overloaded = status_error(openai.InternalServerError, 503, "high demand")
    unauthorized = status_error(openai.AuthenticationError, 401, "bad key")
    llm = LLM(chain=[GROK, QWEN], clients=[ChatFake(overloaded), ChatFake(unauthorized)])
    with pytest.raises(LLMError) as raised:
        llm.complete_json("s", "u", "answer")
    message = str(raised.value)
    assert "xai:grok-4.7: InternalServerError" in message
    assert "groq:qwen/qwen3.8-27b: AuthenticationError" in message


def test_invalid_output_retries_on_the_same_provider_first():
    grok, qwen = ChatFake('{"answer": 3}', ADA), ChatFake()
    llm = LLM(chain=[GROK, QWEN], clients=[grok, qwen])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "xai:grok-4.7"
    assert len(grok.calls) == 2 and qwen.calls == []
    assert "That output is invalid" in grok.calls[1]["messages"][-1]["content"]


def test_request_shape():
    grok = ChatFake(ADA)
    LLM(GROK, grok).complete_json("sys", "user", "answer", seed=3)
    sent = grok.calls[0]
    assert sent["model"] == "grok-4.7" and sent["seed"] == 3 and sent["max_tokens"] == 8192
    assert "extra_body" not in sent
    response_format = sent["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "answer"
    assert "$schema" not in response_format["json_schema"]["schema"]


def test_groq_without_json_schema_falls_back_to_json_mode():
    unsupported = status_error(openai.BadRequestError, 400,
                               "This model does not support response format `json_schema`.")
    qwen = ChatFake(unsupported, ADA, ADA)
    llm = LLM(chain=[QWEN, GROK], clients=[qwen, ChatFake()])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "groq:qwen/qwen3.8-27b"
    fallback = qwen.calls[1]
    assert fallback["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in fallback["messages"][0]["content"] and fallback["temperature"] == 0.0
    llm.complete_json("s", "u2", "answer")
    assert qwen.calls[2]["response_format"] == {"type": "json_object"}


def test_reasoning_model_rejecting_sampling_parameters_retries_without_them():
    rejected = status_error(openai.BadRequestError, 400, "Model grok-4.7 does not support parameter temperature.")
    grok = ChatFake(rejected, ADA, ADA)
    llm = LLM(chain=[GROK, QWEN], clients=[grok, ChatFake()])
    assert llm.complete_json_with_source("s", "u", "answer", seed=1)[1] == "xai:grok-4.7"
    first, retried = grok.calls
    assert first["temperature"] == 0.0 and first["seed"] == 1
    assert "temperature" not in retried and "seed" not in retried
    assert retried["response_format"]["type"] == "json_schema"
    llm.complete_json("s", "u2", "answer")
    assert "temperature" not in grok.calls[-1]


def test_unknown_model_fails_over_for_the_rest_of_the_run():
    missing = status_error(openai.NotFoundError, 404, "The model `grok-2` does not exist")
    grok, qwen = ChatFake(missing), ChatFake(ADA, ADA)
    llm = LLM(chain=[GROK, QWEN], clients=[grok, qwen])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "groq:qwen/qwen3.8-27b"
    assert llm.complete_json_with_source("s", "u2", "answer")[1] == "groq:qwen/qwen3.8-27b"
    assert len(grok.calls) == 1


def test_cache_is_shared_across_chain_members(tmp_path):
    cached = [LLMConfig(**{**c.__dict__, "cache_dir": tmp_path}) for c in (GROK, QWEN)]
    llm = LLM(chain=cached, clients=[ChatFake(rate_limited()), ChatFake(ADA)])
    assert llm.complete_json_with_source("s", "u", "answer")[1] == "groq:qwen/qwen3.8-27b"

    grok, qwen = ChatFake(), ChatFake()
    again = LLM(chain=cached, clients=[grok, qwen])
    assert again.complete_json_with_source("s", "u", "answer") == ({"answer": "Ada"}, "groq:qwen/qwen3.8-27b")
    assert grok.calls == [] and qwen.calls == []
    assert again.label == "xai:grok-4.7,groq:qwen/qwen3.8-27b"


def test_for_role_builds_real_clients(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    extract = LLM.for_role("extract")
    assert extract.label == "xai:grok-4.7,groq:qwen/qwen3.8-27b"
    grok, qwen = (b.client for b in extract._backends)
    assert str(grok.base_url).startswith("https://api.x.ai/v1") and grok.api_key == "x"
    assert str(qwen.base_url).startswith("https://api.groq.com/openai/v1") and qwen.api_key == "q"
    assert LLM.for_role("answer").label == "groq:qwen/qwen3.8-27b,xai:grok-4.7"
