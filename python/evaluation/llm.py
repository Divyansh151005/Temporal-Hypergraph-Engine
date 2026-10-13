"""Multi-provider LLM router with schema-constrained JSON outputs, failover, retries and a disk cache.

Providers (all through the OpenAI SDK; strict ``json_schema``, falling back to JSON mode):

    xai         https://api.x.ai/v1 (Grok); key: XAI_API_KEY
    groq        https://api.groq.com/openai/v1; key: GROQ_API_KEY
    openai      any OpenAI-compatible server (OpenAI, Ollama, vLLM)
                key: LLM_API_KEY (or OPENAI_API_KEY); base URL: LLM_BASE_URL

Each pipeline role has a provider chain of ``provider:model`` entries, tried in
order (see ``DEFAULT_CHAINS``); override with GENERATOR_LLM, EXTRACT_LLM,
ANSWER_LLM or JUDGE_LLM. Entries without an API key are skipped. Setting
LLM_PROVIDER (and optionally LLM_MODEL) forces a single-provider chain.

A provider that hits its quota (429 with a long retry delay), is overloaded
(503), rejects the credentials or exhausts its retry budget is put on a
cooldown and the request moves to the next provider. Every output is validated
against its Draft-07 schema; invalid outputs are retried on the same provider
with the validation errors fed back. LLM_CACHE_DIR (default .cache/llm; "" to
disable) caches results per provider and model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import openai
from openai import OpenAI
from typing_extensions import assert_never

from .schema import errors as schema_errors
from .schema import load as load_schema

log = logging.getLogger(__name__)

Provider = Literal["xai", "groq", "openai"]
PROVIDERS: tuple[Provider, ...] = ("xai", "groq", "openai")
Role = Literal["generator", "extract", "answer", "judge"]
ROLES: tuple[Role, ...] = ("generator", "extract", "answer", "judge")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS: dict[Provider, str] = {
    "xai": "grok-4.7",
    "groq": "qwen/qwen3.8-27b",
    "openai": "gpt-4o-mini",
}
DEFAULT_MODEL = DEFAULT_MODELS["openai"]
ROLE_ENV: dict[Role, str] = {
    "generator": "GENERATOR_LLM",
    "extract": "EXTRACT_LLM",
    "answer": "ANSWER_LLM",
    "judge": "JUDGE_LLM",
}
DEFAULT_CHAINS: dict[Role, str] = {
    "generator": "xai:grok-4.7,groq:qwen/qwen3.8-27b",
    "extract": "xai:grok-4.7,groq:qwen/qwen3.8-27b",
    "answer": "groq:qwen/qwen3.8-27b,xai:grok-4.7",
    "judge": "groq:qwen/qwen3.8-27b,xai:grok-4.7",
}

# A 429 asking to wait longer than this is treated as quota exhaustion and fails over.
FAILOVER_DELAY = 10.0
DEFAULT_COOLDOWN = 60.0
MAX_RETRY_AFTER = 120.0

_OPENAI_TRANSIENT = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
_RETRY_DELAY = re.compile(
    r"(?:retryDelay'?:\s*'|retry\s+in\s+|try\s+again\s+in\s+)(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE
)
_QUOTA = re.compile(r"quota|resource_exhausted|credits", re.IGNORECASE)
_SAMPLING = re.compile(r"\b(temperature|seed)\b", re.IGNORECASE)


class LLMError(RuntimeError):
    """No provider produced a valid answer within its retry budget."""


class _Transient(Exception):
    """A retryable provider failure (rate limit, timeout, 5xx, connection)."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def exhausted(self) -> bool:
        """Quota or overload: waiting on this provider is pointless when another is available."""
        if self.status == 503:
            return True
        if self.status == 429:
            return self.retry_after is None or self.retry_after > FAILOVER_DELAY or bool(_QUOTA.search(str(self)))
        return False


class _Fatal(Exception):
    """A non-retryable provider failure (credentials, unknown model, rejected request)."""


class _Failover(Exception):
    """This provider gave up; the next one in the chain should take the request."""


def retry_after(error: Exception) -> float | None:
    """Server-suggested wait from a 429 (``retryDelay: 'Ns'``, "retry in Ns" or Groq's "try again in 1m2.5s")."""
    match = _RETRY_DELAY.search(str(error))
    if not match:
        return None
    return min(MAX_RETRY_AFTER, 60 * int(match.group(1) or 0) + float(match.group(2)))


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema as sent in ``json_schema`` response formats (without Draft-07 metadata keys)."""
    return {k: v for k, v in schema.items() if k not in ("$schema", "title")}


def provider_key(provider: Provider) -> str | None:
    match provider:
        case "xai":
            return os.getenv("XAI_API_KEY")
        case "groq":
            return os.getenv("GROQ_API_KEY")
        case "openai":
            key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            # Local OpenAI-compatible servers (Ollama, vLLM) accept any key.
            return key or ("sk-local-dummy" if os.getenv("LLM_BASE_URL") else None)
        case _:
            assert_never(provider)


def provider_base_url(provider: Provider) -> str:
    match provider:
        case "xai":
            return XAI_BASE_URL
        case "groq":
            return GROQ_BASE_URL
        case "openai":
            return os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        case _:
            assert_never(provider)


def _provider(name: str) -> Provider:
    name = name.strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {PROVIDERS}")
    return name  # type: ignore[return-value]


def parse_chain(spec: str) -> list[tuple[Provider, str]]:
    """Parses ``provider:model[,provider:model...]``; a bare provider uses its default model."""
    entries: list[tuple[Provider, str]] = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, model = item.partition(":")
        provider = _provider(name)
        entries.append((provider, model.strip() or DEFAULT_MODELS[provider]))
    if not entries:
        raise ValueError(f"empty LLM chain spec {spec!r}")
    return entries


def _cache_dir_from_env() -> Path | None:
    cache = os.getenv("LLM_CACHE_DIR", ".cache/llm")
    return Path(cache) if cache else None


@dataclass
class LLMConfig:
    provider: Provider = "openai"
    api_key: str = "sk-local-dummy"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_retries: int = 4
    timeout: float = 120.0
    max_tokens: int = 8192
    cache_dir: Path | None = field(default_factory=lambda: Path(".cache/llm"))

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @classmethod
    def from_env(cls, provider: str | None = None, *, default_provider: Provider = "openai") -> LLMConfig:
        """A single provider: ``provider`` overrides ``$LLM_PROVIDER``, which overrides ``default_provider``."""
        chosen = _provider(provider or os.getenv("LLM_PROVIDER") or default_provider)
        if chosen == "openai":
            api_key = os.getenv("LLM_API_KEY", "sk-local-dummy")
        else:
            api_key = provider_key(chosen) or os.getenv("LLM_API_KEY", "")
        return cls(
            provider=chosen,
            api_key=api_key,
            base_url=provider_base_url(chosen),
            model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[chosen],
            cache_dir=_cache_dir_from_env(),
        )

    @classmethod
    def chain_for_role(cls, role: Role, spec: str | None = None) -> list[LLMConfig]:
        """The provider chain for ``role``: ``spec``, else ``$LLM_PROVIDER``, else the role's env var or default."""
        if spec is None:
            if os.getenv("LLM_PROVIDER"):
                return [cls.from_env()]
            spec = os.getenv(ROLE_ENV[role]) or DEFAULT_CHAINS[role]
        configs, skipped = [], []
        for provider, model in parse_chain(spec):
            key = provider_key(provider)
            if not key:
                skipped.append(f"{provider}:{model}")
                continue
            configs.append(
                cls(provider=provider, api_key=key, base_url=provider_base_url(provider), model=model,
                    cache_dir=_cache_dir_from_env())
            )
        if skipped:
            log.warning("%s chain: skipping %s (no API key in the environment)", role, ", ".join(skipped))
        if not configs:
            raise LLMError(
                f"no provider in the {role} chain {spec!r} has an API key; set XAI_API_KEY, "
                "GROQ_API_KEY or LLM_API_KEY"
            )
        return configs


def _client(config: LLMConfig) -> Any:
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)


@dataclass
class _Backend:
    config: LLMConfig
    client: Any
    # OpenAI-compatible servers without json_schema support fall back to plain JSON mode.
    schema_mode: bool = True
    # Cleared when the model rejects temperature/seed (reasoning models).
    sampling: bool = True
    available_at: float = 0.0

    @property
    def label(self) -> str:
        return self.config.label


class LLM:
    """Structured-output router over a provider chain; thread-safe, so one instance can serve a worker pool."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        *,
        chain: Sequence[LLMConfig] | None = None,
        clients: Sequence[Any | None] | None = None,
    ) -> None:
        configs = list(chain) if chain else [config or LLMConfig.from_env()]
        given = list(clients) if clients is not None else [client] + [None] * (len(configs) - 1)
        if len(given) != len(configs):
            raise ValueError("clients must match the chain length")
        self._backends = [_Backend(c, cl or _client(c)) for c, cl in zip(configs, given)]
        self._lock = threading.Lock()
        self._clock = time.monotonic

    @classmethod
    def for_role(cls, role: Role, spec: str | None = None) -> LLM:
        return cls(chain=LLMConfig.chain_for_role(role, spec))

    @property
    def config(self) -> LLMConfig:
        """The primary provider's configuration."""
        return self._backends[0].config

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def label(self) -> str:
        """The whole chain, e.g. ``xai:grok-4.7,groq:qwen/qwen3.8-27b``."""
        return ",".join(b.label for b in self._backends)

    def _cache_path(self, backend: _Backend, payload: dict[str, Any]) -> Path | None:
        config = backend.config
        if config.cache_dir is None:
            return None
        keyed = {"provider": config.provider, "base_url": config.base_url, "model": config.model, **payload}
        key = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
        return config.cache_dir / key[:2] / f"{key}.json"

    def _ready(self, backend: _Backend) -> bool:
        return self._clock() >= backend.available_at

    def _cool(self, backend: _Backend, seconds: float) -> None:
        with self._lock:
            backend.available_at = max(backend.available_at, self._clock() + seconds)
        log.warning("%s unavailable for %s", backend.label, "the rest of the run" if math.isinf(seconds) else f"{seconds:.0f}s")

    def _drop_sampling(self, backend: _Backend) -> None:
        with self._lock:
            backend.sampling = False
        log.warning("%s rejects temperature/seed; retrying without them", backend.label)

    def _create(self, backend: _Backend, messages: list[dict[str, str]], schema_name: str, schema: dict,
                temperature: float, seed: int | None) -> str:
        kwargs: dict[str, Any] = {
            "model": backend.config.model,
            "messages": messages,
            "max_tokens": backend.config.max_tokens,
        }
        if backend.sampling:
            kwargs["temperature"] = temperature
            if seed is not None:
                kwargs["seed"] = seed
        if backend.schema_mode:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": wire_schema(schema), "strict": True},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
            schema_hint = (
                "Respond with a single JSON object that validates against this JSON Schema "
                f"(Draft-07):\n{json.dumps(schema)}"
            )
            kwargs["messages"] = [{"role": "system", "content": schema_hint}, *messages]
        try:
            response = backend.client.chat.completions.create(**kwargs)
        except _OPENAI_TRANSIENT as error:
            status = getattr(error, "status_code", None)
            raise _Transient(f"{type(error).__name__}: {error}", status, retry_after(error)) from error
        except openai.BadRequestError as error:
            if backend.sampling and _SAMPLING.search(str(error)):
                self._drop_sampling(backend)
                return self._create(backend, messages, schema_name, schema, temperature, seed)
            raise
        except openai.NotFoundError as error:
            if backend.sampling and "requested parameters" in str(error):
                self._drop_sampling(backend)
                return self._create(backend, messages, schema_name, schema, temperature, seed)
            raise _Fatal(f"{type(error).__name__}: {error}") from error
        except openai.APIStatusError as error:
            raise _Fatal(f"{type(error).__name__}: {error}") from error
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def _run(self, backend: _Backend, messages: list[dict[str, str]], schema_name: str, schema: dict,
             temperature: float, seed: int | None, cache_path: Path | None, can_fail_over: bool) -> dict[str, Any]:
        last_problem = "no attempt made"
        for attempt in range(backend.config.max_retries + 1):
            try:
                content = self._create(backend, messages, schema_name, schema, temperature, seed)
            except openai.BadRequestError as error:
                message = str(error).lower()
                if backend.schema_mode and ("response_format" in message or "json_schema" in message):
                    with self._lock:
                        backend.schema_mode = False
                    log.warning("%s rejected json_schema mode; falling back to JSON mode", backend.label)
                    continue
                if len(self._backends) > 1:
                    raise _Failover(f"BadRequestError: {error}") from error
                raise
            except _Fatal as error:
                if len(self._backends) > 1:
                    self._cool(backend, math.inf)
                    raise _Failover(str(error)) from error
                raise error.__cause__ or error
            except _Transient as error:
                last_problem = str(error)
                if can_fail_over and error.exhausted:
                    self._cool(backend, error.retry_after or DEFAULT_COOLDOWN)
                    raise _Failover(last_problem) from error
                if attempt < backend.config.max_retries:
                    backoff = min(30.0, 2**attempt)
                    time.sleep(max(backoff, error.retry_after or 0.0) + random.random())
                continue
            try:
                result = json.loads(content)
            except json.JSONDecodeError as error:
                problems = [f"invalid JSON: {error}"]
            else:
                problems = schema_errors(schema_name, result)
                if not problems:
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(result))
                    return result
            last_problem = "; ".join(problems[:5])
            messages = [
                *messages[:2],
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That output is invalid ({last_problem}). Reply again with corrected JSON only.",
                },
            ]
        raise _Failover(last_problem)

    def complete_json_with_source(
        self,
        system: str,
        user: str,
        schema_name: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Like ``complete_json``, also returning the ``provider:model`` that produced the answer."""
        schema = load_schema(schema_name)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def payload(backend: _Backend) -> dict[str, Any]:
            temp = backend.config.temperature if temperature is None else temperature
            return {"messages": messages, "schema": schema, "temperature": temp, "seed": seed}

        paths = [self._cache_path(b, payload(b)) for b in self._backends]
        for backend, path in zip(self._backends, paths):
            if path is not None and path.exists():
                return json.loads(path.read_text()), backend.label

        live = [b for b in self._backends if not math.isinf(b.available_at)]
        if live and not any(self._ready(b) for b in live):
            wait = min(b.available_at for b in live) - self._clock()
            time.sleep(min(MAX_RETRY_AFTER, max(0.0, wait)))

        problems: list[str] = []
        for index, (backend, path) in enumerate(zip(self._backends, paths)):
            if not self._ready(backend):
                problems.append(f"{backend.label}: cooling down")
                continue
            can_fail_over = any(self._ready(b) for b in self._backends[index + 1:])
            temp = backend.config.temperature if temperature is None else temperature
            try:
                result = self._run(backend, messages, schema_name, schema, temp, seed, path, can_fail_over)
            except _Failover as error:
                problems.append(f"{backend.label}: {error}")
                if can_fail_over:
                    log.warning("%s failed (%s); trying the next provider", backend.label, str(error)[:200])
                continue
            return result, backend.label
        raise LLMError(f"{schema_name}: no valid output after retries ({' | '.join(problems)})")

    def complete_json(
        self,
        system: str,
        user: str,
        schema_name: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Asks for a JSON object matching ``schema/<schema_name>.schema.json``."""
        return self.complete_json_with_source(system, user, schema_name, temperature=temperature, seed=seed)[0]
