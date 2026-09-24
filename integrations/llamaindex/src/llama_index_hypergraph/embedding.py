"""Accept either a LlamaIndex ``BaseEmbedding`` or a LangChain-style embedder."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hypergraph_client import Embedder


@runtime_checkable
class LlamaEmbedding(Protocol):
    def get_query_embedding(self, query: str) -> list[float]: ...

    def get_text_embedding_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]: ...


class EmbedderAdapter:
    """Presents a LlamaIndex embedding model as an ``Embedder``."""

    def __init__(self, model: LlamaEmbedding) -> None:
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.get_text_embedding_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.get_query_embedding(text)


def as_embedder(model: LlamaEmbedding | Embedder | None) -> Embedder | None:
    if model is None or isinstance(model, Embedder):
        return model
    if isinstance(model, LlamaEmbedding):
        return EmbedderAdapter(model)
    raise TypeError(f"unsupported embedding model {type(model).__name__}")
