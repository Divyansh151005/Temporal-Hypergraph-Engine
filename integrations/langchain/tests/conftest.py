import pytest
from hypergraph_client import HashingEmbedder
from hypergraph_client.testing import FakeHypergraphClient
from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 64) -> None:
        self.inner = HashingEmbedder(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


@pytest.fixture
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture
def fake_client() -> FakeHypergraphClient:
    return FakeHypergraphClient()
