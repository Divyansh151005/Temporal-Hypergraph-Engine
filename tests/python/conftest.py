import pytest

import hypergraph_engine_native as native


@pytest.fixture
def tmp_engine(tmp_path):
    """Disk-backed engine (WAL + mmap KV segments) in a temporary directory."""
    return native.NativeHypergraphEngine(str(tmp_path / "db"), sync="buffered")


@pytest.fixture
def mem_engine():
    """Volatile engine and KV store."""
    return native.NativeHypergraphEngine.in_memory()
