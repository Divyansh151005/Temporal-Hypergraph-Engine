"""Launching a local ``hypergraph-server`` process (tests, benchmarks, demos)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

SERVER_BINARY = "hypergraph-server"


def find_server_binary() -> Path | None:
    """``$HYPERGRAPH_SERVER_BIN``, else the newest ``target/{release,debug}`` build
    found above the working directory or this package, else ``PATH``."""
    explicit = os.environ.get("HYPERGRAPH_SERVER_BIN")
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    candidates: list[Path] = []
    for start in (Path.cwd(), Path(__file__).resolve()):
        for directory in (start, *start.parents):
            for profile in ("release", "debug"):
                candidate = directory / "target" / profile / SERVER_BINARY
                if candidate.exists():
                    candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    on_path = shutil.which(SERVER_BINARY)
    return Path(on_path) if on_path else None


class LocalServer:
    """A ``hypergraph-server`` on a private Unix socket and data directory.

    Use as a context manager; ``endpoint`` is ready once ``__enter__`` returns.
    """

    def __init__(
        self,
        binary: Path | None = None,
        *,
        api_key: str | None = None,
        embedding_dim: int = 0,
        startup_timeout: float = 15.0,
    ) -> None:
        resolved = binary or find_server_binary()
        if resolved is None:
            raise FileNotFoundError(
                "hypergraph-server not found; run `cargo build --release -p grpc-server` "
                "or set HYPERGRAPH_SERVER_BIN"
            )
        self.binary = resolved
        self.api_key = api_key
        self.embedding_dim = embedding_dim
        self.startup_timeout = startup_timeout
        self._workdir: tempfile.TemporaryDirectory | None = None
        self._process: subprocess.Popen | None = None

    @property
    def socket(self) -> Path:
        assert self._workdir is not None, "server not started"
        return Path(self._workdir.name) / "hg.sock"

    @property
    def endpoint(self) -> str:
        return f"unix://{self.socket}"

    def start(self) -> LocalServer:
        # Unix socket paths are limited to ~104 bytes, so stay under /tmp.
        self._workdir = tempfile.TemporaryDirectory(prefix="hg", dir="/tmp")
        command = [
            str(self.binary),
            "--data-dir",
            str(Path(self._workdir.name) / "data"),
            "--listen",
            self.endpoint,
            "--sync",
            "buffered",
            "--embedding-dim",
            str(self.embedding_dim),
        ]
        env = {k: v for k, v in os.environ.items() if k != "HYPERGRAPH_API_KEY"}
        if self.api_key:
            command += ["--api-key", self.api_key]
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        deadline = time.monotonic() + self.startup_timeout
        while not self.socket.exists():
            if self._process.poll() is not None or time.monotonic() > deadline:
                output = self._process.stdout.read().decode() if self._process.stdout else ""
                self.stop()
                raise RuntimeError(f"hypergraph-server did not start:\n{output}")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.send_signal(signal.SIGINT)
                try:
                    self._process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            if self._process.stdout:
                self._process.stdout.close()
            self._process = None
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None

    def __enter__(self) -> LocalServer:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
