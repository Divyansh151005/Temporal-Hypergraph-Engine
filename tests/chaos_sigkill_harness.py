#!/usr/bin/env python3
"""SIGKILL mid-flush stress test for ``hypergraph-server``.

Each run streams ``--edges`` hyperedges into a server over a Unix socket and
SIGKILLs the server's process group after a random delay. It then checks the
WAL segments byte by byte, runs the offline ``chaos_verify`` example (which
performs real crash recovery), and restarts the server to resume the
unacknowledged edges. When every edge has been acknowledged, the server is
stopped gracefully and the final state must hold exactly ``--edges`` edges.

Invariants checked after every kill:

* every WAL frame up to the last batch commit has magic ``THG2``, a valid
  CRC32 and a sequence id contiguous with the previous frame, across segments;
* acknowledged edges ⊆ recovered edges ⊆ sent edges, with no duplicates;
* no dangling, unlinked or orphan nodes, and every payload is intact;
* the recovered ``last_sequence`` covers every acknowledged sequence id, and
  sequence ids returned after a restart are strictly larger than it.

Prerequisites::

    cargo build --release -p grpc-server
    cargo build --release -p engine-core --example chaos_verify
    pip install -e integrations/python-client   # optional: falls back to the source tree

Usage::

    python tests/chaos_sigkill_harness.py --edges 100000 --seed 7
    python tests/chaos_sigkill_harness.py --edges 1000000 --rate 50000 --runs 100
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import queue
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from hypergraph_client._proto import pb, pb_grpc
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "integrations" / "python-client" / "src"))
    from hypergraph_client._proto import pb, pb_grpc

WAL_MAGIC = b"THG2"
WAL_HEADER = struct.Struct("<4sHIQIH")
BATCH_COMMIT_RECORD = struct.Struct("<QI")
OP_BATCH_COMMIT = 5
KNOWN_OPS = {1, 2, 3, 4, 5}
CHANNEL_OPTIONS = [("grpc.default_authority", "localhost")]
RPC_TIMEOUT_S = 60.0


class ChaosFailure(AssertionError):
    """An invariant was violated; artifacts are kept for inspection."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ChaosFailure(message)


# --------------------------------------------------------------- processes

_LIVE: set[subprocess.Popen] = set()
_SPAWNED_PIDS: list[int] = []
_LIVE_LOCK = threading.Lock()


def _linux_pdeathsig() -> None:
    """Child-side: die with the harness even if it is SIGKILLed itself."""
    pr_set_pdeathsig = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(pr_set_pdeathsig, signal.SIGKILL)


def spawn(command: list[str], log_path: Path) -> subprocess.Popen:
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_linux_pdeathsig if sys.platform.startswith("linux") else None,
        )
    finally:
        log.close()
    with _LIVE_LOCK:
        _LIVE.add(proc)
        _SPAWNED_PIDS.append(proc.pid)
    return proc


def reap(proc: subprocess.Popen, sig: int = signal.SIGKILL, timeout: float = 10.0) -> int:
    """Signals the process group, waits for exit and always collects the
    exit status so no zombie is left behind."""
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        return proc.wait()
    finally:
        with _LIVE_LOCK:
            _LIVE.discard(proc)


def reap_all() -> None:
    with _LIVE_LOCK:
        procs = list(_LIVE)
    for proc in procs:
        try:
            reap(proc)
        except Exception as err:  # noqa: BLE001 - best effort at exit
            print(f"warning: failed to reap pid {proc.pid}: {err}", file=sys.stderr)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break


def leftover_processes() -> list[int]:
    """Spawned pids that are still alive (zombies count as alive)."""
    alive = []
    for pid in _SPAWNED_PIDS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        alive.append(pid)
    return alive


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


atexit.register(reap_all)
signal.signal(signal.SIGTERM, _raise_interrupt)
signal.signal(signal.SIGHUP, _raise_interrupt)


# ------------------------------------------------------------- WAL scanner


@dataclass
class WalScan:
    frames: int = 0
    committed_batches: int = 0
    last_committed_seq: int = 0
    torn_bytes: int = 0


def scan_wal(data_dir: Path) -> WalScan:
    """Independent re-implementation of the frame format: validates the
    committed prefix of every segment and measures the torn tail."""
    segments = sorted(data_dir.glob("wal-*.wal"))
    scan = WalScan()
    expected_seq: int | None = None
    for position, segment in enumerate(segments):
        data = segment.read_bytes()
        start_seq = int(segment.stem.removeprefix("wal-"))
        if expected_seq is not None:
            check(
                start_seq == expected_seq,
                f"{segment.name}: starts at {start_seq}, expected {expected_seq}",
            )
        seq = start_seq
        offset = 0
        committed_end = 0
        pending_first: int | None = None
        pending_count = 0
        while offset + WAL_HEADER.size <= len(data):
            magic, op, length, frame_seq, crc, _reserved = WAL_HEADER.unpack_from(data, offset)
            end = offset + WAL_HEADER.size + length
            if magic != WAL_MAGIC or op not in KNOWN_OPS or end > len(data):
                break
            payload = data[offset + WAL_HEADER.size : end]
            if zlib.crc32(data[offset + 4 : offset + 18] + payload) != crc:
                break
            check(frame_seq == seq, f"{segment.name}@{offset}: seq {frame_seq}, expected {seq}")
            if op == OP_BATCH_COMMIT:
                first_seq, frame_count = BATCH_COMMIT_RECORD.unpack(payload)
                check(
                    frame_count == pending_count
                    and (pending_count == 0 or first_seq == pending_first),
                    f"{segment.name}@{offset}: commit record does not match its batch",
                )
                committed_end = end
                scan.committed_batches += 1
                scan.last_committed_seq = frame_seq
                pending_first, pending_count = None, 0
            else:
                if pending_first is None:
                    pending_first = frame_seq
                pending_count += 1
            scan.frames += 1
            seq += 1
            offset = end
        tail = len(data) - committed_end
        if position < len(segments) - 1:
            check(tail == 0, f"{segment.name}: {tail} torn bytes in a non-final segment")
        scan.torn_bytes += tail
        # The next segment continues after the last committed frame.
        expected_seq = scan.last_committed_seq + 1 if committed_end else start_seq
    return scan


# ---------------------------------------------------------------- workload


def edge_request(index: int) -> pb.InsertHyperedgeRequest:
    return pb.InsertHyperedgeRequest(
        session_id="chaos",
        hyperedge=pb.HyperedgePayload(
            event_type="CHAOS",
            summary=f"chaos-{index}",
            participants=[
                pb.Participant(entity_id=f"s{index}"),
                pb.Participant(entity_id=f"o{index}"),
            ],
            temporal_boundary=pb.TemporalBoundary(valid_from_epoch=index, is_ongoing=True),
        ),
    )


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self.next_slot = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: int, stop: threading.Event) -> bool:
        with self.lock:
            slot = max(self.next_slot, time.monotonic())
            self.next_slot = slot + tokens * self.interval
        delay = slot - time.monotonic()
        return not (delay > 0 and stop.wait(delay))


@dataclass
class RunState:
    edges: int
    acked: set[int] = field(default_factory=set)
    sent: set[int] = field(default_factory=set)
    probes_sent: set[int] = field(default_factory=set)
    probes_acked: set[int] = field(default_factory=set)
    next_probe: int = 1
    max_acked_seq: int = 0
    worker_last_seq: dict[int, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def pending(self) -> list[int]:
        return [i for i in range(self.edges) if i not in self.acked]


def worker(
    worker_id: int,
    stub: pb_grpc.HypergraphEngineServiceStub,
    chunks: queue.Queue,
    state: RunState,
    limiter: RateLimiter,
    stop: threading.Event,
    seq_floor: int,
    errors: list[str],
) -> None:
    while not stop.is_set():
        try:
            chunk = chunks.get_nowait()
        except queue.Empty:
            return
        if not limiter.acquire(len(chunk), stop):
            return
        with state.lock:
            state.sent.update(chunk)
        try:
            response = stub.InsertHyperedgeBatch(
                (edge_request(i) for i in chunk), timeout=RPC_TIMEOUT_S
            )
        except grpc.RpcError:
            return
        if not response.all_succeeded or response.processed_count != len(chunk):
            errors.append(f"worker {worker_id}: batch rejected: {list(response.errors)[:3]}")
            stop.set()
            return
        with state.lock:
            state.acked.update(chunk)
            probe = state.next_probe
            state.next_probe += 1
            state.probes_sent.add(probe)
        # Probes use negative valid_from so they never collide with edges.
        try:
            reply = stub.InsertHyperedge(edge_request(-probe), timeout=RPC_TIMEOUT_S)
        except grpc.RpcError:
            return
        with state.lock:
            state.probes_acked.add(probe)
            last = state.worker_last_seq.get(worker_id, seq_floor)
            if reply.sequence_id <= last:
                errors.append(
                    f"worker {worker_id}: sequence id {reply.sequence_id} not above {last}"
                )
                stop.set()
                return
            state.worker_last_seq[worker_id] = reply.sequence_id
            state.max_acked_seq = max(state.max_acked_seq, reply.sequence_id)


# ----------------------------------------------------------------- harness


@dataclass
class Binaries:
    server: Path
    verify: Path


def find_binaries(args: argparse.Namespace) -> Binaries:
    server = Path(
        args.server_bin
        or os.environ.get("HYPERGRAPH_SERVER_BIN")
        or REPO_ROOT / "target" / "release" / "hypergraph-server"
    )
    verify = Path(args.verify_bin or REPO_ROOT / "target" / "release" / "examples" / "chaos_verify")
    for path, build in (
        (server, "cargo build --release -p grpc-server"),
        (verify, "cargo build --release -p engine-core --example chaos_verify"),
    ):
        if not path.exists():
            sys.exit(f"{path} not found; run `{build}`")
    return Binaries(server, verify)


def start_server(binaries: Binaries, workdir: Path, sync: str, log: Path) -> subprocess.Popen:
    socket_path = workdir / "hg.sock"
    socket_path.unlink(missing_ok=True)
    proc = spawn(
        [
            str(binaries.server),
            "--data-dir",
            str(workdir / "data"),
            "--listen",
            f"unix://{socket_path}",
            "--sync",
            sync,
        ],
        log,
    )
    deadline = time.monotonic() + 30
    while not socket_path.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            reap(proc)
            raise ChaosFailure(f"server failed to start; see {log}")
        time.sleep(0.01)
    return proc


def run_verifier(binaries: Binaries, workdir: Path) -> tuple[dict, list[int]]:
    out = workdir / "valid_from.txt"
    result = subprocess.run(
        [
            str(binaries.verify),
            str(workdir / "data"),
            "--valid-from-out",
            str(out),
            "--chaos-summaries",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    check(result.returncode == 0, f"chaos_verify failed: {result.stderr.strip()}")
    report = json.loads(result.stdout)
    valid_from = [int(line) for line in out.read_text().split()]
    return report, valid_from


def verify_recovered(
    binaries: Binaries, workdir: Path, state: RunState, final: bool
) -> dict:
    report, valid_from = run_verifier(binaries, workdir)
    for key in ("dangling_participants", "unlinked_participants", "orphan_nodes", "summary_mismatches"):
        check(report[key] == 0, f"{key} = {report[key]}")
    check(len(valid_from) == len(set(valid_from)), "an edge was recovered twice")
    edges = {v for v in valid_from if v >= 0}
    probes = {-v for v in valid_from if v < 0}
    lost = state.acked - edges
    check(not lost, f"{len(lost)} acknowledged edges lost, e.g. {sorted(lost)[:5]}")
    check(edges <= state.sent, f"phantom edges: {sorted(edges - state.sent)[:5]}")
    lost_probes = state.probes_acked - probes
    check(not lost_probes, f"acknowledged probes lost: {sorted(lost_probes)[:5]}")
    check(probes <= state.probes_sent, f"phantom probes: {sorted(probes - state.probes_sent)[:5]}")
    check(
        report["last_sequence"] >= state.max_acked_seq,
        f"last_sequence {report['last_sequence']} below acknowledged {state.max_acked_seq}",
    )
    check(report["live_nodes"] == 2 * report["live_edges"], "node count does not match edges")
    if final:
        check(edges == set(range(state.edges)), f"final edge set has {len(edges)} edges")
    report["recovered_edges"] = len(edges)
    return report


def run_once(run: int, args: argparse.Namespace, binaries: Binaries, rng: random.Random) -> None:
    workdir = Path(tempfile.mkdtemp(prefix=f"hgchaos{run}-", dir="/tmp"))
    state = RunState(edges=args.edges)
    started = time.monotonic()
    kills = 0
    seq_floor = 0
    ok = False
    try:
        for cycle in range(args.max_cycles):
            pending = state.pending()
            if not pending:
                break
            log = workdir / f"server-{cycle:03}.log"
            proc = start_server(binaries, workdir, args.sync, log)
            channel = grpc.insecure_channel(f"unix://{workdir / 'hg.sock'}", options=CHANNEL_OPTIONS)
            try:
                grpc.channel_ready_future(channel).result(timeout=30)
                stub = pb_grpc.HypergraphEngineServiceStub(channel)
                chunks: queue.Queue = queue.Queue()
                for start in range(0, len(pending), args.chunk):
                    chunks.put(pending[start : start + args.chunk])
                stop = threading.Event()
                errors: list[str] = []
                limiter = RateLimiter(args.rate)
                state.worker_last_seq.clear()
                threads = [
                    threading.Thread(
                        target=worker,
                        args=(w, stub, chunks, state, limiter, stop, seq_floor, errors),
                        daemon=True,
                    )
                    for w in range(args.workers)
                ]
                for thread in threads:
                    thread.start()
                kill_at = time.monotonic() + rng.uniform(args.kill_min, args.kill_max)
                while time.monotonic() < kill_at and any(t.is_alive() for t in threads):
                    time.sleep(0.005)
                    check(proc.poll() is None, f"server exited on its own; see {log}")
                acked_at_kill = len(state.acked)
                reap(proc)
                kills += 1
                stop.set()
                for thread in threads:
                    thread.join(timeout=RPC_TIMEOUT_S)
                    check(not thread.is_alive(), "worker thread hung after SIGKILL")
                check(not errors, "; ".join(errors))
            finally:
                channel.close()
                if proc.poll() is None:
                    reap(proc)

            wal = scan_wal(workdir / "data")
            report = verify_recovered(binaries, workdir, state, final=False)
            seq_floor = report["last_sequence"]
            state.max_acked_seq = max(state.max_acked_seq, seq_floor)
            print(
                f"  run {run} cycle {cycle}: SIGKILL with {acked_at_kill}/{args.edges} acked, "
                f"{report['recovered_edges']} recovered, last_seq {seq_floor}, "
                f"{wal.committed_batches} committed batches, {wal.torn_bytes} torn bytes"
            )
        check(not state.pending(), f"not all edges acknowledged after {args.max_cycles} cycles")

        log = workdir / "server-final.log"
        proc = start_server(binaries, workdir, args.sync, log)
        check(reap(proc, signal.SIGINT, timeout=60) == 0, f"graceful shutdown failed; see {log}")
        report = verify_recovered(binaries, workdir, state, final=True)
        print(
            f"run {run}: PASS - {kills} kills, {report['live_edges']} edges "
            f"({report['live_edges'] - args.edges} probes), last_seq {report['last_sequence']}, "
            f"{time.monotonic() - started:.1f}s"
        )
        ok = True
    finally:
        if ok and not args.keep_artifacts:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"run {run}: artifacts kept in {workdir}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edges", type=int, default=100_000)
    parser.add_argument("--rate", type=float, default=10_000, help="edges per second")
    parser.add_argument("--kill-min", type=float, default=0.1, help="seconds")
    parser.add_argument("--kill-max", type=float, default=5.0, help="seconds")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sync", choices=("fsync", "buffered"), default="fsync")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--server-bin")
    parser.add_argument("--verify-bin")
    args = parser.parse_args()
    if args.kill_min <= 0 or args.kill_max < args.kill_min:
        parser.error("need 0 < --kill-min <= --kill-max")
    return args


def main() -> int:
    args = parse_args()
    binaries = find_binaries(args)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    rng = random.Random(seed)
    print(f"chaos: {args.runs} run(s) x {args.edges} edges at {args.rate:.0f}/s, seed {seed}")
    status = 0
    try:
        for run in range(args.runs):
            run_once(run, args, binaries, rng)
    except ChaosFailure as failure:
        print(f"FAIL: {failure}", file=sys.stderr)
        status = 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        status = 130
    finally:
        reap_all()
        leftovers = leftover_processes()
        if leftovers:
            print(f"FAIL: leftover server processes {leftovers}", file=sys.stderr)
            status = status or 1
    return status


if __name__ == "__main__":
    sys.exit(main())
