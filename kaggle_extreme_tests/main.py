"""Kaggle entrypoint: builds the Temporal Hypergraph Engine v2.0.0 and runs the
10M-edge ingest, the 100k-entity competitor bench and the vLLM soak test."""

import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

REPO_URL = "https://github.com/Divyansh151005/Temporal-Hypergraph-Engine.git"
REPO_TAG = "v2.0.0"
# Outside /kaggle/working so the source and Rust target/ are not saved as kernel output.
REPO_DIR = "/tmp/Temporal-Hypergraph-Engine"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_working")

# scripts/kaggle_job.sh replaces these lines with {filename: source} for the
# job's other *.py files and a base64 tar.gz of the repo (the GitHub repo is private).
EMBEDDED_FILES = None
SOURCE_ARCHIVE = None


def run(cmd, cwd=None, shell=False):
    print(f"\n===== {cmd if shell else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, shell=shell, check=True)


def bootstrap():
    run("nvidia-smi", shell=True)
    run("free -h && nproc", shell=True)

    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", shell=True)
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ["PATH"]
    run(["cargo", "--version"])

    if os.path.isdir(REPO_DIR):
        pass
    elif SOURCE_ARCHIVE is not None:
        print(f"\n===== extracting embedded {REPO_TAG} source to {REPO_DIR}", flush=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(SOURCE_ARCHIVE)), mode="r:gz") as tar:
            tar.extractall(REPO_DIR, filter="data")
    else:
        run(["git", "clone", "--depth", "1", "--branch", REPO_TAG, REPO_URL, REPO_DIR])

    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("soak_test.py",):
        path = os.path.join(REPO_DIR, name)
        if EMBEDDED_FILES is not None:
            with open(path, "w") as f:
                f.write(EMBEDDED_FILES[name])
        else:
            shutil.copy(os.path.join(here, name), path)

    pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    run(pip + ["maturin", "vllm", "torch", "transformers",
               "networkx", "faiss-cpu", "tiktoken", "psutil", "nvidia-ml-py"], cwd=REPO_DIR)

    bindings = os.path.join(REPO_DIR, "crates", "pyo3-bindings")
    run([sys.executable, "-m", "maturin", "build", "--release", "--interpreter", sys.executable], cwd=bindings)
    wheels = sorted(glob.glob(os.path.join(REPO_DIR, "target", "wheels", "hypergraph_engine_native-*.whl")))
    if not wheels:
        raise RuntimeError("maturin produced no wheel")
    run(pip + ["--force-reinstall", "--no-deps", wheels[-1]])
    run(pip + ["--no-deps", "-e", os.path.join(REPO_DIR, "python")])
    run(pip + ["-e", os.path.join(REPO_DIR, "python", "evaluation")])
    run(pip + ["-e", os.path.join(REPO_DIR, "integrations", "python-client")])


def execute(name, cmd, env=None, log_path=None):
    """Runs one execution without raising; output is echoed to the kernel log
    and, when log_path is set, also written there."""
    print(f"\n########## {name}\n===== {' '.join(cmd)}", flush=True)
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, env={**os.environ, **(env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = open(log_path, "w") if log_path else None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log:
                log.write(line)
                log.flush()
    finally:
        if log:
            log.close()
    code = proc.wait()
    result = {"name": name, "command": cmd, "exit_code": code, "passed": code == 0,
              "duration_secs": round(time.time() - start, 1)}
    print(f"########## {name}: {'PASSED' if code == 0 else f'FAILED (exit {code})'} "
          f"in {result['duration_secs']}s", flush=True)
    return result


def main():
    os.makedirs(OUT, exist_ok=True)
    bootstrap()

    results = [
        execute("Execution 1: 10M edge in-memory ingest", [
            "cargo", "bench", "-p", "thg-benchmarks", "--bench", "scale_stress", "--",
            "--edges", "10000000", "--modes", "memory", "--out", f"{OUT}/scale_10m.json",
        ]),
        execute("Execution 2: 100k entity competitor bench", [
            sys.executable, "-m", "hypergraph_evaluation.competitor_bench",
            "--entities", "100000", "--facts", "500000", "--queries", "1000", "--hops", "3",
            "--embedder", "hashing", "--out", f"{OUT}/competitor_100k.json", "--cache-dir", "/tmp/thg-cache",
        ]),
        execute("Execution 3: vLLM soak test", [sys.executable, "soak_test.py"],
                env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}, log_path=f"{OUT}/soak_test_logs.txt"),
    ]

    with open(f"{OUT}/run_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "\n".join(f"{'PASS' if r['passed'] else 'FAIL'}  {r['duration_secs']:8.1f}s  {r['name']}"
                           for r in results))
    if all(r["passed"] for r in results):
        print("ALL EXECUTIONS PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
