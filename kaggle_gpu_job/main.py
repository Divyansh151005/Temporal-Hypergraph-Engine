"""Kaggle entrypoint: builds the Temporal Hypergraph Engine with CUDA and runs the vLLM test."""

import base64
import glob
import io
import os
import shutil
import subprocess
import sys
import tarfile

REPO_URL = "https://github.com/Divyansh151005/Temporal-Hypergraph-Engine.git"
REPO_TAG = "v1.0.1"
# Outside /kaggle/working so the clone and Rust target/ are not saved as kernel output.
REPO_DIR = "/tmp/Temporal-Hypergraph-Engine"

# scripts/kaggle_job.sh replaces these lines with {filename: source} for the
# job's other *.py files and a base64 tar.gz of the repo (the GitHub repo is private).
EMBEDDED_FILES = None
SOURCE_ARCHIVE = None


def run(cmd, cwd=None, shell=False):
    print(f"\n===== {cmd if shell else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, shell=shell, check=True)


def main():
    run("nvidia-smi", shell=True)

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

    pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    run(pip + ["maturin", "vllm", "torch", "transformers"], cwd=REPO_DIR)

    bindings = os.path.join(REPO_DIR, "crates", "pyo3-bindings")
    run([sys.executable, "-m", "maturin", "build", "--release", "--interpreter", sys.executable], cwd=bindings)
    wheels = sorted(glob.glob(os.path.join(REPO_DIR, "target", "wheels", "hypergraph_engine_native-*.whl")))
    if not wheels:
        raise RuntimeError("maturin produced no wheel")
    run(pip + ["--force-reinstall", "--no-deps", wheels[-1]])
    run(pip + ["--no-deps", "-e", os.path.join(REPO_DIR, "python")])

    test_path = os.path.join(REPO_DIR, "run_vllm_test.py")
    if EMBEDDED_FILES is not None:
        with open(test_path, "w") as f:
            f.write(EMBEDDED_FILES["run_vllm_test.py"])
    else:
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_vllm_test.py"), test_path)

    run([sys.executable, "run_vllm_test.py"], cwd=REPO_DIR)


if __name__ == "__main__":
    main()
