#!/usr/bin/env bash
# Runs the vLLM + offloader CUDA test as a private Kaggle GPU kernel (T4).
#
#   scripts/run_kaggle_test.sh
#
# Script kernels upload only main.py, so the test and a `git archive` of
# SOURCE_REF (default v1.0.1) are embedded into a staged copy of it.
#
# Needs Kaggle CLI 2.x (the 1.7 CLI cannot use KGAT_ tokens). Set KAGGLE_BIN
# to override the binary; ~/.local/bin/kaggle (uv tool install) is preferred.
# SKIP_PUSH=1 re-attaches to the latest run without pushing a new version.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_DIR="$ROOT/kaggle_gpu_job"
STAGE_DIR="$ROOT/build/kaggle_push"
OUT_DIR="$ROOT/kaggle_output"
ACCELERATOR="${ACCELERATOR:-NvidiaTeslaT4}"
SOURCE_REF="${SOURCE_REF:-v1.0.1}"
POLL_SECONDS=30
MAX_WAIT_SECONDS=$((3 * 3600))

if [[ -n "${KAGGLE_BIN:-}" ]]; then
  KAGGLE="$KAGGLE_BIN"
elif [[ -x "$HOME/.local/bin/kaggle" ]]; then
  KAGGLE="$HOME/.local/bin/kaggle"
else
  KAGGLE="kaggle"
fi

SLUG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$JOB_DIR/kernel-metadata.json")"

echo "==> staging $STAGE_DIR (kernel $SLUG, source $SOURCE_REF, accelerator $ACCELERATOR)"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$JOB_DIR/kernel-metadata.json" "$STAGE_DIR/"
# Only the Cargo workspace and the Python offloader are needed to build and test.
git -C "$ROOT" archive --format=tar.gz -o "$STAGE_DIR/source.tar.gz" "$SOURCE_REF" \
  Cargo.toml Cargo.lock rust-toolchain.toml README.md LICENSE \
  crates benchmarks proto python
python3 - "$JOB_DIR/main.py" "$JOB_DIR/run_vllm_test.py" "$STAGE_DIR/source.tar.gz" "$STAGE_DIR/main.py" <<'EOF'
import base64, pathlib, sys
main_src, test_src, archive, out = (pathlib.Path(p) for p in sys.argv[1:])
text = main_src.read_text()
replacements = {
    "TEST_SOURCE = None\n": f"TEST_SOURCE = {test_src.read_text()!r}\n",
    "SOURCE_ARCHIVE = None\n": f"SOURCE_ARCHIVE = {base64.b64encode(archive.read_bytes()).decode()!r}\n",
}
for placeholder, value in replacements.items():
    if text.count(placeholder) != 1:
        sys.exit(f"expected exactly one '{placeholder.strip()}' line in {main_src}")
    text = text.replace(placeholder, value)
out.write_text(text)
EOF
rm "$STAGE_DIR/source.tar.gz"

if [[ "${SKIP_PUSH:-0}" == "1" ]]; then
  echo "==> SKIP_PUSH=1: attaching to the latest run of $SLUG"
else
  echo "==> pushing"
  "$KAGGLE" kernels push -p "$STAGE_DIR" --accelerator "$ACCELERATOR"
fi

echo "==> polling every ${POLL_SECONDS}s"
start=$(date +%s)
sleep 10
while true; do
  status="$("$KAGGLE" kernels status "$SLUG" 2>&1 || true)"
  lower="$(tr '[:upper:]' '[:lower:]' <<<"$status")"
  elapsed=$(( $(date +%s) - start ))
  printf '[%02d:%02d] %s\n' $((elapsed / 60)) $((elapsed % 60)) "$status"
  if [[ "$lower" != *"has status"* ]]; then
    echo "unexpected status response; is the metadata id the kernel's URL slug?" >&2
    exit 1
  fi
  if [[ "$lower" != *queued* && "$lower" != *running* ]]; then
    break
  fi
  if (( elapsed > MAX_WAIT_SECONDS )); then
    echo "gave up after ${MAX_WAIT_SECONDS}s; kernel still running on Kaggle" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "==> downloading output to $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
"$KAGGLE" kernels output "$SLUG" -p "$OUT_DIR"

echo "==> output files"
ls -la "$OUT_DIR"

# Kaggle logs are a JSON list of {stream_name, time, data} records.
for log in "$OUT_DIR"/*.log; do
  [[ -e "$log" ]] || continue
  echo "===== $(basename "$log")"
  python3 - "$log" <<'EOF'
import json, sys
raw = open(sys.argv[1]).read()
try:
    records = json.loads(raw)
except json.JSONDecodeError:
    sys.stdout.write(raw)
else:
    sys.stdout.write("".join(r.get("data", "") for r in records))
EOF
done

if [[ "$lower" == *complete* ]] && grep -qs "RESULT: PASS" "$OUT_DIR"/*.log; then
  echo "==> RESULT: PASS"
else
  echo "==> RESULT: FAIL ($status)" >&2
  exit 1
fi
