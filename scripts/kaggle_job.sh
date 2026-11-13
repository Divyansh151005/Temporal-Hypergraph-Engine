#!/usr/bin/env bash
# Pushes a Kaggle script kernel, polls it to completion and downloads its output.
#
#   JOB_DIR=kaggle_gpu_job SOURCE_REF=v1.0.1 OUT_DIR=kaggle_output scripts/kaggle_job.sh
#
# Script kernels upload only main.py, so the job's other *.py files
# (EMBEDDED_FILES) and a `git archive` of SOURCE_REF (SOURCE_ARCHIVE) are
# embedded into a staged copy of it. The GitHub repo is private.
#
# Settings (environment):
#   JOB_DIR           job folder with kernel-metadata.json and main.py (required)
#   SOURCE_REF        git ref to embed (default HEAD)
#   OUT_DIR           where kernel output is downloaded (default kaggle_output)
#   POLL_SECONDS      status poll interval (default 30)
#   MAX_WAIT_SECONDS  give up polling after this long (default 3h)
#   ACCELERATOR       Kaggle accelerator (default NvidiaTeslaT4)
#   PASS_MARKER       log line that marks success (default "RESULT: PASS")
#   SKIP_PUSH=1       attach to the latest run without pushing a new version
#   KAGGLE_BIN        Kaggle CLI 2.x binary (the 1.7 CLI cannot use KGAT_ tokens);
#                     ~/.local/bin/kaggle (uv tool install) is preferred
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
abs() { [[ "$1" == /* ]] && echo "$1" || echo "$ROOT/$1"; }

JOB_DIR="$(abs "${JOB_DIR:?JOB_DIR is required}")"
SOURCE_REF="${SOURCE_REF:-HEAD}"
OUT_DIR="$(abs "${OUT_DIR:-kaggle_output}")"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-$((3 * 3600))}"
ACCELERATOR="${ACCELERATOR:-NvidiaTeslaT4}"
PASS_MARKER="${PASS_MARKER:-RESULT: PASS}"
STAGE_DIR="$ROOT/build/kaggle_push/$(basename "$JOB_DIR")"

if [[ -n "${KAGGLE_BIN:-}" ]]; then
  KAGGLE="$KAGGLE_BIN"
elif [[ -x "$HOME/.local/bin/kaggle" ]]; then
  KAGGLE="$HOME/.local/bin/kaggle"
else
  KAGGLE="kaggle"
fi

# Kaggle derives the kernel URL slug from the title, so polling the metadata id
# only works when the two agree.
SLUG="$(python3 - "$JOB_DIR/kernel-metadata.json" <<'EOF'
import json, re, sys
meta = json.load(open(sys.argv[1]))
owner, slug = meta["id"].split("/", 1)
title_slug = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
if slug != title_slug:
    sys.exit(f"kernel-metadata.json: id slug {slug!r} does not match title slug {title_slug!r}")
print(meta["id"])
EOF
)"

echo "==> staging $STAGE_DIR (kernel $SLUG, source $SOURCE_REF, accelerator $ACCELERATOR)"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$JOB_DIR/kernel-metadata.json" "$STAGE_DIR/"

# The Cargo workspace plus the Python packages the jobs install; paths missing
# at SOURCE_REF (e.g. license files renamed between tags) are skipped.
archive_paths=()
for path in Cargo.toml Cargo.lock rust-toolchain.toml README.md LICENSE LICENSE-APACHE LICENSE-MIT \
  crates benchmarks proto python integrations/python-client; do
  if [[ -n "$(git -C "$ROOT" ls-tree --name-only "$SOURCE_REF" -- "$path")" ]]; then
    archive_paths+=("$path")
  fi
done
git -C "$ROOT" archive --format=tar.gz -o "$STAGE_DIR/source.tar.gz" "$SOURCE_REF" "${archive_paths[@]}"

python3 - "$JOB_DIR" "$STAGE_DIR/source.tar.gz" "$STAGE_DIR/main.py" <<'EOF'
import base64, pathlib, sys
job_dir, archive, out = (pathlib.Path(p) for p in sys.argv[1:])
main_src = job_dir / "main.py"
embedded = {p.name: p.read_text() for p in sorted(job_dir.glob("*.py")) if p.name != "main.py"}
text = main_src.read_text()
replacements = {
    "EMBEDDED_FILES = None\n": f"EMBEDDED_FILES = {embedded!r}\n",
    "SOURCE_ARCHIVE = None\n": f"SOURCE_ARCHIVE = {base64.b64encode(archive.read_bytes()).decode()!r}\n",
}
for placeholder, value in replacements.items():
    if text.count(placeholder) != 1:
        sys.exit(f"expected exactly one '{placeholder.strip()}' line in {main_src}")
    text = text.replace(placeholder, value)
out.write_text(text)
print(f"embedded {sorted(embedded)} and {archive.stat().st_size} bytes of source")
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

if [[ "$lower" == *complete* ]] && grep -qsF "$PASS_MARKER" "$OUT_DIR"/*.log; then
  echo "==> RESULT: PASS"
else
  echo "==> RESULT: FAIL ($status)" >&2
  exit 1
fi
