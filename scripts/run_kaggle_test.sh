#!/usr/bin/env bash
# Runs the vLLM + offloader CUDA test as a private Kaggle GPU kernel (T4).
#
#   scripts/run_kaggle_test.sh
#
# See scripts/kaggle_job.sh for the settings (SOURCE_REF, SKIP_PUSH, ...).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_DIR="${JOB_DIR:-kaggle_gpu_job}" \
SOURCE_REF="${SOURCE_REF:-v1.0.1}" \
OUT_DIR="${OUT_DIR:-kaggle_output}" \
POLL_SECONDS="${POLL_SECONDS:-30}" \
PASS_MARKER="${PASS_MARKER:-RESULT: PASS}" \
  exec "$ROOT/scripts/kaggle_job.sh"
