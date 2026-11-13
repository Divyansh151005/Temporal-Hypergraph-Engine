#!/usr/bin/env bash
# Runs the 10M-edge ingest, the 100k-entity competitor bench and the vLLM soak
# test as one private Kaggle GPU kernel (T4), then prints a metrics digest.
#
#   scripts/run_kaggle_extreme.sh
#
# See scripts/kaggle_job.sh for the settings (SOURCE_REF, SKIP_PUSH, ...).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-results/extreme_scale}"

status=0
JOB_DIR="${JOB_DIR:-kaggle_extreme_tests}" \
SOURCE_REF="${SOURCE_REF:-v2.0.0}" \
OUT_DIR="$OUT_DIR" \
POLL_SECONDS="${POLL_SECONDS:-60}" \
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-$((4 * 3600))}" \
PASS_MARKER="${PASS_MARKER:-ALL EXECUTIONS PASSED}" \
  "$ROOT/scripts/kaggle_job.sh" || status=$?

[[ "$OUT_DIR" == /* ]] || OUT_DIR="$ROOT/$OUT_DIR"
echo
echo "==> metrics digest ($OUT_DIR)"
python3 - "$OUT_DIR" <<'EOF'
import json, pathlib, sys

out = pathlib.Path(sys.argv[1])

def load(name):
    path = out / name
    return json.loads(path.read_text()) if path.exists() else None

MiB = 2**20

summary = load("run_summary.json") or []
for r in summary:
    print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['duration_secs']:8.1f}s  {r['name']}")

scale = load("scale_10m.json")
print("\n-- 10M edge ingest (memory mode)")
if scale:
    env = scale.get("environment", {})
    print(f"   host: {env.get('cpu', '?')}, {env.get('logical_cores', '?')} cores, "
          f"{env.get('memory_bytes', 0) / 2**30:.1f} GiB")
    for m in scale["modes"]:
        print(f"   [{m['mode']}] ingested {m['ingested_edges']:,} / {m['target_edges']:,} edges in {m['wall_secs']:.1f}s"
              f" (stop: {m.get('stop_reason') or 'target reached'})")
        print(f"   sustained {m['sustained_edges_per_sec']:,.0f} edges/s, peak {m['peak_edges_per_sec']:,.0f} edges/s,"
              f" {m['amortized_ns_per_edge']:,.0f} ns/edge amortized")
        print(f"   batch p50 {m['batch_latency']['p50_us'] / 1e3:.1f} ms, p99 {m['batch_latency']['p99_us'] / 1e3:.1f} ms")
        print(f"   heap {m['heap_bytes_per_million_edges'] / MiB:.0f} MiB per 1M edges, "
              f"engine heap {m['engine_heap_bytes'] / 2**30:.2f} GiB, peak RSS {m['peak_rss_bytes'] / 2**30:.2f} GiB")
else:
    print("   missing scale_10m.json")

comp = load("competitor_100k.json")
print("\n-- 100k entity competitor bench")
if comp and "contenders" in comp:
    ds = comp.get("dataset", {})
    print(f"   {ds.get('entities', '?'):,} entities, {ds.get('facts', '?'):,} facts, {ds.get('queries', '?')} queries")
    contenders = comp["contenders"]
    hg = next((c for c in contenders if c["name"] == "hypergraph"), None)
    for c in contenders:
        lat = c["latency_ms"]
        gap = ""
        if hg and c is not hg and hg["latency_ms"]["p50"] > 0:
            gap = f"  ({lat['p50'] / hg['latency_ms']['p50']:.1f}x p50, {lat['p95'] / hg['latency_ms']['p95']:.1f}x p95 vs hypergraph)"
        print(f"   {c['name']:<22} p50 {lat['p50']:8.3f} ms  p95 {lat['p95']:8.3f} ms  "
              f"F1 {c['f1']:.3f}  stale {c['stale_rate']:.3f}{gap}")
else:
    print("   missing competitor_100k.json")

soak = load("soak_summary.json")
print("\n-- vLLM soak")
if soak:
    for s in soak["samples"]:
        split = ""
        if s.get("rss_anon_bytes") is not None:
            split = f" (anon {s['rss_anon_bytes'] / MiB:7.1f}, file {s['rss_file_bytes'] / MiB:7.1f})"
        print(f"   iter {s['iteration']:3d}: RSS {s['rss_bytes'] / MiB:8.1f} MiB{split}  VRAM {s['vram_used_bytes'] / MiB:8.1f} MiB"
              f"  torch alloc {s['cuda_allocated_bytes'] / MiB:7.1f} MiB  {s.get('window_tokens_per_sec', s['tokens_per_sec']):5.1f} tok/s")
    print(f"   growth iter {soak['baseline_iteration']} -> {soak['iterations']}: "
          f"RSS {soak['rss_growth_bytes'] / MiB:+.1f} MiB, VRAM {soak['vram_growth_bytes'] / MiB:+.1f} MiB")
    if soak.get("rss_anon_growth_bytes") is not None:
        print(f"   RSS split: anon (heap) {soak['rss_anon_growth_bytes'] / MiB:+.1f} MiB, file-backed "
              f"{soak['rss_file_growth_bytes'] / MiB:+.1f} MiB vs {soak['kv_bytes_appended_since_baseline'] / MiB:.1f} MiB "
              f"appended to KV segments")
    print(f"   {soak['offload_cycles']} offload cycles, {soak['blocks_moved']} blocks, "
          f"{soak['mismatches']} mismatches, {soak['leftover_blocks']} leftover")
    print(f"   {soak['total_tokens']} tokens at {soak['total_tokens'] / soak['generation_secs']:.1f} tok/s")
    print(f"   verdict: {'PASS' if soak['passed'] else 'FAIL'} {soak['checks']}")
else:
    print("   missing soak_summary.json")
EOF

exit "$status"
