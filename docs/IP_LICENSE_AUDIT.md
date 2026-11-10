# IP / License Compliance Audit: SSPL-1.0 Enforcement

**Date:** 2026-11-10
**Scope:** Temporal Hypergraph Engine v2.0.0 and later, covering the working tree at `main` (`104a31c`), the git tags `v2.0.0` and `v2.0.1`, the GitHub releases, the publish workflow and Dockerfile, and live PyPI and npm metadata.
**Result:** **Compliant.** No v2.0.0+ artifact is marked MIT, Apache-2.0 or any other permissive license. One stale local build artifact was removed, and three hardening fixes were applied (see "Fixes applied"). One release-pipeline issue that is not a license problem remains open.

## 1. Source and manifest audit

The scan searched for `MIT`, `Apache`, `Apache-2.0` and `SPDX-License-Identifier` across `.rs .py .ts .js .toml .json .md .yml .yaml` files and the `Dockerfile`. It ran once over tracked files, then again with `--no-ignore` to catch generated artifacts. `.venv/`, `target/`, `node_modules/` and lockfiles were excluded.

- **No SPDX license headers** appear in any first-party source file.
- `LICENSE` contains the SSPL v1 text. No `LICENSE-MIT` or `LICENSE-APACHE` file exists.

| Manifest | Package | License field | Status |
|---|---|---|---|
| `Cargo.toml` (workspace) | - | `license = "SSPL-1.0"` | PASS |
| `crates/engine-core/Cargo.toml` | engine-core | `license.workspace = true` | PASS |
| `crates/grpc-server/Cargo.toml` | grpc-server | `license.workspace = true` | PASS |
| `crates/pyo3-bindings/Cargo.toml` | pyo3-bindings | `license.workspace = true` | PASS |
| `crates/temporal-math/Cargo.toml` | temporal-math | `license.workspace = true` | PASS |
| `crates/vllm-kv-store/Cargo.toml` | vllm-kv-store | `license.workspace = true` | PASS |
| `benchmarks/Cargo.toml` | thg-benchmarks (`publish = false`) | `license.workspace = true` | PASS |
| `crates/pyo3-bindings/pyproject.toml` | hypergraph-engine-native | `SSPL-1.0` + Proprietary classifier | PASS |
| `integrations/python-client/pyproject.toml` | hypergraph-client | `SSPL-1.0` + Proprietary classifier (added) | PASS |
| `integrations/langchain/pyproject.toml` | langchain-hypergraph | `SSPL-1.0` + Proprietary classifier (added) | PASS |
| `integrations/llamaindex/pyproject.toml` | llama-index-hypergraph | `SSPL-1.0` + Proprietary classifier (added) | PASS |
| `python/pyproject.toml` | hypergraph-offloader | `SSPL-1.0` + Proprietary classifier (added) | PASS |
| `python/evaluation/pyproject.toml` | hypergraph-evaluation (not published) | `SSPL-1.0` | PASS |
| `integrations/vercel-ai-sdk/package.json` | @temporal-hypergraph/sdk | `"license": "SSPL-1.0"` | PASS |

### Matches deliberately left in place

| Location | Why it stays |
|---|---|
| `README.md:501`, `docs/LAUNCH_POST.md:223`, `docs/GTM_AUDIT_REPORT.md:28`, v2.0.0 GitHub release notes | These are accurate historical disclosures that v1.0.1 and earlier were published under `MIT OR Apache-2.0`. Copies obtained under those grants can't be relicensed retroactively, so deleting the disclosures would misstate the licensing history. None of them claims that v2.0.0+ is permissive. |
| `integrations/vercel-ai-sdk/package-lock.json` (`"license": "MIT"` entries) | These are the licenses of third-party dependencies, not ours. Changing them would misrepresent those packages. |
| `scripts/kaggle_job.sh:64` (`LICENSE-APACHE LICENSE-MIT`) | This is a file list used to archive older git refs where those files existed. Missing paths are skipped. It makes no license claim. |

## 2. Live registry audit

The audit script queried `https://pypi.org/pypi/{pkg}/json`, `https://pypi.org/pypi/{pkg}/{version}/json` and `https://registry.npmjs.org/@temporal-hypergraph%2Fsdk`.

| Package | Version | Check | Result |
|---|---|---|---|
| hypergraph-engine-native | 2.0.0 | `license = SSPL-1.0` | PASS |
| hypergraph-engine-native | 2.0.1 | `license = SSPL-1.0`, Proprietary classifier | PASS |
| hypergraph-engine-native | 1.0.1 | all 21 files `yanked: true` | PASS |
| hypergraph-client | 2.0.0 | `license = SSPL-1.0` | PASS |
| langchain-hypergraph | 2.0.0 | `license = SSPL-1.0` | PASS |
| llama-index-hypergraph | 2.0.0 | `license = SSPL-1.0` | PASS |
| hypergraph-offloader | 2.0.0 | `license = SSPL-1.0` | PASS |
| hypergraph-client, langchain-hypergraph, llama-index-hypergraph, hypergraph-offloader | 2.0.1 | - | NOT PUBLISHED (see open action 1) |
| @temporal-hypergraph/sdk | latest = 2.0.1 | `"license": "SSPL-1.0"` | PASS |
| @temporal-hypergraph/sdk | 1.0.1 | `deprecated` set | PASS |
| @temporal-hypergraph/sdk | 0.0.0-stage | `deprecated` set | PASS |

Notes:
- The four pure-Python SDKs have **no pre-2.0.0 releases on PyPI**: their first upload was 2.0.0. So there is nothing to yank, and no permissively licensed version of them was ever distributed through PyPI.
- `hypergraph-engine-native` 2.0.0 was uploaded without license classifiers. Its `license` field is `SSPL-1.0`, which is enough. PyPI metadata for an existing release can't be edited.

## 3. Docker and Git release audit

- `.github/workflows/publish.yml:239` sets `org.opencontainers.image.licenses=SSPL-1.0` on the GHCR image. PASS
- `Dockerfile` had no license label, so a local `docker build` produced an unlabeled image. **Fixed:** the runtime stage now includes `LABEL org.opencontainers.image.licenses="SSPL-1.0"`.
- Tags: `v1.0.0`, `v1.0.1`, `v2.0.0` (annotated, at `6476251`) and `v2.0.1` (lightweight, at `b27963c`; fetched from origin during the audit).
  - At both v2 tags, `LICENSE` is the SSPL text and `Cargo.toml` has `license = "SSPL-1.0"`. PASS
  - The tag messages make no license claims. PASS
  - The GitHub release notes for v2.0.0 announce the SSPL change and mention MIT/Apache only for v1.0.1 and earlier. The v2.0.1 notes make no license claims. PASS

## Fixes applied

| File | Change |
|---|---|
| `python/evaluation/hypergraph_evaluation.egg-info/` (gitignored) | Deleted. Its stale `PKG-INFO` (version 1.0.0) said `License: MIT OR Apache-2.0` and could leak into a locally built sdist. |
| `integrations/python-client/pyproject.toml`, `integrations/langchain/pyproject.toml`, `integrations/llamaindex/pyproject.toml`, `python/pyproject.toml` | Added `classifiers = ["License :: Other/Proprietary License"]` so every published Python package carries the same license classifier as `hypergraph-engine-native`. Takes effect on the next upload. |
| `Dockerfile` | Added `LABEL org.opencontainers.image.licenses="SSPL-1.0"` to the runtime stage. |

No registry writes (yank or deprecate) were needed or performed.

## Open actions

1. **2.0.1 of the four Python SDKs never reached PyPI.** This is a release-pipeline problem, not a license violation. In Publish run `42826170798` (tag v2.0.1), the `publish-python-sdks` job failed with `invalid-pending-publisher: valid token, but project already exists`. The same job also failed for v2.0.0 (run `42789267840`), but with an upload-time `400 Bad Request`, after the 2.0.0 files that are now on PyPI had been published. Fix: on PyPI, add a trusted publisher on each existing project (hypergraph-client, langchain-hypergraph, llama-index-hypergraph, hypergraph-offloader) for this repository and `publish.yml`, replacing the pending publisher. Then re-run the job.
2. When 2.0.1 (or 2.0.2) is uploaded, re-check that the Proprietary classifier appears on PyPI for all five packages.
