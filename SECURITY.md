# Security Policy

I take security seriously. The engine uses `unsafe` Rust and crosses FFI boundaries: the KV-cache block store copies directly between raw tensor pointers and memory-mapped segments, and the PyO3 bindings accept raw pointers from Python. A bug in these paths can cause memory corruption, not just a wrong answer, so I treat reports about them as the highest priority.

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Instead, report it privately using the **Report a vulnerability** button on the repository's [Security tab](https://github.com/Divyansh151005/Temporal-Hypergraph-Engine/security). Only the maintainer can see the report.

Please include:

- The affected version and component (for example the gRPC server, the PyO3 bindings or the KV-cache block store).
- Steps or a proof of concept that reproduces the issue.
- The impact you believe it has (memory corruption, data loss, authentication bypass, denial of service).

I will acknowledge your report, keep you updated while I investigate, and credit you in the advisory when the fix is released unless you prefer to stay anonymous.

## Supported versions

| Version | Supported |
| --- | --- |
| 2.x | Yes |
| Earlier than 2.0.0 | No |

## Scope

In scope:

- Memory-safety bugs in `unsafe` code, mainly `crates/vllm-kv-store/src/raw.rs`, `crates/pyo3-bindings` and WAL recovery in `crates/engine-core`.
- WAL or checkpoint bugs that lose or corrupt committed data.
- Bypassing API-key authentication on the gRPC server.
- Inputs that crash the server or exhaust its memory remotely.

Out of scope: vulnerabilities in third-party dependencies that are already publicly known (please report those upstream), and issues that require an attacker who already controls the host or the engine's data directory.
