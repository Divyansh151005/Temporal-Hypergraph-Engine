# Publishing the Rust crates to crates.io

The Python wheels and the npm SDK are published automatically by
`.github/workflows/publish.yml` when a GitHub Release is published. The Rust
crates are published by hand, because crates.io only accepts a crate once
every dependency it names is already on the index.

## Order

Publish strictly in this order, one crate at a time:

1. `temporal-math` (no internal dependencies)
2. `engine-core` (depends on `temporal-math`)
3. `vllm-kv-store` (depends on `engine-core`)
4. `grpc-server` (depends on `engine-core` and `vllm-kv-store`)
5. `pyo3-bindings` (optional; Python users get it from PyPI as
   `hypergraph-engine-native`)

`thg-benchmarks` has `publish = false` and is never published.

Wait for each crate to appear on crates.io before publishing the next one.
Recent versions of `cargo publish` wait for the index on their own; if a later
crate fails with "no matching package named ...", wait a minute and retry.

## Steps

```bash
git checkout v2.0.0            # the release tag, with a clean tree
cargo login                    # token from https://crates.io/settings/tokens

for crate in temporal-math engine-core vllm-kv-store grpc-server; do
  cargo publish --dry-run -p "$crate"
  cargo publish -p "$crate"
done
```

The dry run for a crate can only resolve its internal dependencies once those
are already published, so run it crate by crate as above, not all up front.

## Notes

- **`engine-core` is already taken on crates.io** by an unrelated project.
  Before the first publish, rename the package (for example
  `temporal-hypergraph-core`) and keep the Rust import name with
  `engine-core = { package = "temporal-hypergraph-core", ... }` in the root
  `Cargo.toml`. Check every other name with
  `curl -s -o /dev/null -w '%{http_code}' https://crates.io/api/v1/crates/<name>`:
  a `404` means the name is free.
- Internal crates are declared in `[workspace.dependencies]` with both `path`
  and `version`. crates.io drops `path` and requires `version`; keep it in
  sync with `workspace.package.version`.
- All crates share `version.workspace`, so bump the version once in the root
  `Cargo.toml` and republish every crate in the order above.
- `grpc-server` builds its gRPC stubs with `protoc-bin-vendored`, so no system
  `protoc` is needed to publish or install it.
- A published version can be yanked (`cargo yank --version 1.0.0 -p <crate>`)
  but never overwritten or deleted. Fix forward with a new patch version.
