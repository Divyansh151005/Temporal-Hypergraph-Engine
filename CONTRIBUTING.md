# Contributing

Thanks for your interest in the Temporal Hypergraph Engine. Bug reports, benchmark critiques and pull requests are all welcome.

## Reporting issues

Open a GitHub issue with the engine version (`hypergraph-server --version`, or the package version from `pip show` / `npm ls`), your platform, and a minimal reproduction. For benchmark disputes, include the exact command line and the JSON output.

## Development setup

You need the Rust toolchain pinned in `rust-toolchain.toml`, Python 3.10+, and Node.js (version in `integrations/vercel-ai-sdk/.nvmrc`).

The checks below are the ones CI runs, so run them before opening a pull request:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# TypeScript SDK
cd integrations/vercel-ai-sdk && npm ci && npm run build && npx tsc --noEmit && npm test
```

Changes to the WAL or the concurrency code in `engine-core` are also checked in CI under Miri, Loom and the WAL corruption suites (see `.github/workflows/engine-validation.yml`).

## Pull requests

- Keep each pull request focused on one change, and add tests for any behavior change.
- Performance claims need a reproducible benchmark command and before/after numbers.
- Do not bump versions in pull requests; versions are bumped at release time.

## Contributor License Agreement (CLA)

All external contributors must agree to a CLA. You retain the copyright to your contribution, but you grant the project a perpetual, irrevocable license to use, modify, distribute, and sublicense the contribution. This grant exists strictly so the project can maintain its dual-licensing model (SSPL-1.0 alongside commercial licenses) and stay commercially viable in the future.

The project uses [cla-assistant](https://cla-assistant.io/). When you open your first pull request, it automatically comments with a link to the CLA; sign it once with your GitHub account and the check passes for that and all future pull requests. Pull requests from external contributors cannot be merged until the CLA has been signed.

## License

The codebase is licensed under the [Server Side Public License v1](LICENSE) (SSPL-1.0), and all contributions are accepted under it. See the License section of the [README](README.md#license) for what that means in practice.
