//! `hypergraph-server`: serves an engine directory over gRPC.
//!
//! ```text
//! hypergraph-server [--data-dir DIR] [--listen tcp://HOST:PORT | unix:///PATH]
//!                   [--embedding-dim N] [--sync fsync|buffered] [--api-key KEY]
//! ```
//! The default listen address is `tcp://0.0.0.0:$PORT` (port 50051 when
//! `PORT` is unset). KV blocks live under `DIR/kv`. When `--api-key` or
//! `HYPERGRAPH_API_KEY` is set, every call must send it as `x-api-key`.
//!
//! Python `grpcio` clients connecting over a Unix socket must set the
//! channel option `("grpc.default_authority", "localhost")`; otherwise they
//! send the socket path as `:authority`, which the HTTP/2 layer rejects.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use engine_core::{ConcurrentHypergraphEngine, EngineConfig, SyncMode};
use grpc_server::{serve_with_auth, ApiKeyCheck, Listen};
use vllm_kv_store::{KvBlockStore, KvStoreConfig};

struct Args {
    data_dir: PathBuf,
    listen: Listen,
    config: EngineConfig,
    api_key: Option<String>,
}

const USAGE: &str = "usage: hypergraph-server [--data-dir DIR] [--listen tcp://HOST:PORT|unix:///PATH] [--embedding-dim N] [--sync fsync|buffered] [--api-key KEY]";

fn parse_args() -> Result<Args, String> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "50051".into());
    let mut args = Args {
        data_dir: PathBuf::from("data"),
        listen: format!("tcp://0.0.0.0:{port}").parse()?,
        config: EngineConfig::default(),
        api_key: std::env::var("HYPERGRAPH_API_KEY").ok(),
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.into());
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--data-dir" => args.data_dir = value.into(),
            "--listen" => args.listen = value.parse()?,
            "--embedding-dim" => {
                args.config.embedding_dim = value
                    .parse()
                    .map_err(|e| format!("invalid --embedding-dim: {e}"))?;
            }
            "--sync" => {
                args.config.sync_mode = match value.as_str() {
                    "fsync" => SyncMode::Fsync,
                    "buffered" => SyncMode::OsBuffered,
                    other => return Err(format!("invalid --sync {other:?}\n{USAGE}")),
                };
            }
            "--api-key" => args.api_key = Some(value),
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
    }
    Ok(args)
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!(%err, "failed to listen for Ctrl-C");
    }
    tracing::info!("shutting down");
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let engine = match ConcurrentHypergraphEngine::open_with_config(&args.data_dir, args.config) {
        Ok(engine) => Arc::new(engine),
        Err(err) => {
            eprintln!(
                "failed to open engine at {}: {err}",
                args.data_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let kv = match KvBlockStore::open(args.data_dir.join("kv"), KvStoreConfig::default()) {
        Ok(kv) => Arc::new(kv),
        Err(err) => {
            eprintln!("failed to open KV store: {err}");
            return ExitCode::FAILURE;
        }
    };
    let auth = ApiKeyCheck::new(args.api_key.as_deref());
    if let Err(err) =
        serve_with_auth(engine.clone(), kv, args.listen, auth, shutdown_signal()).await
    {
        eprintln!("server error: {err}");
        return ExitCode::FAILURE;
    }
    if let Err(err) = engine.checkpoint() {
        eprintln!("final checkpoint failed: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
