//! Listener setup and serving.

use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use engine_core::ConcurrentHypergraphEngine;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;
use vllm_kv_store::KvBlockStore;

use crate::auth::ApiKeyCheck;
use crate::pb::hypergraph_engine_service_server::HypergraphEngineServiceServer;
use crate::service::HypergraphService;

/// Message size limit; an 8 MiB KV block exceeds Tonic's 4 MiB default.
pub const MAX_MESSAGE_BYTES: usize = 16 << 20;

/// Where the server listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Listen {
    /// `tcp://host:port`
    Tcp(SocketAddr),
    /// `unix:///path/to.sock`
    Unix(PathBuf),
}

impl FromStr for Listen {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if let Some(addr) = text.strip_prefix("tcp://") {
            addr.parse()
                .map(Self::Tcp)
                .map_err(|e| format!("invalid TCP address {addr:?}: {e}"))
        } else if let Some(path) = text.strip_prefix("unix://") {
            Ok(Self::Unix(PathBuf::from(path)))
        } else {
            Err(format!(
                "listen address {text:?} must start with tcp:// or unix://"
            ))
        }
    }
}

/// The Tonic service with message limits and API-key checks applied.
pub fn grpc_service(
    service: HypergraphService,
    auth: ApiKeyCheck,
) -> InterceptedService<HypergraphEngineServiceServer<HypergraphService>, ApiKeyCheck> {
    InterceptedService::new(
        HypergraphEngineServiceServer::new(service)
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES),
        auth,
    )
}

/// Serves without authentication until `shutdown` resolves.
pub async fn serve(
    engine: Arc<ConcurrentHypergraphEngine>,
    kv: Arc<KvBlockStore>,
    listen: Listen,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with_auth(engine, kv, listen, ApiKeyCheck::default(), shutdown).await
}

/// Serves until `shutdown` resolves, enforcing `auth` on every call.
pub async fn serve_with_auth(
    engine: Arc<ConcurrentHypergraphEngine>,
    kv: Arc<KvBlockStore>,
    listen: Listen,
    auth: ApiKeyCheck,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if auth.enabled() {
        tracing::info!("API key authentication enabled");
    }
    let service = HypergraphService::new(engine, kv);
    let router = Server::builder()
        .layer(service.in_flight_layer())
        .add_service(grpc_service(service, auth));
    match listen {
        Listen::Tcp(addr) => {
            tracing::info!(%addr, "serving gRPC over TCP");
            router.serve_with_shutdown(addr, shutdown).await?;
        }
        Listen::Unix(path) => {
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            let listener = UnixListener::bind(&path)?;
            tracing::info!(path = %path.display(), "serving gRPC over UDS");
            let result = router
                .serve_with_incoming_shutdown(UnixListenerStream::new(listener), shutdown)
                .await;
            let _ = std::fs::remove_file(&path);
            result?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_listen_addresses() {
        assert_eq!(
            "tcp://0.0.0.0:50051".parse::<Listen>().unwrap(),
            Listen::Tcp("0.0.0.0:50051".parse().unwrap())
        );
        assert_eq!(
            "unix:///tmp/hg.sock".parse::<Listen>().unwrap(),
            Listen::Unix("/tmp/hg.sock".into())
        );
        assert!("http://x".parse::<Listen>().is_err());
    }
}
