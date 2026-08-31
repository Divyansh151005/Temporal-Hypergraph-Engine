use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use engine_core::{ConcurrentHypergraphEngine, EngineConfig, SyncMode};
use grpc_server::pb::hypergraph_engine_service_client::HypergraphEngineServiceClient;
use grpc_server::{serve_with_auth, ApiKeyCheck, Listen, MAX_MESSAGE_BYTES};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tokio::sync::oneshot;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use vllm_kv_store::{KvBlockStore, KvStoreConfig};

pub type Client = HypergraphEngineServiceClient<Channel>;

/// An in-process server on a temporary Unix socket; stops when dropped.
pub struct TestServer {
    pub dir: tempfile::TempDir,
    pub socket: PathBuf,
    shutdown: Option<oneshot::Sender<()>>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

pub async fn start_server() -> TestServer {
    start_server_with_auth(ApiKeyCheck::default()).await
}

/// Like [`start_server`], enforcing `auth` on every call.
pub async fn start_server_with_auth(auth: ApiKeyCheck) -> TestServer {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        ConcurrentHypergraphEngine::open_with_config(
            dir.path().join("graph"),
            EngineConfig {
                sync_mode: SyncMode::OsBuffered,
                ..EngineConfig::default()
            },
        )
        .unwrap(),
    );
    let kv = Arc::new(KvBlockStore::open(dir.path().join("kv"), KvStoreConfig::default()).unwrap());
    let socket = dir.path().join("hg.sock");
    let (tx, rx) = oneshot::channel();
    let listen = Listen::Unix(socket.clone());
    tokio::spawn(async move {
        serve_with_auth(engine, kv, listen, auth, async {
            let _ = rx.await;
        })
        .await
        .unwrap();
    });
    for _ in 0..200 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    TestServer {
        dir,
        socket,
        shutdown: Some(tx),
    }
}

pub async fn connect(socket: &Path) -> Client {
    let socket = socket.to_path_buf();
    let channel = Endpoint::try_from("http://[::]:50051")
        .unwrap()
        .connect_with_connector(service_fn(move |_: Uri| {
            let socket = socket.clone();
            async move { Ok::<_, std::io::Error>(TokioIo::new(UnixStream::connect(socket).await?)) }
        }))
        .await
        .unwrap();
    HypergraphEngineServiceClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES)
}
