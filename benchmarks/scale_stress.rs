//! Extreme-scale ingest: streams up to 10M synthetic temporal hyperedges into
//! the engine and records sustained/peak throughput, per-batch and
//! per-request write latency, and memory growth per million edges.
//!
//! ```text
//! cargo bench -p thg-benchmarks --bench scale_stress -- \
//!   --edges 10000000 --modes memory,wal-buffered,wal-fsync --fsync-edges 1000000 \
//!   --out results/phase7/scale_stress.json
//! ```

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use engine_core::{ConcurrentHypergraphEngine, EngineConfig, SyncMode};
use serde::Serialize;
use thg_benchmarks::{
    histogram, rss_bytes, slope, workspace_path, write_json, Args, CountingAlloc, Environment,
    LatencySummary, Workload,
};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Memory,
    WalBuffered,
    WalFsync,
}

impl Mode {
    fn parse(s: &str) -> Self {
        match s {
            "memory" => Self::Memory,
            "wal-buffered" => Self::WalBuffered,
            "wal-fsync" => Self::WalFsync,
            other => panic!("unknown mode {other:?}; expected memory, wal-buffered or wal-fsync"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::WalBuffered => "wal-buffered",
            Self::WalFsync => "wal-fsync",
        }
    }

    fn sync_mode(self) -> Option<SyncMode> {
        match self {
            Self::Memory => None,
            Self::WalBuffered => Some(SyncMode::OsBuffered),
            Self::WalFsync => Some(SyncMode::Fsync),
        }
    }
}

struct Config {
    edges: usize,
    nodes: usize,
    skew: f64,
    batch: usize,
    fsync_edges: usize,
    sample_every: usize,
    rss_guard: f64,
    single_threads: usize,
    single_secs: u64,
    single_preload: usize,
    seed: u64,
}

#[derive(Debug, Serialize)]
struct MemorySample {
    edges: usize,
    elapsed_secs: f64,
    rss_bytes: usize,
    heap_live_bytes: usize,
}

#[derive(Debug, Serialize)]
struct SingleWrite {
    threads: usize,
    preloaded_edges: usize,
    duration_secs: f64,
    writes: u64,
    writes_per_sec: f64,
    latency: LatencySummary,
}

#[derive(Debug, Serialize)]
struct ModeResult {
    mode: &'static str,
    target_edges: usize,
    ingested_edges: usize,
    failed_inserts: usize,
    stop_reason: Option<String>,
    wall_secs: f64,
    engine_busy_secs: f64,
    producer_stall_secs: f64,
    sustained_edges_per_sec: f64,
    /// Cumulative rate at the last sample where the live heap still fit in RSS, i.e. before paging.
    resident_edges: usize,
    resident_edges_per_sec: f64,
    engine_edges_per_sec: f64,
    peak_edges_per_sec: f64,
    batch_size: usize,
    batch_latency: LatencySummary,
    amortized_ns_per_edge: f64,
    live_nodes: u32,
    live_hyperedges: u32,
    max_node_degree: usize,
    heap_bytes_per_million_edges: f64,
    rss_bytes_per_million_edges: f64,
    engine_heap_bytes: usize,
    peak_rss_bytes: usize,
    wal_disk_bytes: Option<u64>,
    memory_samples: Vec<MemorySample>,
    single_write: Option<SingleWrite>,
}

#[derive(Debug, Serialize)]
struct Report {
    benchmark: &'static str,
    environment: Environment,
    config: serde_json::Value,
    modes: Vec<ModeResult>,
}

fn dir_size(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn open_engine(
    mode: Mode,
    batch: usize,
) -> (ConcurrentHypergraphEngine, Option<tempfile::TempDir>) {
    let config = EngineConfig {
        max_batch: batch,
        queue_capacity: (batch * 2).max(16 * 1024),
        ..EngineConfig::default()
    };
    match mode.sync_mode() {
        None => (ConcurrentHypergraphEngine::in_memory(config), None),
        Some(sync_mode) => {
            let dir = tempfile::tempdir().expect("temp dir");
            let engine = ConcurrentHypergraphEngine::open_with_config(
                dir.path(),
                EngineConfig {
                    sync_mode,
                    ..config
                },
            )
            .expect("open engine");
            (engine, Some(dir))
        }
    }
}

fn ingest(mode: Mode, cfg: &Config, memory_limit: usize) -> ModeResult {
    let target = if mode == Mode::WalFsync {
        cfg.edges.min(cfg.fsync_edges)
    } else {
        cfg.edges
    };
    eprintln!(
        "[{}] ingesting {target} edges in batches of {}",
        mode.name(),
        cfg.batch
    );
    let (engine, dir) = open_engine(mode, cfg.batch);
    let heap_start = CountingAlloc::live();
    let rss_start = rss_bytes();

    let (tx, rx) = crossbeam_channel::bounded(4);
    let (batch, nodes, skew, seed) = (cfg.batch, cfg.nodes, cfg.skew, cfg.seed);
    let producer = thread::spawn(move || {
        let mut workload = Workload::new(seed, nodes, skew);
        let mut sent = 0;
        while sent < target {
            let n = batch.min(target - sent);
            if tx.send(workload.batch(n)).is_err() {
                return;
            }
            sent += n;
        }
    });

    let mut batch_hist = histogram();
    let mut samples = vec![MemorySample {
        edges: 0,
        elapsed_secs: 0.0,
        rss_bytes: rss_start,
        heap_live_bytes: heap_start,
    }];
    let (mut ingested, mut failed, mut next_sample) = (0usize, 0usize, cfg.sample_every);
    let (mut busy, mut stall) = (Duration::ZERO, Duration::ZERO);
    let mut peak_rate = 0.0f64;
    let mut peak_rss = rss_start;
    let mut stop_reason = None;
    let started = Instant::now();
    let (mut window_start, mut window_edges) = (started, 0usize);

    loop {
        let wait = Instant::now();
        let Ok(edges) = rx.recv() else { break };
        stall += wait.elapsed();
        let n = edges.len();
        let t0 = Instant::now();
        let results = engine.insert_batch(edges);
        let dt = t0.elapsed();
        busy += dt;
        batch_hist.saturating_record(dt.as_nanos() as u64);
        failed += results.iter().filter(|r| r.is_err()).count();
        drop(results);
        ingested += n;
        window_edges += n;

        let window = window_start.elapsed();
        if window >= Duration::from_secs(1) {
            peak_rate = peak_rate.max(window_edges as f64 / window.as_secs_f64());
            window_start = Instant::now();
            window_edges = 0;
        }
        if ingested >= next_sample || ingested == target {
            let rss = rss_bytes();
            peak_rss = peak_rss.max(rss);
            samples.push(MemorySample {
                edges: ingested,
                elapsed_secs: started.elapsed().as_secs_f64(),
                rss_bytes: rss,
                heap_live_bytes: CountingAlloc::live(),
            });
            eprintln!(
                "[{}] {ingested:>10} edges  {:>8.0} edges/s  rss {:>6} MiB  heap {:>6} MiB",
                mode.name(),
                ingested as f64 / started.elapsed().as_secs_f64(),
                rss >> 20,
                CountingAlloc::live() >> 20,
            );
            while next_sample <= ingested {
                next_sample += cfg.sample_every;
            }
            let heap = CountingAlloc::live();
            let reason = if rss > memory_limit {
                Some(format!(
                    "RSS {} MiB exceeded the {} MiB guard",
                    rss >> 20,
                    memory_limit >> 20
                ))
            } else if heap > 1 << 30 && heap as f64 > rss as f64 * 1.3 {
                Some(format!(
                    "memory pressure: live heap {} MiB exceeds RSS {} MiB, so the OS is paging the graph out",
                    heap >> 20,
                    rss >> 20
                ))
            } else {
                None
            };
            if let Some(reason) = reason {
                eprintln!("[{}] stopping early: {reason}", mode.name());
                stop_reason = Some(reason);
                break;
            }
        }
    }
    let wall = started.elapsed();
    drop(rx);
    producer.join().expect("producer thread");
    if peak_rate == 0.0 {
        peak_rate = ingested as f64 / wall.as_secs_f64();
    }

    let snapshot = engine.snapshot();
    let max_degree = snapshot
        .iter_nodes()
        .map(|n| n.hyperedges.len())
        .max()
        .unwrap_or(0);
    let (live_nodes, live_hyperedges) =
        (snapshot.live_node_count(), snapshot.live_hyperedge_count());
    drop(snapshot);
    let heap_points: Vec<(f64, f64)> = samples
        .iter()
        .map(|s| (s.edges as f64, s.heap_live_bytes as f64))
        .collect();
    let rss_points: Vec<(f64, f64)> = samples
        .iter()
        .map(|s| (s.edges as f64, s.rss_bytes as f64))
        .collect();
    let (resident_edges, resident_secs) = samples
        .iter()
        .rev()
        .find(|s| s.edges > 0 && s.heap_live_bytes <= s.rss_bytes)
        .map_or((ingested, wall.as_secs_f64()), |s| {
            (s.edges, s.elapsed_secs)
        });
    let engine_heap = CountingAlloc::live().saturating_sub(heap_start);
    let wal_disk_bytes = dir.as_ref().map(|d| dir_size(d.path()));
    let single_write = (cfg.single_secs > 0).then(|| single_write(mode, cfg));

    let result = ModeResult {
        mode: mode.name(),
        target_edges: target,
        ingested_edges: ingested,
        failed_inserts: failed,
        stop_reason,
        wall_secs: wall.as_secs_f64(),
        engine_busy_secs: busy.as_secs_f64(),
        producer_stall_secs: stall.as_secs_f64(),
        sustained_edges_per_sec: ingested as f64 / wall.as_secs_f64(),
        resident_edges,
        resident_edges_per_sec: resident_edges as f64 / resident_secs.max(f64::EPSILON),
        engine_edges_per_sec: ingested as f64 / busy.as_secs_f64().max(f64::EPSILON),
        peak_edges_per_sec: peak_rate,
        batch_size: cfg.batch,
        batch_latency: LatencySummary::from_ns(&batch_hist),
        amortized_ns_per_edge: busy.as_nanos() as f64 / ingested.max(1) as f64,
        live_nodes,
        live_hyperedges,
        max_node_degree: max_degree,
        heap_bytes_per_million_edges: slope(&heap_points) * 1e6,
        rss_bytes_per_million_edges: slope(&rss_points) * 1e6,
        engine_heap_bytes: engine_heap,
        peak_rss_bytes: peak_rss,
        wal_disk_bytes,
        memory_samples: samples,
        single_write,
    };
    drop(engine);
    drop(dir);
    result
}

/// Per-request latency through the group-commit queue: several client
/// threads each issue one `insert` at a time against a preloaded engine.
fn single_write(mode: Mode, cfg: &Config) -> SingleWrite {
    let (engine, _dir) = open_engine(mode, cfg.batch);
    let engine = Arc::new(engine);
    let mut workload = Workload::new(cfg.seed ^ 0x5157, cfg.nodes, cfg.skew);
    let mut loaded = 0;
    while loaded < cfg.single_preload {
        let n = cfg.batch.min(cfg.single_preload - loaded);
        engine.insert_batch(workload.batch(n));
        loaded += n;
    }
    eprintln!(
        "[{}] single-write latency: {} threads x {}s on {loaded} preloaded edges",
        mode.name(),
        cfg.single_threads,
        cfg.single_secs
    );
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    let handles: Vec<_> = (0..cfg.single_threads)
        .map(|t| {
            let (engine, stop) = (Arc::clone(&engine), Arc::clone(&stop));
            let mut workload = Workload::new(cfg.seed + 101 + t as u64, cfg.nodes, cfg.skew);
            thread::spawn(move || {
                let mut hist = histogram();
                while !stop.load(Ordering::Relaxed) {
                    let edge = workload.edge();
                    let t0 = Instant::now();
                    engine.insert(edge).expect("single insert");
                    hist.saturating_record(t0.elapsed().as_nanos() as u64);
                }
                hist
            })
        })
        .collect();
    thread::sleep(Duration::from_secs(cfg.single_secs));
    stop.store(true, Ordering::Relaxed);
    let mut merged = histogram();
    for h in handles {
        merged
            .add(h.join().expect("writer thread"))
            .expect("merge histogram");
    }
    let elapsed = started.elapsed().as_secs_f64();
    SingleWrite {
        threads: cfg.single_threads,
        preloaded_edges: loaded,
        duration_secs: elapsed,
        writes: merged.len(),
        writes_per_sec: merged.len() as f64 / elapsed,
        latency: LatencySummary::from_ns(&merged),
    }
}

fn main() {
    let args = Args::parse();
    let edges: usize = args.get("edges", 10_000_000);
    let cfg = Config {
        edges,
        nodes: args.get("nodes", (edges / 5).max(1_000)),
        skew: args.get("skew", 2.0),
        batch: args.get("batch", 65_536),
        fsync_edges: args.get("fsync-edges", 1_000_000),
        sample_every: args.get("sample-every", (edges / 20).max(1)),
        rss_guard: args.get("rss-guard", 0.75),
        single_threads: args.get("single-threads", 4),
        single_secs: args.get("single-secs", 10),
        single_preload: args.get("single-preload", edges.min(1_000_000)),
        seed: args.get("seed", 42),
    };
    let modes: Vec<Mode> = args
        .string("modes", "memory,wal-buffered,wal-fsync")
        .split(',')
        .map(Mode::parse)
        .collect();
    let out = workspace_path(&args.string("out", "results/phase7/scale_stress.json"));
    let environment = Environment::capture();
    let memory_limit = (environment.memory_bytes as f64 * cfg.rss_guard) as usize;

    let mut results = Vec::new();
    for mode in modes {
        results.push(ingest(mode, &cfg, memory_limit));
        let report = Report {
            benchmark: "scale_stress",
            environment: environment.clone(),
            config: serde_json::json!({
                "edges": cfg.edges,
                "nodes": cfg.nodes,
                "skew": cfg.skew,
                "batch": cfg.batch,
                "fsync_edges": cfg.fsync_edges,
                "sample_every": cfg.sample_every,
                "rss_guard": cfg.rss_guard,
                "single_threads": cfg.single_threads,
                "single_secs": cfg.single_secs,
                "single_preload": cfg.single_preload,
                "seed": cfg.seed,
                "arity": "2..=4",
                "event_types": 32,
                "closed_fraction": 0.1,
            }),
            modes: std::mem::take(&mut results),
        };
        write_json(&out, &report);
        results = report.modes;
    }
}
