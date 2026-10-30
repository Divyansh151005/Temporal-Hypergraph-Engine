//! Lock-free read scaling: N reader threads run 3-hop traversals while one
//! heavy writer ingests back-to-back batches. The same workload is repeated
//! over a conventional `RwLock`-protected graph as a control.
//!
//! Contention overhead is measured at a fixed reader count (with writer vs.
//! readers only), so CPU oversubscription above the core count does not get
//! mistaken for lock contention.
//!
//! ```text
//! cargo bench -p thg-benchmarks --bench concurrency_scaling -- \
//!   --readers 1,8,16,32,64 --duration-secs 10 --preload 1000000 \
//!   --out results/phase7/concurrency.json
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use engine_core::{BipartiteArena, ConcurrentHypergraphEngine, EngineConfig, NodeId};
use hdrhistogram::Histogram;
use parking_lot::RwLock;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::Serialize;
use thg_benchmarks::{
    histogram, rss_bytes, workspace_path, write_json, Args, CountingAlloc, Environment,
    LatencySummary, Workload,
};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    ArcSwapReadersOnly,
    ArcSwapWithWriter,
    RwLockReadersOnly,
    RwLockWithWriter,
}

impl Scenario {
    const ALL: [Scenario; 4] = [
        Self::ArcSwapReadersOnly,
        Self::ArcSwapWithWriter,
        Self::RwLockReadersOnly,
        Self::RwLockWithWriter,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::ArcSwapReadersOnly => "arcswap_readers_only",
            Self::ArcSwapWithWriter => "arcswap_with_writer",
            Self::RwLockReadersOnly => "rwlock_readers_only",
            Self::RwLockWithWriter => "rwlock_with_writer",
        }
    }

    fn has_writer(self) -> bool {
        matches!(self, Self::ArcSwapWithWriter | Self::RwLockWithWriter)
    }

    fn uses_rwlock(self) -> bool {
        matches!(self, Self::RwLockReadersOnly | Self::RwLockWithWriter)
    }
}

#[derive(Clone)]
struct Config {
    preload: usize,
    nodes: usize,
    skew: f64,
    hops: usize,
    limit: usize,
    writer_batch: usize,
    writer_rate: u64,
    duration: Duration,
    warmup: Duration,
    seed: u64,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    scenario: &'static str,
    readers: usize,
    active_cores: usize,
    duration_secs: f64,
    reads: u64,
    reads_per_sec: f64,
    reads_per_sec_per_core: f64,
    mean_edges_per_read: f64,
    read_latency: LatencySummary,
    snapshot_acquire_latency: LatencySummary,
    snapshot_release_latency: LatencySummary,
    writer_edges: u64,
    writer_edges_per_sec: f64,
    writer_publishes: u64,
    final_hyperedges: u32,
    rss_bytes: usize,
}

#[derive(Debug, Serialize)]
struct Derived {
    readers: usize,
    arcswap_p50_overhead: Option<f64>,
    arcswap_p99_overhead: Option<f64>,
    rwlock_p50_overhead: Option<f64>,
    rwlock_p99_overhead: Option<f64>,
    arcswap_scaling_efficiency: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    benchmark: &'static str,
    environment: Environment,
    config: serde_json::Value,
    timer_overhead_ns: f64,
    results: Vec<ScenarioResult>,
    derived: Vec<Derived>,
}

fn build(cfg: &Config) -> (Arc<ConcurrentHypergraphEngine>, Vec<NodeId>) {
    let engine = ConcurrentHypergraphEngine::in_memory(EngineConfig {
        max_batch: 8_192,
        queue_capacity: 32 * 1024,
        ..EngineConfig::default()
    });
    let mut workload = Workload::new(cfg.seed, cfg.nodes, cfg.skew);
    let mut loaded = 0;
    while loaded < cfg.preload {
        let n = 8_192.min(cfg.preload - loaded);
        engine.insert_batch(workload.batch(n));
        loaded += n;
    }
    let snapshot = engine.snapshot();
    let step = (snapshot.live_node_count() as usize / 4_096).max(1);
    let seeds: Vec<NodeId> = snapshot.iter_nodes().step_by(step).map(|n| n.id).collect();
    (Arc::new(engine), seeds)
}

/// Median cost of an empty `Instant::now()` pair, subtracted mentally from
/// the sub-microsecond acquire timings.
fn timer_overhead_ns() -> f64 {
    let mut samples: Vec<u128> = (0..100_000)
        .map(|_| {
            let t = Instant::now();
            t.elapsed().as_nanos()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

struct ReaderOut {
    reads: Histogram<u64>,
    acquires: Histogram<u64>,
    releases: Histogram<u64>,
    edges: u64,
}

fn run(
    scenario: Scenario,
    readers: usize,
    engine: Arc<ConcurrentHypergraphEngine>,
    seeds: Arc<Vec<NodeId>>,
    cfg: &Config,
) -> ScenarioResult {
    let locked: Arc<RwLock<Arc<BipartiteArena>>> = Arc::new(RwLock::new(engine.snapshot()));
    let stop = Arc::new(AtomicBool::new(false));
    let measuring = Arc::new(AtomicBool::new(false));
    let writer_edges = Arc::new(AtomicU64::new(0));
    let publishes = Arc::new(AtomicU64::new(0));
    let as_of = i64::MAX - 1;

    let writer = scenario.has_writer().then(|| {
        let (engine, locked, stop, measuring) = (
            Arc::clone(&engine),
            Arc::clone(&locked),
            Arc::clone(&stop),
            Arc::clone(&measuring),
        );
        let (writer_edges, publishes) = (Arc::clone(&writer_edges), Arc::clone(&publishes));
        let cfg = cfg.clone();
        thread::spawn(move || {
            let mut workload = Workload::new(cfg.seed ^ 0xD00D, cfg.nodes, cfg.skew);
            let started = Instant::now();
            let mut written = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let batch = workload.batch(cfg.writer_batch);
                if scenario.uses_rwlock() {
                    let mut guard = locked.write();
                    engine.insert_batch(batch);
                    *guard = engine.snapshot();
                } else {
                    engine.insert_batch(batch);
                }
                written += cfg.writer_batch as u64;
                if measuring.load(Ordering::Relaxed) {
                    writer_edges.fetch_add(cfg.writer_batch as u64, Ordering::Relaxed);
                    publishes.fetch_add(1, Ordering::Relaxed);
                }
                if cfg.writer_rate > 0 {
                    let due = Duration::from_secs_f64(written as f64 / cfg.writer_rate as f64);
                    if let Some(wait) = due.checked_sub(started.elapsed()) {
                        thread::sleep(wait);
                    }
                }
            }
        })
    });

    let handles: Vec<_> = (0..readers)
        .map(|t| {
            let (engine, locked, stop, measuring, seeds) = (
                Arc::clone(&engine),
                Arc::clone(&locked),
                Arc::clone(&stop),
                Arc::clone(&measuring),
                Arc::clone(&seeds),
            );
            let (hops, limit, seed) = (cfg.hops, cfg.limit, cfg.seed);
            thread::spawn(move || {
                let mut rng = SmallRng::seed_from_u64(seed + 1_000 + t as u64);
                let mut out = ReaderOut {
                    reads: histogram(),
                    acquires: histogram(),
                    releases: histogram(),
                    edges: 0,
                };
                while !stop.load(Ordering::Relaxed) {
                    let start = [seeds[rng.gen_range(0..seeds.len())]];
                    let t0 = Instant::now();
                    let (acquired, released, reached) = if scenario.uses_rwlock() {
                        let guard = locked.read();
                        let acquired = t0.elapsed();
                        let reached = guard.traverse(&start, hops, as_of, limit).len();
                        let t1 = Instant::now();
                        drop(guard);
                        (acquired, t1.elapsed(), reached)
                    } else {
                        let guard = engine.graph_snapshot.load();
                        let acquired = t0.elapsed();
                        let reached = guard.traverse(&start, hops, as_of, limit).len();
                        let t1 = Instant::now();
                        drop(guard);
                        (acquired, t1.elapsed(), reached)
                    };
                    let total = t0.elapsed();
                    if measuring.load(Ordering::Relaxed) {
                        out.reads.saturating_record(total.as_nanos() as u64);
                        out.acquires.saturating_record(acquired.as_nanos() as u64);
                        out.releases.saturating_record(released.as_nanos() as u64);
                        out.edges += reached as u64;
                    }
                }
                out
            })
        })
        .collect();

    thread::sleep(cfg.warmup);
    measuring.store(true, Ordering::Relaxed);
    let started = Instant::now();
    thread::sleep(cfg.duration);
    measuring.store(false, Ordering::Relaxed);
    let elapsed = started.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);

    let (mut reads, mut acquires, mut releases, mut edges) =
        (histogram(), histogram(), histogram(), 0u64);
    for h in handles {
        let out = h.join().expect("reader thread");
        reads.add(&out.reads).expect("merge reads");
        acquires.add(&out.acquires).expect("merge acquires");
        releases.add(&out.releases).expect("merge releases");
        edges += out.edges;
    }
    if let Some(w) = writer {
        w.join().expect("writer thread");
    }

    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    let active_cores = readers.min(cores);
    let reads_per_sec = reads.len() as f64 / elapsed;
    let written = writer_edges.load(Ordering::Relaxed);
    let result = ScenarioResult {
        scenario: scenario.name(),
        readers,
        active_cores,
        duration_secs: elapsed,
        reads: reads.len(),
        reads_per_sec,
        reads_per_sec_per_core: reads_per_sec / active_cores as f64,
        mean_edges_per_read: edges as f64 / reads.len().max(1) as f64,
        read_latency: LatencySummary::from_ns(&reads),
        snapshot_acquire_latency: LatencySummary::from_ns(&acquires),
        snapshot_release_latency: LatencySummary::from_ns(&releases),
        writer_edges: written,
        writer_edges_per_sec: written as f64 / elapsed,
        writer_publishes: publishes.load(Ordering::Relaxed),
        final_hyperedges: engine.snapshot().live_hyperedge_count(),
        rss_bytes: rss_bytes(),
    };
    eprintln!(
        "{:>22} readers={readers:<3} reads/s={:>10.0} p50={:>8.1}us p99={:>8.1}us acquire_p99={:>7.2}us release_p99={:>7.2}us writer={:>9.0} edges/s",
        result.scenario,
        result.reads_per_sec,
        result.read_latency.p50_us,
        result.read_latency.p99_us,
        result.snapshot_acquire_latency.p99_us,
        result.snapshot_release_latency.p99_us,
        result.writer_edges_per_sec,
    );
    result
}

fn find(results: &[ScenarioResult], s: Scenario, readers: usize) -> Option<&ScenarioResult> {
    results
        .iter()
        .find(|r| r.scenario == s.name() && r.readers == readers)
}

fn overhead(
    results: &[ScenarioResult],
    with: Scenario,
    base: Scenario,
    readers: usize,
    pick: fn(&LatencySummary) -> f64,
) -> Option<f64> {
    let (w, b) = (find(results, with, readers)?, find(results, base, readers)?);
    Some(pick(&w.read_latency) / pick(&b.read_latency) - 1.0)
}

fn derive(results: &[ScenarioResult], reader_counts: &[usize]) -> Vec<Derived> {
    let single = find(results, Scenario::ArcSwapReadersOnly, 1).map(|r| r.reads_per_sec);
    reader_counts
        .iter()
        .map(|&n| Derived {
            readers: n,
            arcswap_p50_overhead: overhead(
                results,
                Scenario::ArcSwapWithWriter,
                Scenario::ArcSwapReadersOnly,
                n,
                |l| l.p50_us,
            ),
            arcswap_p99_overhead: overhead(
                results,
                Scenario::ArcSwapWithWriter,
                Scenario::ArcSwapReadersOnly,
                n,
                |l| l.p99_us,
            ),
            rwlock_p50_overhead: overhead(
                results,
                Scenario::RwLockWithWriter,
                Scenario::RwLockReadersOnly,
                n,
                |l| l.p50_us,
            ),
            rwlock_p99_overhead: overhead(
                results,
                Scenario::RwLockWithWriter,
                Scenario::RwLockReadersOnly,
                n,
                |l| l.p99_us,
            ),
            arcswap_scaling_efficiency: single
                .zip(find(results, Scenario::ArcSwapReadersOnly, n))
                .map(|(one, r)| r.reads_per_sec / (one * r.active_cores as f64)),
        })
        .collect()
}

fn write_csv(path: &std::path::Path, results: &[ScenarioResult]) {
    let mut csv = String::from(
        "scenario,readers,reads_per_sec,read_p50_us,read_p99_us,read_p999_us,acquire_p50_us,acquire_p99_us,release_p99_us,writer_edges_per_sec\n",
    );
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.1},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.1}\n",
            r.scenario,
            r.readers,
            r.reads_per_sec,
            r.read_latency.p50_us,
            r.read_latency.p99_us,
            r.read_latency.p999_us,
            r.snapshot_acquire_latency.p50_us,
            r.snapshot_acquire_latency.p99_us,
            r.snapshot_release_latency.p99_us,
            r.writer_edges_per_sec,
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    std::fs::write(path, csv).expect("write csv");
}

fn main() {
    let args = Args::parse();
    let cfg = Config {
        preload: args.get("preload", 1_000_000),
        nodes: args.get("nodes", 100_000),
        skew: args.get("skew", 1.0),
        hops: args.get("hops", 3),
        limit: args.get("limit", 1_024),
        writer_batch: args.get("writer-batch", 1_024),
        writer_rate: args.get("writer-rate", 0),
        duration: Duration::from_secs_f64(args.get("duration-secs", 10.0)),
        warmup: Duration::from_secs_f64(args.get("warmup-secs", 2.0)),
        seed: args.get("seed", 7),
    };
    let reader_counts: Vec<usize> = args.list("readers", "1,8,16,32,64");
    let wanted: Vec<String> = args.list("scenarios", "all");
    let scenarios: Vec<Scenario> = Scenario::ALL
        .into_iter()
        .filter(|s| wanted.iter().any(|w| w == "all" || w == s.name()))
        .collect();
    let out = workspace_path(&args.string("out", "results/phase7/concurrency.json"));
    let environment = Environment::capture();
    let timer = timer_overhead_ns();

    eprintln!(
        "preloading {} hyperedges over {} nodes ({} cores)",
        cfg.preload, cfg.nodes, environment.logical_cores
    );
    let (base, base_seeds) = build(&cfg);
    let base_seeds = Arc::new(base_seeds);
    eprintln!(
        "base graph ready: {} hyperedges, {} seeds, rss {} MiB",
        base.snapshot().live_hyperedge_count(),
        base_seeds.len(),
        rss_bytes() >> 20
    );

    let mut results = Vec::new();
    for &readers in &reader_counts {
        for &scenario in &scenarios {
            let result = if scenario.has_writer() {
                let (engine, seeds) = build(&cfg);
                run(scenario, readers, engine, Arc::new(seeds), &cfg)
            } else {
                run(
                    scenario,
                    readers,
                    Arc::clone(&base),
                    Arc::clone(&base_seeds),
                    &cfg,
                )
            };
            results.push(result);
        }
    }

    let derived = derive(&results, &reader_counts);
    write_csv(&out.with_extension("csv"), &results);
    write_json(
        &out,
        &Report {
            benchmark: "concurrency_scaling",
            environment,
            config: serde_json::json!({
                "preload": cfg.preload,
                "nodes": cfg.nodes,
                "skew": cfg.skew,
                "hops": cfg.hops,
                "limit": cfg.limit,
                "writer_batch": cfg.writer_batch,
                "writer_rate": cfg.writer_rate,
                "duration_secs": cfg.duration.as_secs_f64(),
                "warmup_secs": cfg.warmup.as_secs_f64(),
                "readers": reader_counts,
                "seed": cfg.seed,
            }),
            timer_overhead_ns: timer,
            results,
            derived,
        },
    );
}
