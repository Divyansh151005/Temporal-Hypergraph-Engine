//! Shared harness code for the Phase 7 benchmarks: a counting allocator,
//! a seeded temporal workload, CLI parsing, latency summaries and the
//! environment header written into every result file.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use engine_core::NewHyperedge;
use hdrhistogram::Histogram;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::Serialize;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// `System` allocator that tracks live and peak heap bytes. Install it with
/// `#[global_allocator] static ALLOC: CountingAlloc = CountingAlloc;`.
pub struct CountingAlloc;

impl CountingAlloc {
    /// Heap bytes currently allocated.
    pub fn live() -> usize {
        LIVE_BYTES.load(Ordering::Relaxed)
    }

    /// Highest `live()` seen since the last [`CountingAlloc::reset_peak`].
    pub fn peak() -> usize {
        PEAK_BYTES.load(Ordering::Relaxed)
    }

    /// Resets the peak to the current live size.
    pub fn reset_peak() {
        PEAK_BYTES.store(Self::live(), Ordering::Relaxed);
    }

    fn grew(bytes: usize) {
        let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` with the caller's layout unchanged;
// the counters are plain atomics and never affect allocation behaviour.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            Self::grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            Self::grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            if new_size >= layout.size() {
                Self::grew(new_size - layout.size());
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new
    }
}

/// Resident set size of this process, in bytes.
pub fn rss_bytes() -> usize {
    memory_stats::memory_stats().map_or(0, |m| m.physical_mem)
}

/// `--key value` / `--flag` command-line options. Cargo's injected
/// `--bench` flag is ignored.
pub struct Args(BTreeMap<String, String>);

impl Args {
    /// Parses `std::env::args()`.
    pub fn parse() -> Self {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut map = BTreeMap::new();
        let mut i = 0;
        while i < raw.len() {
            if let Some(key) = raw[i].strip_prefix("--") {
                let value = raw.get(i + 1).filter(|v| !v.starts_with("--"));
                if let Some(v) = value {
                    map.insert(key.to_string(), v.clone());
                    i += 1;
                } else {
                    map.insert(key.to_string(), String::new());
                }
            }
            i += 1;
        }
        map.remove("bench");
        Self(map)
    }

    /// Value of `--key`, or `default` when absent.
    pub fn get<T: FromStr>(&self, key: &str, default: T) -> T {
        match self.0.get(key) {
            Some(v) => v
                .replace('_', "")
                .parse()
                .unwrap_or_else(|_| panic!("invalid value for --{key}: {v}")),
            None => default,
        }
    }

    /// String value of `--key`, or `default`.
    pub fn string(&self, key: &str, default: &str) -> String {
        self.0
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }

    /// Comma-separated list value of `--key`, or `default`.
    pub fn list<T: FromStr>(&self, key: &str, default: &str) -> Vec<T> {
        self.string(key, default)
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let s = s.trim();
                s.parse()
                    .or_else(|_| s.replace('_', "").parse())
                    .unwrap_or_else(|_| panic!("invalid list item for --{key}: {s}"))
            })
            .collect()
    }
}

/// Seeded synthetic temporal hyperedge stream.
///
/// Participants are drawn from `nodes` labels with a power-law skew
/// (`index = nodes * u^skew`, so low indices become hubs); `valid_from`
/// advances monotonically with jitter and ~10% of edges are closed.
pub struct Workload {
    rng: SmallRng,
    nodes: usize,
    skew: f64,
    event_types: usize,
    clock: i64,
}

impl Workload {
    /// Creates a workload over `nodes` entity labels.
    pub fn new(seed: u64, nodes: usize, skew: f64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            nodes: nodes.max(2),
            skew,
            event_types: 32,
            clock: 1_600_000_000,
        }
    }

    /// Index of a participant node.
    pub fn node(&mut self) -> usize {
        let u: f64 = self.rng.gen();
        ((self.nodes as f64 * u.powf(self.skew)) as usize).min(self.nodes - 1)
    }

    /// Label of node `index`.
    pub fn label(index: usize) -> String {
        format!("n{index}")
    }

    /// Next hyperedge.
    pub fn edge(&mut self) -> NewHyperedge {
        let arity = self.rng.gen_range(2..=4);
        let participants: Vec<String> = (0..arity).map(|_| Self::label(self.node())).collect();
        self.clock += self.rng.gen_range(0..4);
        let event = format!("EVENT_{}", self.rng.gen_range(0..self.event_types));
        let edge = NewHyperedge::new(event, participants, self.clock);
        if self.rng.gen_bool(0.1) {
            let end = self.clock + self.rng.gen_range(1..86_400);
            edge.valid_to(end)
        } else {
            edge
        }
    }

    /// Next `n` hyperedges.
    pub fn batch(&mut self, n: usize) -> Vec<NewHyperedge> {
        (0..n).map(|_| self.edge()).collect()
    }

    /// Current position of the workload clock.
    pub fn clock(&self) -> i64 {
        self.clock
    }
}

/// Nanosecond-resolution latency histogram (1 ns .. 1 hour, 3 sig. digits).
pub fn histogram() -> Histogram<u64> {
    Histogram::new_with_bounds(1, 3_600_000_000_000, 3).expect("valid histogram bounds")
}

/// Latency percentiles in microseconds.
#[derive(Debug, Clone, Serialize)]
pub struct LatencySummary {
    /// Recorded samples.
    pub count: u64,
    /// Mean, µs.
    pub mean_us: f64,
    /// Median, µs.
    pub p50_us: f64,
    /// 99th percentile, µs.
    pub p99_us: f64,
    /// 99.9th percentile, µs.
    pub p999_us: f64,
    /// Maximum, µs.
    pub max_us: f64,
}

impl LatencySummary {
    /// Summarizes a nanosecond histogram.
    pub fn from_ns(h: &Histogram<u64>) -> Self {
        let us = |ns: u64| ns as f64 / 1_000.0;
        Self {
            count: h.len(),
            mean_us: h.mean() / 1_000.0,
            p50_us: us(h.value_at_quantile(0.50)),
            p99_us: us(h.value_at_quantile(0.99)),
            p999_us: us(h.value_at_quantile(0.999)),
            max_us: us(h.max()),
        }
    }
}

/// Machine and build provenance stored with every result.
#[derive(Debug, Clone, Serialize)]
pub struct Environment {
    /// CPU model.
    pub cpu: String,
    /// Logical cores available to the process.
    pub logical_cores: usize,
    /// Physical memory, bytes.
    pub memory_bytes: u64,
    /// Operating system.
    pub os: String,
    /// `rustc --version`.
    pub rustc: String,
    /// Short git SHA, with `-dirty` when the tree has local changes.
    pub git_sha: String,
    /// Build profile of the benchmark binary.
    pub profile: &'static str,
    /// Unix seconds at which the run started.
    pub started_at: u64,
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn total_memory() -> u64 {
    if let Some(v) = command_output("sysctl", &["-n", "hw.memsize"]) {
        return v.parse().unwrap_or(0);
    }
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

fn cpu_model() -> String {
    if let Some(v) = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]) {
        return v;
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

impl Environment {
    /// Captures the current machine and build.
    pub fn capture() -> Self {
        let sha = command_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_default();
        let dirty =
            command_output("git", &["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
        Self {
            cpu: cpu_model(),
            logical_cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
            memory_bytes: total_memory(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            rustc: command_output("rustc", &["--version"]).unwrap_or_default(),
            git_sha: if dirty { format!("{sha}-dirty") } else { sha },
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }
}

/// Resolves a relative output path against the workspace root, because
/// `cargo bench` runs bench binaries from the package directory.
pub fn workspace_path(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        return path;
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("benchmarks crate lives inside the workspace")
        .join(path)
}

/// Writes `value` as pretty JSON to `path`, creating parent directories.
pub fn write_json(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let json = serde_json::to_string_pretty(value).expect("serialize results");
    std::fs::write(path, json + "\n").expect("write results");
    eprintln!("wrote {}", path.display());
}

/// Least-squares slope of `y` over `x`.
pub fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if points.len() < 2 {
        return 0.0;
    }
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), (x, y)| {
        (num + (x - mx) * (y - my), den + (x - mx) * (x - mx))
    });
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}
