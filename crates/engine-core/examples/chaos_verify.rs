//! Offline integrity report for an engine directory, used by
//! `tests/chaos_sigkill_harness.py` after it has reaped the server.
//!
//! ```text
//! chaos_verify <data_dir> [--valid-from-out PATH] [--chaos-summaries]
//! ```
//!
//! Opens the directory (running crash recovery, which truncates torn WAL
//! tails), prints one JSON object on stdout and optionally writes the sorted
//! `valid_from` of every live hyperedge to `PATH`, one per line. With
//! `--chaos-summaries`, every edge's summary must be `chaos-{valid_from}`.
//! The directory must not be open in any other process.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use engine_core::{ConcurrentHypergraphEngine, EngineConfig};

const USAGE: &str = "usage: chaos_verify <data_dir> [--valid-from-out PATH] [--chaos-summaries]";

struct Args {
    data_dir: PathBuf,
    valid_from_out: Option<PathBuf>,
    chaos_summaries: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut iter = std::env::args().skip(1);
    let data_dir = iter.next().ok_or(USAGE)?.into();
    let mut args = Args {
        data_dir,
        valid_from_out: None,
        chaos_summaries: false,
    };
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--valid-from-out" => {
                args.valid_from_out = Some(iter.next().ok_or(USAGE)?.into());
            }
            "--chaos-summaries" => args.chaos_summaries = true,
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let engine =
        match ConcurrentHypergraphEngine::open_with_config(&args.data_dir, EngineConfig::default())
        {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("failed to open {}: {err}", args.data_dir.display());
                return ExitCode::FAILURE;
            }
        };
    let snapshot = engine.snapshot();

    let mut dangling_participants = 0u64;
    let mut unlinked_participants = 0u64;
    let mut summary_mismatches = 0u64;
    let mut valid_from = Vec::with_capacity(snapshot.live_hyperedge_count() as usize);
    for edge in snapshot.iter_hyperedges() {
        valid_from.push(edge.valid_from);
        for participant in &edge.participants {
            match snapshot.node(*participant) {
                None => dangling_participants += 1,
                Some(node) if !node.hyperedges.contains(&edge.id) => unlinked_participants += 1,
                Some(_) => {}
            }
        }
        if args.chaos_summaries {
            let expected = format!("chaos-{}", edge.valid_from);
            if snapshot.payload(edge).map(|s| &**s) != Some(expected.as_str()) {
                summary_mismatches += 1;
            }
        }
    }
    let orphan_nodes = snapshot
        .iter_nodes()
        .filter(|node| {
            !node
                .hyperedges
                .iter()
                .any(|id| snapshot.hyperedge(*id).is_some())
        })
        .count();
    valid_from.sort_unstable();

    if let Some(path) = &args.valid_from_out {
        let mut text = String::with_capacity(valid_from.len() * 8);
        for v in &valid_from {
            text.push_str(&v.to_string());
            text.push('\n');
        }
        if let Err(err) = fs::write(path, text) {
            eprintln!("failed to write {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    }

    println!(
        "{{\"last_sequence\":{},\"live_nodes\":{},\"live_edges\":{},\"dangling_participants\":{},\"unlinked_participants\":{},\"orphan_nodes\":{},\"summary_mismatches\":{}}}",
        engine.last_sequence(),
        snapshot.live_node_count(),
        snapshot.live_hyperedge_count(),
        dangling_participants,
        unlinked_participants,
        orphan_nodes,
        summary_mismatches,
    );
    ExitCode::SUCCESS
}
