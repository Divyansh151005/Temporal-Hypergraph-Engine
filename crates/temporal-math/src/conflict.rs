//! Deterministic temporal conflict resolution.
//!
//! When a new hyperedge restates a relation that already exists, the older
//! fact is never deleted: its validity window is truncated so the relation's
//! history stays a contiguous, non-overlapping timeline.

/// Cosine similarity above which two same-time facts are considered duplicates.
pub const DUPLICATE_COSINE_THRESHOLD: f32 = 0.98;

/// A borrowed, storage-agnostic view of a hyperedge used for conflict checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeView<'a> {
    /// Opaque handle the caller uses to identify the edge (e.g. a packed id).
    pub handle: u64,
    /// Hash of the semantic event type.
    pub event_type_hash: u64,
    /// Participant identifiers, sorted ascending and deduplicated.
    pub participants: &'a [u64],
    /// Start of validity (Unix seconds).
    pub valid_from: i64,
    /// End of validity (Unix seconds), `None` while ongoing.
    pub valid_to: Option<i64>,
    /// Optional embedding used for duplicate detection.
    pub embedding: Option<&'a [f32]>,
}

/// Which existing edges describe "the same relation" as the new edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictScope<'a> {
    /// Same event type and exactly the same participant set.
    #[default]
    ExactParticipants,
    /// Same event type and containing all of these anchor participants
    /// (sorted). Models functional relations such as `CEO_OF(x, OpenAI)`,
    /// where a new CEO supersedes the previous one.
    Anchors(&'a [u64]),
}

/// Mutation the storage layer must durably apply to settle a conflict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WalMutationCommand {
    /// Temporal truncation: close an existing edge at `valid_to`.
    TruncateExisting {
        /// Handle of the existing edge.
        handle: u64,
        /// New end of validity.
        valid_to: i64,
    },
    /// Historical insert: the new edge ends where the next known fact begins.
    BoundNew {
        /// End of validity for the new edge.
        valid_to: i64,
    },
    /// The new edge restates an existing fact and must not be inserted.
    DropDuplicate {
        /// Handle of the existing equivalent edge.
        duplicate_of: u64,
    },
    /// Same relation at the same instant but a distinct fact: keep it as an
    /// alias sharing the canonical edge's validity window.
    MergeAlias {
        /// Handle of the canonical edge.
        canonical: u64,
        /// Validity end inherited from the canonical edge.
        valid_to: Option<i64>,
    },
}

/// Cosine similarity of two vectors; `0.0` on length mismatch or zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn embeddings_match(a: Option<&[f32]>, b: Option<&[f32]>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => cosine_similarity(a, b) > DUPLICATE_COSINE_THRESHOLD,
        _ => true,
    }
}

fn is_subset(sub: &[u64], sup: &[u64]) -> bool {
    sub.iter().all(|x| sup.binary_search(x).is_ok())
}

fn same_relation(new_edge: &EdgeView, other: &EdgeView, scope: ConflictScope) -> bool {
    other.handle != new_edge.handle
        && other.event_type_hash == new_edge.event_type_hash
        && match scope {
            ConflictScope::ExactParticipants => other.participants == new_edge.participants,
            ConflictScope::Anchors(anchors) => is_subset(anchors, other.participants),
        }
}

/// Resolves conflicts for edges with the exact same participants and event
/// type (the default policy of the specification).
pub fn resolve_conflicts(
    new_edge: &EdgeView,
    existing_edges: &[EdgeView],
) -> Vec<WalMutationCommand> {
    resolve_conflicts_scoped(new_edge, existing_edges, ConflictScope::ExactParticipants)
}

/// Resolves conflicts between `new_edge` and `existing_edges` under `scope`.
///
/// * Equal start time: identical participants with matching embeddings yield
///   [`WalMutationCommand::DropDuplicate`]; otherwise the edge becomes an alias.
/// * The latest earlier fact still valid at `new_edge.valid_from` is truncated
///   there (temporal truncation).
/// * If a later fact exists, the new edge is bounded by its start (historical
///   insert). Together this keeps out-of-order inserts contiguous.
pub fn resolve_conflicts_scoped(
    new_edge: &EdgeView,
    existing_edges: &[EdgeView],
    scope: ConflictScope,
) -> Vec<WalMutationCommand> {
    let mut predecessor: Option<&EdgeView> = None;
    let mut successor: Option<&EdgeView> = None;
    let mut same_instant: Option<&EdgeView> = None;

    for edge in existing_edges
        .iter()
        .filter(|e| same_relation(new_edge, e, scope))
    {
        if edge.valid_from == new_edge.valid_from {
            if edge.participants == new_edge.participants
                && embeddings_match(edge.embedding, new_edge.embedding)
            {
                return vec![WalMutationCommand::DropDuplicate {
                    duplicate_of: edge.handle,
                }];
            }
            if same_instant.map_or(true, |c| edge.handle < c.handle) {
                same_instant = Some(edge);
            }
        } else if edge.valid_from < new_edge.valid_from {
            if predecessor.map_or(true, |p| {
                (edge.valid_from, edge.handle) > (p.valid_from, p.handle)
            }) {
                predecessor = Some(edge);
            }
        } else if successor.map_or(true, |s| {
            (edge.valid_from, edge.handle) < (s.valid_from, s.handle)
        }) {
            successor = Some(edge);
        }
    }

    if let Some(canonical) = same_instant {
        return vec![WalMutationCommand::MergeAlias {
            canonical: canonical.handle,
            valid_to: canonical.valid_to,
        }];
    }

    let mut commands = Vec::with_capacity(2);
    if let Some(pred) = predecessor {
        if pred.valid_to.map_or(true, |end| end > new_edge.valid_from) {
            commands.push(WalMutationCommand::TruncateExisting {
                handle: pred.handle,
                valid_to: new_edge.valid_from,
            });
        }
    }
    if let Some(succ) = successor {
        if new_edge.valid_to.map_or(true, |end| end > succ.valid_from) {
            commands.push(WalMutationCommand::BoundNew {
                valid_to: succ.valid_from,
            });
        }
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const YEAR: i64 = 31_536_000;
    const CEO_OF: u64 = 0xCE0;

    #[derive(Debug, Clone)]
    struct StoredEdge {
        handle: u64,
        participants: Vec<u64>,
        valid_from: i64,
        valid_to: Option<i64>,
        embedding: Option<Vec<f32>>,
    }

    /// Minimal in-memory store that applies resolver commands the same way
    /// the storage engine does.
    #[derive(Default)]
    struct Store {
        edges: Vec<StoredEdge>,
    }

    impl Store {
        fn insert(
            &mut self,
            participants: &[u64],
            valid_from: i64,
            embedding: Option<Vec<f32>>,
            scope: ConflictScope,
        ) -> Vec<WalMutationCommand> {
            let mut participants = participants.to_vec();
            participants.sort_unstable();
            let handle = self.edges.len() as u64 + 1;
            let views: Vec<EdgeView> = self
                .edges
                .iter()
                .map(|e| EdgeView {
                    handle: e.handle,
                    event_type_hash: CEO_OF,
                    participants: &e.participants,
                    valid_from: e.valid_from,
                    valid_to: e.valid_to,
                    embedding: e.embedding.as_deref(),
                })
                .collect();
            let new_view = EdgeView {
                handle,
                event_type_hash: CEO_OF,
                participants: &participants,
                valid_from,
                valid_to: None,
                embedding: embedding.as_deref(),
            };
            let commands = resolve_conflicts_scoped(&new_view, &views, scope);

            let mut valid_to = None;
            for cmd in &commands {
                match *cmd {
                    WalMutationCommand::TruncateExisting { handle, valid_to } => {
                        self.edges[(handle - 1) as usize].valid_to = Some(valid_to);
                    }
                    WalMutationCommand::BoundNew { valid_to: end } => valid_to = Some(end),
                    WalMutationCommand::MergeAlias { valid_to: end, .. } => valid_to = end,
                    WalMutationCommand::DropDuplicate { .. } => return commands,
                }
            }
            self.edges.push(StoredEdge {
                handle,
                participants,
                valid_from,
                valid_to,
                embedding,
            });
            commands
        }

        fn assert_contiguous_timeline(&self) {
            let mut sorted = self.edges.clone();
            sorted.sort_by_key(|e| e.valid_from);
            for pair in sorted.windows(2) {
                assert_eq!(pair[0].valid_to, Some(pair[1].valid_from), "{sorted:?}");
            }
            assert_eq!(sorted.last().unwrap().valid_to, None);
        }
    }

    #[test]
    fn newer_fact_truncates_ongoing_fact() {
        let mut store = Store::default();
        store.insert(&[1, 2], 2023 * YEAR, None, ConflictScope::ExactParticipants);
        let cmds = store.insert(&[1, 2], 2025 * YEAR, None, ConflictScope::ExactParticipants);
        assert_eq!(
            cmds,
            vec![WalMutationCommand::TruncateExisting {
                handle: 1,
                valid_to: 2025 * YEAR
            }]
        );
        store.assert_contiguous_timeline();
    }

    #[test]
    fn backfill_is_bounded_by_next_fact() {
        let mut store = Store::default();
        store.insert(&[1, 2], 2025 * YEAR, None, ConflictScope::ExactParticipants);
        let cmds = store.insert(&[1, 2], 2023 * YEAR, None, ConflictScope::ExactParticipants);
        assert_eq!(
            cmds,
            vec![WalMutationCommand::BoundNew {
                valid_to: 2025 * YEAR
            }]
        );
        store.assert_contiguous_timeline();
    }

    #[test]
    fn out_of_order_2025_2023_2024_is_contiguous() {
        let mut store = Store::default();
        for year in [2025, 2023, 2024] {
            store.insert(&[7, 9], year * YEAR, None, ConflictScope::ExactParticipants);
        }
        store.assert_contiguous_timeline();
        assert!(
            store.edges.iter().all(|e| e.handle > 0),
            "no edge is ever deleted"
        );
        assert_eq!(store.edges.len(), 3);
    }

    #[test]
    fn anchored_scope_supersedes_ceo() {
        // Sam(1) CEO_OF OpenAI(100) in 2023, Mira(2) CEO_OF OpenAI in 2025.
        let anchors = [100u64];
        let mut store = Store::default();
        store.insert(
            &[1, 100],
            2023 * YEAR,
            None,
            ConflictScope::Anchors(&anchors),
        );
        let cmds = store.insert(
            &[2, 100],
            2025 * YEAR,
            None,
            ConflictScope::Anchors(&anchors),
        );
        assert_eq!(
            cmds,
            vec![WalMutationCommand::TruncateExisting {
                handle: 1,
                valid_to: 2025 * YEAR
            }]
        );
        store.assert_contiguous_timeline();
    }

    #[test]
    fn exact_scope_ignores_different_participants() {
        let mut store = Store::default();
        store.insert(
            &[1, 100],
            2023 * YEAR,
            None,
            ConflictScope::ExactParticipants,
        );
        let cmds = store.insert(
            &[2, 100],
            2025 * YEAR,
            None,
            ConflictScope::ExactParticipants,
        );
        assert!(cmds.is_empty());
    }

    #[test]
    fn same_instant_duplicate_is_dropped() {
        let mut store = Store::default();
        store.insert(
            &[1, 2],
            10,
            Some(vec![1.0, 0.0, 0.0]),
            ConflictScope::ExactParticipants,
        );
        let cmds = store.insert(
            &[2, 1],
            10,
            Some(vec![0.99, 0.01, 0.0]),
            ConflictScope::ExactParticipants,
        );
        assert_eq!(
            cmds,
            vec![WalMutationCommand::DropDuplicate { duplicate_of: 1 }]
        );
        assert_eq!(store.edges.len(), 1);
    }

    #[test]
    fn same_instant_distinct_embedding_becomes_alias() {
        let mut store = Store::default();
        store.insert(
            &[1, 2],
            10,
            Some(vec![1.0, 0.0]),
            ConflictScope::ExactParticipants,
        );
        let cmds = store.insert(
            &[1, 2],
            10,
            Some(vec![0.0, 1.0]),
            ConflictScope::ExactParticipants,
        );
        assert_eq!(
            cmds,
            vec![WalMutationCommand::MergeAlias {
                canonical: 1,
                valid_to: None
            }]
        );
        assert_eq!(store.edges.len(), 2);
    }

    #[test]
    fn cosine_similarity_edge_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn any_insertion_order_yields_contiguous_timeline(
            years in proptest::collection::hash_set(1990i64..2060, 1..12)
                .prop_map(|s| s.into_iter().collect::<Vec<_>>())
                .prop_shuffle()
        ) {
            let mut store = Store::default();
            for year in &years {
                store.insert(&[3, 4, 5], year * YEAR, None, ConflictScope::ExactParticipants);
            }
            prop_assert_eq!(store.edges.len(), years.len());
            store.assert_contiguous_timeline();
        }
    }
}
