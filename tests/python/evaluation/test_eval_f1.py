"""F1 harness: identity, perturbations, IoU edge cases, alias alignment, Hungarian optimality."""

import copy
import itertools
import json

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from hypergraph_evaluation.evaluate_f1 import (
    DAY,
    Config,
    Horizon,
    TfidfEmbedder,
    align_nodes,
    evaluate,
    jaccard,
    main,
    prf,
    regressions,
    score_document,
    temporal_score,
)

Y2020, Y2021, Y2022, Y2023 = 1577836800, 1609459200, 1640995200, 1672531200

TRUTH = {
    "nodes": [
        {"id": "org:microsoft", "label": "Microsoft", "aliases": ["MSFT"]},
        {"id": "org:activision", "label": "Activision Blizzard", "aliases": ["Activision"]},
        {"id": "per:bobby_kotick", "label": "Bobby Kotick"},
        {"id": "loc:santa_monica", "label": "Santa Monica"},
    ],
    "hyperedges": [
        {"id": "t1", "event_type": "ACQUISITION", "participants": ["org:microsoft", "org:activision"],
         "valid_from": Y2022, "valid_to": None},
        {"id": "t2", "event_type": "EXECUTIVE_ROLE", "participants": ["per:bobby_kotick", "org:activision"],
         "valid_from": Y2020, "valid_to": Y2023},
        {"id": "t3", "event_type": "HEADQUARTERS", "participants": ["org:activision", "loc:santa_monica"],
         "valid_from": None, "valid_to": None},
    ],
}


def score(pred, truth=TRUTH, **config):
    return score_document("d", pred, truth, TfidfEmbedder(), Config(**config)).metrics()


def test_identity_scores_one():
    metrics = score(copy.deepcopy(TRUTH))
    assert all(v == pytest.approx(1.0) for v in metrics.values()), metrics


def test_prediction_ids_and_aliases_do_not_matter():
    pred = copy.deepcopy(TRUTH)
    renames = {"org:microsoft": "org:msft_corp", "org:activision": "org:activision_blizzard_inc"}
    for node in pred["nodes"]:
        node["id"] = renames.get(node["id"], node["id"])
    pred["nodes"][1]["label"] = "Activision"  # alias of the truth node
    pred["nodes"][1]["aliases"] = []
    for edge in pred["hyperedges"]:
        edge["participants"] = [renames.get(p, p) for p in edge["participants"]]
    metrics = score(pred)
    assert metrics["node_f1"] == 1.0 and metrics["edge_f1"] == 1.0


def test_single_field_perturbations():
    dropped_node = copy.deepcopy(TRUTH)
    del dropped_node["nodes"][3]
    metrics = score(dropped_node)
    assert metrics["node_precision"] == 1.0 and metrics["node_recall"] == pytest.approx(0.75)

    wrong_participant = copy.deepcopy(TRUTH)
    wrong_participant["hyperedges"][0]["participants"] = ["org:microsoft", "loc:santa_monica"]
    metrics = score(wrong_participant)
    # jaccard 1/3 -> 0.3 + 0.5/3 + 0.2 = 0.67 < 0.80: one false positive and one false negative.
    assert metrics["edge_precision"] == pytest.approx(2 / 3) and metrics["edge_recall"] == pytest.approx(2 / 3)
    assert metrics["participant_f1"] == pytest.approx(5 / 6)

    shifted = copy.deepcopy(TRUTH)
    shifted["hyperedges"][1]["valid_to"] = Y2021  # [2020, 2021) vs [2020, 2023)
    metrics = score(shifted)
    assert metrics["edge_f1"] == 1.0  # 0.3 + 0.5 + 0.2 * 0.33 = 0.87
    assert metrics["temporal_accuracy"] == pytest.approx((1 + 1 + (Y2021 - Y2020) / (Y2023 - Y2020)) / 3)

    renamed_type = copy.deepcopy(TRUTH)
    renamed_type["hyperedges"][0]["event_type"] = "MERGER"
    metrics = score(renamed_type)
    # A different event type is a different fact, even with identical participants and validity.
    assert metrics["edge_recall"] == pytest.approx(2 / 3) and metrics["node_f1"] == 1.0


def test_empty_graphs():
    empty = {"nodes": [], "hyperedges": []}
    assert score(empty, empty)["edge_f1"] == 1.0
    metrics = score(empty)
    assert metrics["node_recall"] == 0.0 and metrics["edge_recall"] == 0.0 and metrics["temporal_accuracy"] == 0.0
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0) and prf(0, 3, 0) == (0.0, 0.0, 0.0)


def test_temporal_iou_edge_cases():
    horizon = Horizon(Y2020, Y2023)
    assert temporal_score((None, None), (None, None), horizon) == 1.0
    assert temporal_score((Y2020, Y2021), (Y2020, Y2021), horizon) == 1.0
    assert temporal_score((Y2020, Y2021), (Y2021, Y2022), horizon) == 0.0  # half-open: touching is disjoint
    assert temporal_score((Y2020, Y2022), (Y2021, Y2023), horizon) == pytest.approx(1 / 3, rel=1e-2)
    # Unknown bounds clamp to the horizon.
    assert temporal_score((None, Y2021), (Y2020, Y2021), horizon) == 1.0
    assert temporal_score((Y2022, None), (Y2022, Y2023), horizon) == 1.0
    assert temporal_score((Y2020, None), (None, None), horizon) == 1.0
    assert temporal_score((Y2021, None), (None, None), horizon) == pytest.approx(2 / 3, rel=1e-2)
    # Points compare by start within the tolerance.
    assert temporal_score((Y2021, Y2021), (Y2021, Y2021), horizon) == 1.0
    assert temporal_score((Y2021, Y2021), (Y2021 + 10 * DAY, None), horizon) == 1.0
    assert temporal_score((Y2021, Y2021), (Y2022, None), horizon) == 0.0
    assert temporal_score((Y2021, Y2021), (None, None), horizon) == 0.0
    # ISO strings are normalized.
    assert Horizon.of({"hyperedges": [{"valid_from": "2020-01-01T00:00:00Z", "valid_to": "2023-01-01"}]}) == Horizon(
        Y2020, Y2023 + DAY
    )


def test_alias_alignment_and_threshold():
    truth = [{"id": "org:aws", "label": "Amazon Web Services", "aliases": ["AWS"]}]
    pred = [{"id": "org:aws", "label": "AWS"}]
    assert align_nodes(pred, truth, TfidfEmbedder(), 0.90) == {"org:aws": ("org:aws", pytest.approx(1.0))}
    unrelated = [{"id": "org:x", "label": "Anthropic"}]
    assert align_nodes(unrelated, truth, TfidfEmbedder(), 0.90) == {}
    near = [{"id": "org:y", "label": "Amazon Web Service"}]
    assert align_nodes(near, truth, TfidfEmbedder(), 0.50) != {}


def greedy(matrix: np.ndarray) -> float:
    total, used_rows, used_cols = 0.0, set(), set()
    for value, i, j in sorted(((matrix[i, j], i, j) for i, j in np.ndindex(matrix.shape)), reverse=True):
        if i not in used_rows and j not in used_cols:
            total += value
            used_rows.add(i)
            used_cols.add(j)
    return total


def test_hungarian_beats_greedy():
    matrix = np.array([[0.9, 0.8], [0.85, 0.1]])
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    assert matrix[rows, cols].sum() == pytest.approx(1.65)
    assert greedy(matrix) == pytest.approx(1.0)

    rng = np.random.default_rng(3)
    for _ in range(50):
        m = rng.random((4, 4))
        rows, cols = linear_sum_assignment(m, maximize=True)
        best = max(sum(m[i, p[i]] for i in range(4)) for p in itertools.permutations(range(4)))
        assert m[rows, cols].sum() == pytest.approx(best)
        assert best >= greedy(m) - 1e-12


def test_edge_matching_is_optimal_not_greedy():
    # Pred edge A overlaps both truth edges; greedy would take its best match and strand edge B.
    truth = {
        "nodes": [{"id": f"e:{c}", "label": f"Entity {c.upper()}"} for c in "abcd"],
        "hyperedges": [
            {"event_type": "LINK", "participants": ["e:a", "e:b", "e:c"], "valid_from": None, "valid_to": None},
            {"event_type": "LINK", "participants": ["e:a", "e:b"], "valid_from": None, "valid_to": None},
        ],
    }
    pred = copy.deepcopy(truth)
    pred["hyperedges"] = [
        {"event_type": "LINK", "participants": ["e:a", "e:b"], "valid_from": None, "valid_to": None},
        {"event_type": "LINK", "participants": ["e:a", "e:b", "e:c"], "valid_from": None, "valid_to": None},
    ]
    assert score(pred, truth)["edge_f1"] == 1.0


def test_jaccard():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0


def test_dataset_cli_regression_and_recall_gates(tmp_path, capsys):
    truth = [{"dataset_id": "d1", "raw_text": "", "ground_truth_graph": TRUTH}]
    weak = copy.deepcopy(TRUTH)
    weak["hyperedges"] = weak["hyperedges"][:1]
    preds = [{"dataset_id": "d1", "graph": weak}]
    docs, summary = evaluate(preds, truth)
    assert summary["micro"]["edge_recall"] == pytest.approx(1 / 3)
    assert summary["counts"]["edge_truth"] == 3

    truth_path, pred_path = tmp_path / "truth.json", tmp_path / "pred.jsonl"
    truth_path.write_text(json.dumps(truth))
    pred_path.write_text(json.dumps({"dataset_id": "d1", "graph": TRUTH}) + "\n")
    baseline, csv_path = tmp_path / "base.json", tmp_path / "f1.csv"
    assert main(["--pred", str(pred_path), "--truth", str(truth_path), "--out-json", str(baseline),
                 "--out-csv", str(csv_path)]) == 0
    assert "__micro__" in csv_path.read_text()

    pred_path.write_text(json.dumps({"dataset_id": "d1", "graph": weak}) + "\n")
    assert main(["--pred", str(pred_path), "--truth", str(truth_path), "--baseline", str(baseline)]) == 1
    assert "edge_f1 dropped" in capsys.readouterr().err
    assert main(["--pred", str(pred_path), "--truth", str(truth_path), "--min-recall", "0.8"]) == 1
    assert regressions({"micro": {"node_f1": 0.99, "edge_f1": 0.99, "participant_f1": 1, "temporal_accuracy": 1}},
                       {"micro": {"node_f1": 1.0, "edge_f1": 1.0}}, 0.02) == []
