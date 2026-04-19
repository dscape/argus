import json

from autoresearch import prepare


def _report(
    *,
    board_exact: float,
    false_change_rate: float,
    move_recall: float,
    macro_f1: float,
    non_empty_accuracy: float,
) -> dict[str, object]:
    return {
        "metrics": {
            "board_exact_match": board_exact,
            "macro_f1": macro_f1,
            "non_empty_accuracy": non_empty_accuracy,
        },
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": false_change_rate,
    }


def test_record_successful_run_does_not_overwrite_best_result_for_worse_same_config(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    runs_dir = tmp_path / "runs"
    snapshots_dir = tmp_path / "snapshots"
    results_path = tmp_path / "results.tsv"
    train_path = tmp_path / "train.py"
    best_train_path = tmp_path / "best_train.py"
    best_result_path = tmp_path / "best_result.json"

    monkeypatch.setattr(prepare, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(prepare, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(prepare, "SNAPSHOTS_DIR", snapshots_dir)
    monkeypatch.setattr(prepare, "RESULTS_PATH", results_path)
    monkeypatch.setattr(prepare, "TRAIN_PATH", train_path)
    monkeypatch.setattr(prepare, "BEST_TRAIN_PATH", best_train_path)
    monkeypatch.setattr(prepare, "BEST_RESULT_PATH", best_result_path)
    monkeypatch.setattr(prepare, "authrelative", lambda path: str(path))

    train_path.write_text("current train\n")

    best_snapshot = snapshots_dir / "best.py"
    best_snapshot.parent.mkdir(parents=True, exist_ok=True)
    best_snapshot.write_text("best config\n")
    best_report_path = runs_dir / "best.json"
    best_report_path.parent.mkdir(parents=True, exist_ok=True)
    best_report_path.write_text("{}")
    best_report = _report(
        board_exact=0.2000004,
        false_change_rate=0.01,
        move_recall=0.3,
        macro_f1=0.4,
        non_empty_accuracy=0.5,
    )

    first = prepare.record_successful_run(
        train_path=train_path,
        snapshot_path=best_snapshot,
        report_path=best_report_path,
        report=best_report,
        description="best",
    )

    assert first.status == "keep"
    assert json.loads(best_result_path.read_text())["metrics"]["board_exact_match"] == 0.2000004

    train_path.write_text(best_train_path.read_text())
    worse_snapshot = snapshots_dir / "worse_same_config.py"
    worse_snapshot.write_text(best_train_path.read_text())
    worse_report_path = runs_dir / "worse.json"
    worse_report_path.write_text("{}")
    worse_report = _report(
        board_exact=0.2000004,
        false_change_rate=0.02,
        move_recall=0.2,
        macro_f1=0.6,
        non_empty_accuracy=0.5,
    )

    second = prepare.record_successful_run(
        train_path=train_path,
        snapshot_path=worse_snapshot,
        report_path=worse_report_path,
        report=worse_report,
        description="worse same config",
    )

    assert second.status == "discard"
    assert second.restored_train is True
    best_result = json.loads(best_result_path.read_text())
    assert best_result["metrics"]["board_exact_match"] == 0.2000004
    assert best_result["static_frame_false_change_rate"] == 0.01

    lines = results_path.read_text().splitlines()
    assert lines[-1].split("\t")[6] == "discard"
