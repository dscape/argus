from __future__ import annotations

import numpy as np
import torch
from pipeline.physical.board_probe import PhysicalBoardStateEnsembleProbe, PhysicalBoardStateProbe
from pipeline.physical.square_classifier import (
    _build_probe_from_checkpoint,
    _class_ids_to_board_fen,
    _encoder_kwargs_from_checkpoint,
)


def test_class_ids_to_board_fen_encodes_empty_runs() -> None:
    class_ids = [0] * 64
    class_ids[0] = 10  # black rook on a8
    class_ids[63] = 4  # white rook on h1

    fen = _class_ids_to_board_fen(class_ids)

    assert fen == "r7/8/8/8/8/8/8/7R"


def test_encoder_kwargs_from_checkpoint_reads_yolo_metadata() -> None:
    kwargs = _encoder_kwargs_from_checkpoint(
        {
            "model_name": "weights/yolo_base/yolo11n.pt",
            "metadata": {
                "encoder_type": "yolo",
                "feature_layer_indices": [16, 19, 22],
                "output_grid_size": 16,
            },
        }
    )

    assert kwargs == {
        "model_name": "weights/yolo_base/yolo11n.pt",
        "frozen": True,
        "encoder_type": "yolo",
        "feature_layer_indices": [16, 19, 22],
        "output_grid_size": 16,
    }


def test_build_probe_from_checkpoint_returns_board_probe_for_board_architecture() -> None:
    probe = _build_probe_from_checkpoint(
        {
            "architecture": "board_probe",
            "probe_config": {"head_type": "transformer", "transformer_layers": 1},
        },
        embed_dim=8,
    )

    assert isinstance(probe, PhysicalBoardStateProbe)
    assert probe.head_type == "transformer"


def test_build_probe_from_checkpoint_returns_ensemble_for_logit_ensemble_architecture() -> None:
    member = PhysicalBoardStateProbe(8)
    probe = _build_probe_from_checkpoint(
        {
            "architecture": "board_probe_ensemble",
            "ensemble_weights": [0.25, 0.75],
            "members": [
                {"state_dict": member.state_dict(), "probe_config": {"head_type": "linear"}},
                {"state_dict": member.state_dict(), "probe_config": {"head_type": "linear"}},
            ],
        },
        embed_dim=8,
    )

    assert isinstance(probe, PhysicalBoardStateEnsembleProbe)


def test_read_board_observation_from_frame_applies_constraints(monkeypatch) -> None:
    import pipeline.physical.square_classifier as square_classifier

    class DummyModule:
        pass

    logits = np.zeros((64, 13), dtype=np.float32)
    logits[:, 0] = 1.0
    logits[0, 1] = 5.0
    logits[0, 3] = 4.0
    logits[4, 12] = 6.0
    logits[60, 4] = 3.0
    logits[60, 6] = 2.5

    monkeypatch.setattr(
        square_classifier,
        "_get_runtime_model",
        lambda device: ({}, DummyModule(), DummyModule()),
    )
    monkeypatch.setattr(
        square_classifier,
        "_predict_board_logits",
        lambda **kwargs: torch.tensor(logits, dtype=torch.float32),
    )

    observation = square_classifier.read_board_observation_from_frame(
        np.zeros((64, 64, 3), dtype=np.uint8),
    )

    assert observation is not None
    assert observation.fen.split("/", 1)[0] == "B3k3"
    assert observation.fen.split("/")[-1] == "4K3"


def test_read_fen_from_frame_returns_none_without_weights(monkeypatch) -> None:
    import pipeline.physical.square_classifier as square_classifier

    monkeypatch.setattr(
        square_classifier,
        "_resolve_weights_path",
        lambda: (_ for _ in ()).throw(FileNotFoundError()),
    )

    result = square_classifier.read_fen_from_frame(np.zeros((64, 64, 3), dtype=np.uint8))

    assert result is None
