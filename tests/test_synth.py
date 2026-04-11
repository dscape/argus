from __future__ import annotations

from typing import Any

from PIL import Image

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.datagen.synth import generate_clip, generate_dataset


class _FakeBlenderServer:
    def render_clip(self, manifest_dict: dict[str, Any], image_size: int) -> list[Image.Image]:
        return [
            Image.new("RGB", (image_size, image_size), color=(127, 127, 127))
            for _ in manifest_dict["frames"]
        ]


def test_generate_clip_defaults_to_legal_move_targets() -> None:
    clip = generate_clip(
        moves=["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
        clip_length=16,
        frames_per_move=3,
        image_size=32,
        augment=False,
        seed=123,
        server=_FakeBlenderServer(),
    )

    move_targets = clip["move_targets"]
    legal_masks = clip["legal_masks"]
    detect_targets = clip["detect_targets"]

    for frame_idx in range(len(move_targets)):
        if detect_targets[frame_idx] <= 0.5:
            continue
        target = int(move_targets[frame_idx].item())
        assert target != NO_MOVE_IDX
        assert legal_masks[frame_idx, target].item() is True


def test_generate_dataset_can_source_games_from_pgn_file(tmp_path) -> None:
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(
        """[Event \"SyntheticTest\"]
[Site \"Local\"]
[Date \"2026.04.11\"]
[Round \"1\"]
[White \"White\"]
[Black \"Black\"]
[Result \"*\"]
[WhiteElo \"2400\"]
[BlackElo \"2400\"]

1. a4 h5 2. Ra3 Rh6 3. Rah3 d5 4. d4 Bxh3 5. Nxh3 *
"""
    )

    clip = generate_dataset(
        num_clips=1,
        clip_length=12,
        image_size=32,
        frames_per_move=2,
        augment=False,
        illegal_clip_prob=0.0,
        game_source="pgn_file",
        pgn_path=pgn_path,
        min_moves=6,
        max_moves=12,
        seed=123,
        server=_FakeBlenderServer(),
    )[0]

    vocab = get_vocabulary()
    clip_moves = [
        vocab.index_to_uci(int(target.item()))
        for target, detect in zip(clip["move_targets"], clip["detect_targets"])
        if detect.item() > 0.5
    ]
    game_moves = ["a2a4", "h7h5", "a1a3", "h8h6", "a3h3", "d7d5", "d2d4", "c8h3", "g1h3"]

    assert clip_moves
    assert any(
        clip_moves == game_moves[start : start + len(clip_moves)]
        for start in range(len(game_moves) - len(clip_moves) + 1)
    )


def test_generate_dataset_requires_pgn_path_for_pgn_file_source() -> None:
    try:
        generate_dataset(
            num_clips=1,
            game_source="pgn_file",
            augment=False,
            server=_FakeBlenderServer(),
        )
    except ValueError as exc:
        assert "pgn_path" in str(exc)
    else:
        raise AssertionError("Expected generate_dataset() to require pgn_path")
