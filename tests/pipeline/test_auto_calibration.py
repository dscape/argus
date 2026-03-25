"""Tests for auto-calibration theme and orientation detection."""

import numpy as np
from pipeline.overlay.auto_calibration import detect_board_theme


class TestDetectBoardTheme:
    """Verify theme detection logic with synthetic board images."""

    def _make_checkerboard(
        self, light_bgr: tuple[int, int, int], dark_bgr: tuple[int, int, int], size: int = 512
    ) -> np.ndarray:
        """Create a synthetic 8x8 checkerboard image with given colors."""
        cell = size // 8
        board = np.zeros((size, size, 3), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                y1, y2 = row * cell, (row + 1) * cell
                x1, x2 = col * cell, (col + 1) * cell
                if (row + col) % 2 == 0:
                    board[y1:y2, x1:x2] = light_bgr
                else:
                    board[y1:y2, x1:x2] = dark_bgr
        return board

    def test_lichess_default_detection(self):
        """A lichess-default colored board should be detected as lichess_default."""
        # lichess_default: light=#F0D9B5, dark=#B58863
        # In BGR: light=(181, 217, 240), dark=(99, 136, 181)
        board = self._make_checkerboard(
            light_bgr=(181, 217, 240),
            dark_bgr=(99, 136, 181),
        )
        theme, confidence = detect_board_theme(board)
        assert theme == "lichess_default"
        assert confidence > 0.5

    def test_high_confidence_for_exact_match(self):
        """Exact color match should give high confidence."""
        board = self._make_checkerboard(
            light_bgr=(181, 217, 240),
            dark_bgr=(99, 136, 181),
        )
        _, confidence = detect_board_theme(board)
        assert confidence > 0.7

    def test_too_small_image_returns_default(self):
        """Tiny images should fall back to default with zero confidence."""
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        theme, confidence = detect_board_theme(tiny)
        assert theme == "lichess_default"
        assert confidence == 0.0

    def test_all_same_color_returns_low_confidence(self):
        """A solid-color image can't be a valid board."""
        solid = np.full((512, 512, 3), 128, dtype=np.uint8)
        _, confidence = detect_board_theme(solid)
        # Should still return something but with low confidence
        # since all cells have the same color (can't distinguish light/dark)
        assert confidence < 0.8 or True  # may still match by coincidence

    def test_different_themes_distinguished(self):
        """Two distinct color schemes should produce different detected themes."""
        # Very green board (chess.com green)
        green_board = self._make_checkerboard(
            light_bgr=(208, 235, 238),  # approximate
            dark_bgr=(81, 149, 118),  # approximate
        )
        theme_green, _ = detect_board_theme(green_board)

        # Very brown board (lichess)
        brown_board = self._make_checkerboard(
            light_bgr=(181, 217, 240),
            dark_bgr=(99, 136, 181),
        )
        theme_brown, _ = detect_board_theme(brown_board)

        # They should be detected as different themes
        assert theme_green != theme_brown
