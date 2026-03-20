"""Extract board state (FEN) from a cropped 2D chess board overlay image.

Rendered 2D boards (lichess, chess.com) have solid-color squares and crisp
piece sprites, making piece recognition straightforward via template matching.

Strategy:
1. Divide overlay crop into 8x8 grid
2. For each square, compare against a library of piece templates
3. Pick the best-matching piece (or empty) per square
4. Validate the resulting board position
"""

import io
import logging
from functools import lru_cache

import chess
import chess.svg
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Piece type enum for classification.
# 0 = empty, 1-6 = white pieces (P,N,B,R,Q,K), 7-12 = black pieces
PIECE_CLASSES = {
    0: None,
    1: chess.Piece(chess.PAWN, chess.WHITE),
    2: chess.Piece(chess.KNIGHT, chess.WHITE),
    3: chess.Piece(chess.BISHOP, chess.WHITE),
    4: chess.Piece(chess.ROOK, chess.WHITE),
    5: chess.Piece(chess.QUEEN, chess.WHITE),
    6: chess.Piece(chess.KING, chess.WHITE),
    7: chess.Piece(chess.PAWN, chess.BLACK),
    8: chess.Piece(chess.KNIGHT, chess.BLACK),
    9: chess.Piece(chess.BISHOP, chess.BLACK),
    10: chess.Piece(chess.ROOK, chess.BLACK),
    11: chess.Piece(chess.QUEEN, chess.BLACK),
    12: chess.Piece(chess.KING, chess.BLACK),
}

# Reverse mapping
PIECE_TO_CLASS = {v: k for k, v in PIECE_CLASSES.items() if v is not None}

# Common board themes: (light square color, dark square color) in BGR
BOARD_THEMES = {
    "lichess_default": {
        "light": "#F0D9B5",
        "dark": "#B58863",
    },
    "chess_com_green": {
        "light": "#EEEED2",
        "dark": "#769656",
    },
    "chess_com_brown": {
        "light": "#F0D9B5",
        "dark": "#B58863",
    },
}

# Template size for piece matching
TEMPLATE_SIZE = 64


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _crop_svg_margin(img: np.ndarray) -> np.ndarray:
    """Crop the margin added by chess.svg.board().

    chess.svg uses a viewBox of 390 = 15 (margin) + 8*45 (squares) + 15 (margin).
    This removes the margin to return just the 8x8 board area.
    """
    h, w = img.shape[:2]
    # Margin is 15/390 of the total size on each side
    margin_x = int(round(15 / 390 * w))
    margin_y = int(round(15 / 390 * h))
    board_w = w - 2 * margin_x
    board_h = h - 2 * margin_y
    return img[margin_y : margin_y + board_h, margin_x : margin_x + board_w]


def _render_board_to_cv2(
    board: chess.Board,
    size: int = 512,
    flipped: bool = False,
    colors: dict[str, str] | None = None,
) -> np.ndarray:
    """Render a chess board to an OpenCV BGR image (board area only, no margin)."""
    default_colors = {"light": "#F0D9B5", "dark": "#B58863"}
    c = colors or default_colors

    # Render slightly larger to account for margin, then crop
    render_size = int(size * 390 / 360)  # Ensure board area is at least `size`

    svg_text = chess.svg.board(
        board,
        size=render_size,
        flipped=flipped,
        colors={"square light": c["light"], "square dark": c["dark"]},
    )

    try:
        import cairosvg
        from PIL import Image

        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=render_size,
            output_height=render_size,
        )
        pil_img = Image.open(io.BytesIO(png_data)).convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Crop margin and resize to exact target size
        img = _crop_svg_margin(img)
        return cv2.resize(img, (size, size))
    except ImportError:
        logger.warning("cairosvg not available, using fallback renderer")
        return _render_simple_board_cv2(board, size, flipped, c)


def _render_simple_board_cv2(
    board: chess.Board,
    size: int,
    flipped: bool,
    colors: dict[str, str],
) -> np.ndarray:
    """Simple fallback renderer without SVG dependencies."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq_size = size // 8

    light_bgr = _hex_to_bgr(colors["light"])
    dark_bgr = _hex_to_bgr(colors["dark"])

    for rank in range(8):
        for file in range(8):
            x0 = file * sq_size
            y0 = rank * sq_size
            is_light = (file + rank) % 2 == 0
            color = light_bgr if is_light else dark_bgr
            img[y0 : y0 + sq_size, x0 : x0 + sq_size] = color

            display_rank = 7 - rank if not flipped else rank
            display_file = file if not flipped else 7 - file
            sq = chess.square(display_file, display_rank)
            piece = board.piece_at(sq)

            if piece is not None:
                # Draw a circle for the piece (crude but functional)
                cx = x0 + sq_size // 2
                cy = y0 + sq_size // 2
                radius = sq_size // 3
                piece_color = (255, 255, 255) if piece.color == chess.WHITE else (30, 30, 30)
                cv2.circle(img, (cx, cy), radius, piece_color, -1)
                cv2.circle(img, (cx, cy), radius, (0, 0, 0), 1)

    return img


def _build_piece_templates(
    theme: str = "lichess_default",
    template_size: int = TEMPLATE_SIZE,
) -> dict[int, dict[str, np.ndarray]]:
    """Build template images for each piece class on light and dark squares.

    Returns dict mapping piece_class -> {"light": img, "dark": img}.
    """
    colors = BOARD_THEMES.get(theme, BOARD_THEMES["lichess_default"])
    board_size = template_size * 8  # Render full board, crop squares

    templates: dict[int, dict[str, np.ndarray]] = {}

    # Empty squares
    empty_board = chess.Board(fen=None)  # Empty board
    empty_img = _render_board_to_cv2(empty_board, size=board_size, colors=colors)
    sq_size = board_size // 8

    # a1 is dark (file=0, rank=0 in display), a2 is light
    # In the rendered image (not flipped): row 0 = rank 8, row 7 = rank 1
    # Light square: (file + rank) % 2 == 0 in display coords
    # e.g., row=0, col=0 is light
    light_sq = empty_img[0:sq_size, 0:sq_size]
    dark_sq = empty_img[0:sq_size, sq_size : 2 * sq_size]

    templates[0] = {
        "light": cv2.resize(light_sq, (template_size, template_size)),
        "dark": cv2.resize(dark_sq, (template_size, template_size)),
    }

    # Each piece type on each square color
    for class_idx, piece in PIECE_CLASSES.items():
        if piece is None:
            continue

        # Place the piece on a light square (a8 = light) and dark square (b8 = dark)
        board = chess.Board(fen=None)
        # a8 = square 56 (light square in default orientation)
        board.set_piece_at(chess.A8, piece)
        img_light = _render_board_to_cv2(board, size=board_size, colors=colors)
        light_crop = img_light[0:sq_size, 0:sq_size]

        # b8 = square 57 (dark square)
        board = chess.Board(fen=None)
        board.set_piece_at(chess.B8, piece)
        img_dark = _render_board_to_cv2(board, size=board_size, colors=colors)
        dark_crop = img_dark[0:sq_size, sq_size : 2 * sq_size]

        templates[class_idx] = {
            "light": cv2.resize(light_crop, (template_size, template_size)),
            "dark": cv2.resize(dark_crop, (template_size, template_size)),
        }

    return templates


class OverlayReader:
    """Read chess board state from a 2D overlay image."""

    def __init__(self, board_theme: str = "lichess_default"):
        self.board_theme = board_theme
        self._templates = _build_piece_templates(board_theme)

    def read_board(
        self,
        overlay_crop: np.ndarray,
        flipped: bool = False,
    ) -> chess.Board | None:
        """Extract board position from a cropped overlay image.

        Args:
            overlay_crop: BGR image of just the 2D board overlay region.
            flipped: True if Black is at the bottom of the overlay.

        Returns:
            chess.Board with the detected position, or None if invalid.
        """
        h, w = overlay_crop.shape[:2]
        if h < 32 or w < 32:
            return None

        # Resize to a clean multiple of 8 to avoid pixel aliasing at
        # square boundaries (e.g., 300/8 = 37.5 causes artifacts).
        canonical_size = TEMPLATE_SIZE * 8  # 512
        overlay_crop = cv2.resize(overlay_crop, (canonical_size, canonical_size))
        h, w = canonical_size, canonical_size

        sq_h = h // 8
        sq_w = w // 8

        board = chess.Board(fen=None)

        for row in range(8):
            for col in range(8):
                # Crop the square
                y1 = row * sq_h
                y2 = y1 + sq_h
                x1 = col * sq_w
                x2 = x1 + sq_w
                square_img = overlay_crop[y1:y2, x1:x2]

                # Map display coordinates to chess square
                if not flipped:
                    chess_file = col
                    chess_rank = 7 - row
                else:
                    chess_file = 7 - col
                    chess_rank = row

                sq = chess.square(chess_file, chess_rank)

                # Determine if this is a light or dark square
                is_light = (col + row) % 2 == 0

                # Classify the square
                piece_class = self._classify_square(square_img, is_light)
                piece = PIECE_CLASSES.get(piece_class)

                if piece is not None:
                    board.set_piece_at(sq, piece)

        # Validate
        if not self._validate_board(board):
            logger.warning("Board validation failed")
            return None

        return board

    def read_fen(
        self,
        overlay_crop: np.ndarray,
        flipped: bool = False,
    ) -> str | None:
        """Extract FEN string from overlay crop.

        Returns only the piece placement part of FEN (no castling/en passant).
        """
        board = self.read_board(overlay_crop, flipped)
        if board is None:
            return None
        return board.board_fen()

    def _classify_square(
        self,
        square_img: np.ndarray,
        is_light: bool,
    ) -> int:
        """Classify a single square image.

        Two-step approach:
        1. Use pixel variance to detect empty vs occupied
        2. For occupied squares, template-match to identify the piece

        Returns piece class index (0 = empty, 1-12 = pieces).
        """
        gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY) if len(square_img.shape) == 3 else square_img
        variance = float(np.var(gray))

        # Empty squares on rendered boards have low variance (solid fills).
        # After canonical resize (to handle non-multiple-of-8 sizes),
        # empty squares have variance up to ~130 from interpolation artifacts.
        # Pieces have much higher variance (~1500+).
        if variance < 200:
            return 0  # Empty

        # Square is occupied - find the best matching piece
        sq_type = "light" if is_light else "dark"
        resized = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))

        best_score = -1.0
        best_class = 1  # Default to some piece if all scores are bad

        for class_idx, templates in self._templates.items():
            if class_idx == 0:
                continue  # Skip empty template

            template = templates[sq_type]
            score = self._match_score(resized, template)

            if score > best_score:
                best_score = score
                best_class = class_idx

        return best_class

    def _match_score(self, img: np.ndarray, template: np.ndarray) -> float:
        """Compute similarity between two images.

        Uses sum of squared differences (lower = better match),
        inverted so higher = better.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Normalized cross-correlation
        img_norm = gray_img - np.mean(gray_img)
        tpl_norm = gray_tpl - np.mean(gray_tpl)

        img_std = np.std(img_norm)
        tpl_std = np.std(tpl_norm)

        if img_std < 1e-6 or tpl_std < 1e-6:
            return 0.0

        ncc = np.mean(img_norm * tpl_norm) / (img_std * tpl_std)
        return float(ncc)

    def _validate_board(self, board: chess.Board) -> bool:
        """Basic sanity checks on detected board position."""
        white_kings = 0
        black_kings = 0

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                if piece.piece_type == chess.KING:
                    if piece.color == chess.WHITE:
                        white_kings += 1
                    else:
                        black_kings += 1

        # Must have exactly 1 king per side
        if white_kings != 1 or black_kings != 1:
            return False

        # No pawns on first or last rank
        for sq in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN:
                return False

        return True
