"""Diagnostic tools for the overlay pipeline.

Provides functions to:
- Test overlay detection + board reading on a single image
- Inspect .pt training clip files
- Test the overlay reader on a specific image region
"""

import logging
import os

import chess
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── overlay-test: full pipeline test on a single image ───────────────────


def test_image(
    image_path: str,
    overlay_bbox: tuple[int, int, int, int] | None = None,
    flipped: bool = False,
    theme: str = "lichess_default",
    output_path: str | None = None,
) -> None:
    """Run scanner + reader on a single image and produce annotated output.

    Args:
        image_path: Path to a screenshot/frame image.
        overlay_bbox: Optional manual overlay bbox (x, y, w, h). If None,
                      auto-detect via the scanner.
        flipped: Whether the board is from Black's perspective.
        theme: Board theme for the reader.
        output_path: Where to save the annotated image. Defaults to
                     {input_stem}_annotated.png next to the input.
    """
    from pipeline.overlay.overlay_reader import OverlayReader
    from pipeline.overlay.scanner import detect_overlay_in_frame

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"Image: {image_path} ({w}x{h})")
    print()

    # ── Step 1: Detect overlay region ──
    if overlay_bbox is not None:
        x, y, bw, bh = overlay_bbox
        print(f"Using manual overlay bbox: x={x}, y={y}, w={bw}, h={bh}")
        detection_score = None
    else:
        print("Scanning for 2D overlay board...")
        detection = detect_overlay_in_frame(frame)
        if not detection.found:
            print("  NOT FOUND: No 2D overlay board detected in this image.")
            print("  Tip: Try providing --overlay x,y,w,h manually.")
            return
        x, y, bw, bh = detection.bbox
        detection_score = detection.score
        print(f"  FOUND: bbox=({x}, {y}, {bw}, {bh}), score={detection_score:.3f}")

    print()

    # ── Step 2: Read board from overlay region ──
    overlay_crop = frame[y : y + bh, x : x + bw]
    reader = OverlayReader(board_theme=theme)
    board = reader.read_board(overlay_crop, flipped=flipped)

    if board is None:
        print("READER FAILED: Could not extract a valid board from the overlay region.")
        print("  Possible causes:")
        print("  - Overlay bbox doesn't align with the board squares")
        print("  - Board theme mismatch (try --theme chess_com_green)")
        print("  - Image quality too low")
    else:
        print("Extracted board:")
        print()
        # Print ASCII board indented
        for line in str(board).split("\n"):
            print(f"  {line}")
        print()
        print(f"FEN: {board.board_fen()}")
        piece_count = sum(1 for sq in chess.SQUARES if board.piece_at(sq) is not None)
        print(f"Pieces on board: {piece_count}")

    print()

    # ── Step 3: Annotated output ──
    annotated = frame.copy()

    # Draw overlay bbox
    color = (0, 255, 0) if board is not None else (0, 0, 255)
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

    # Label
    label = "OVERLAY"
    if detection_score is not None:
        label += f" (score={detection_score:.2f})"
    cv2.putText(
        annotated, label, (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )

    # Draw 8x8 grid over overlay region
    sq_w = bw // 8
    sq_h = bh // 8
    for i in range(1, 8):
        # Vertical lines
        lx = x + i * sq_w
        cv2.line(annotated, (lx, y), (lx, y + bh), (0, 255, 255), 1)
        # Horizontal lines
        ly = y + i * sq_h
        cv2.line(annotated, (x, ly), (x + bw, ly), (0, 255, 255), 1)

    # Label squares with piece symbols
    if board is not None:
        for row in range(8):
            for col in range(8):
                if not flipped:
                    chess_file = col
                    chess_rank = 7 - row
                else:
                    chess_file = 7 - col
                    chess_rank = row

                sq = chess.square(chess_file, chess_rank)
                piece = board.piece_at(sq)

                if piece is not None:
                    symbol = piece.symbol()
                    cx = x + col * sq_w + sq_w // 2 - 5
                    cy = y + row * sq_h + sq_h // 2 + 5
                    cv2.putText(
                        annotated, symbol, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                    )

    # FEN text at bottom
    if board is not None:
        fen_text = f"FEN: {board.board_fen()}"
        cv2.putText(
            annotated, fen_text, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Save
    if output_path is None:
        stem = os.path.splitext(image_path)[0]
        output_path = f"{stem}_annotated.png"

    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved: {output_path}")


# ── inspect-clip: inspect a .pt training clip file ───────────────────────


def inspect_clip(
    clip_path: str,
    save_frames: bool = False,
    output_dir: str | None = None,
) -> None:
    """Inspect a .pt training clip file.

    Args:
        clip_path: Path to a .pt clip file.
        save_frames: If True, save individual frames as images.
        output_dir: Directory for saved frames.
    """
    import torch

    try:
        from argus.chess.move_vocabulary import NO_MOVE_IDX, UNKNOWN_IDX, get_vocabulary
        vocab = get_vocabulary()
    except ImportError:
        print("WARNING: argus package not installed. Move decoding unavailable.")
        vocab = None

    if not os.path.exists(clip_path):
        print(f"ERROR: File not found: {clip_path}")
        return

    clip = torch.load(clip_path, map_location="cpu", weights_only=False)

    print(f"Clip: {clip_path}")
    print(f"Size: {os.path.getsize(clip_path) / 1024 / 1024:.1f} MB")
    print()

    # ── Tensor summary ──
    print("Tensors:")
    for key in sorted(clip.keys()):
        val = clip[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s}  shape={str(list(val.shape)):20s}  dtype={val.dtype}")
        else:
            print(f"  {key:20s}  type={type(val).__name__}  value={repr(val)[:60]}")

    print()

    # ── Frame summary ──
    if "frames" in clip:
        frames = clip["frames"]
        T = frames.shape[0]
        print(f"Frames: {T} total")
        print(f"  Shape per frame: {list(frames.shape[1:])}")
        print(f"  Pixel range: [{frames.min().item()}, {frames.max().item()}]")
        print()

    # ── Move sequence ──
    if "move_targets" in clip and vocab is not None:
        move_targets = clip["move_targets"]
        detect_targets = clip.get("detect_targets")
        T = move_targets.shape[0]

        total_moves = 0
        no_move_count = 0
        unknown_count = 0
        moves_list = []

        for t in range(T):
            idx = move_targets[t].item()
            if idx == NO_MOVE_IDX:
                no_move_count += 1
            elif idx == UNKNOWN_IDX:
                unknown_count += 1
            else:
                uci = vocab.index_to_uci(idx)
                detect = ""
                if detect_targets is not None:
                    detect = f"  detect={detect_targets[t].item():.0f}"
                moves_list.append((t, uci, detect))
                total_moves += 1

        print(f"Moves: {total_moves} move frames, {no_move_count} NO_MOVE frames, {unknown_count} UNKNOWN frames")
        print()

        if moves_list:
            print("Move sequence:")
            for t, uci, detect in moves_list:
                print(f"  frame {t:4d}: {uci}{detect}")
            print()

            # ── Replay validation ──
            print("Game replay validation:")
            board = chess.Board()
            valid = True
            for i, (t, uci, _) in enumerate(moves_list):
                try:
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        print(f"  ILLEGAL at ply {i}: {uci} (frame {t})")
                        valid = False
                        break
                    board.push(move)
                except ValueError:
                    print(f"  INVALID UCI at ply {i}: {uci} (frame {t})")
                    valid = False
                    break

            if valid:
                print(f"  VALID: All {len(moves_list)} moves form a legal game")
                print()
                print("Final position:")
                for line in str(board).split("\n"):
                    print(f"  {line}")
            print()

    # ── Legal masks summary ──
    if "legal_masks" in clip:
        legal_masks = clip["legal_masks"]
        avg_legal = legal_masks.float().sum(dim=1).mean().item()
        print(f"Legal masks: avg {avg_legal:.1f} legal moves per frame")
        print()

    # ── Save frames ──
    if save_frames and "frames" in clip:
        frames = clip["frames"]
        if output_dir is None:
            stem = os.path.splitext(clip_path)[0]
            output_dir = f"{stem}_frames"

        os.makedirs(output_dir, exist_ok=True)

        for t in range(frames.shape[0]):
            frame = frames[t]
            if frame.dtype == torch.uint8:
                img = frame.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
            else:
                img = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            # RGB → BGR for cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frame_path = os.path.join(output_dir, f"frame_{t:04d}.png")
            cv2.imwrite(frame_path, img)

        print(f"Saved {frames.shape[0]} frames to: {output_dir}")


# ── overlay-test-reader: test just the reader on a crop ──────────────────


def test_reader(
    image_path: str,
    overlay_bbox: tuple[int, int, int, int],
    flipped: bool = False,
    theme: str = "lichess_default",
) -> None:
    """Test overlay reader on a specific image region.

    Args:
        image_path: Path to the image.
        overlay_bbox: Overlay region (x, y, w, h).
        flipped: Board from Black's perspective.
        theme: Board theme for piece recognition.
    """
    from pipeline.overlay.overlay_reader import OverlayReader

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return

    x, y, w, h = overlay_bbox
    print(f"Image: {image_path}")
    print(f"Overlay crop: x={x}, y={y}, w={w}, h={h}")
    print(f"Theme: {theme}, Flipped: {flipped}")
    print()

    overlay_crop = frame[y : y + h, x : x + w]

    if overlay_crop.size == 0:
        print("ERROR: Crop region is empty. Check coordinates.")
        return

    reader = OverlayReader(board_theme=theme)

    # Detailed per-square classification
    sq_h = overlay_crop.shape[0] // 8
    sq_w = overlay_crop.shape[1] // 8

    print("Per-square analysis:")
    print("     a    b    c    d    e    f    g    h")

    for row in range(8):
        if not flipped:
            rank_label = str(8 - row)
        else:
            rank_label = str(row + 1)

        cells = []
        for col in range(8):
            y1 = row * sq_h
            x1 = col * sq_w
            sq_img = overlay_crop[y1 : y1 + sq_h, x1 : x1 + sq_w]
            gray = cv2.cvtColor(sq_img, cv2.COLOR_BGR2GRAY)
            var = float(np.var(gray))
            is_light = (col + row) % 2 == 0

            if var < 100:
                cells.append(f"  .  ")
            else:
                piece_class = reader._classify_square(sq_img, is_light)
                from pipeline.overlay.overlay_reader import PIECE_CLASSES
                piece = PIECE_CLASSES.get(piece_class)
                symbol = piece.symbol() if piece else "?"
                cells.append(f" {symbol}({int(var):3d})")

        print(f" {rank_label} {''.join(cells)}")

    print()

    # Full board read
    board = reader.read_board(overlay_crop, flipped=flipped)
    if board is not None:
        print("Extracted board:")
        for line in str(board).split("\n"):
            print(f"  {line}")
        print()
        print(f"FEN: {board.board_fen()}")
    else:
        print("READER FAILED: Board validation did not pass.")

    # Save annotated crop
    annotated = overlay_crop.copy()
    for i in range(1, 8):
        cv2.line(annotated, (i * sq_w, 0), (i * sq_w, overlay_crop.shape[0]), (0, 255, 255), 1)
        cv2.line(annotated, (0, i * sq_h), (overlay_crop.shape[1], i * sq_h), (0, 255, 255), 1)

    stem = os.path.splitext(image_path)[0]
    crop_output = f"{stem}_reader_annotated.png"
    cv2.imwrite(crop_output, annotated)
    print(f"\nAnnotated crop saved: {crop_output}")
