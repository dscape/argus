"""CLI entry point for the Argus data pipeline."""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()


def cmd_db_init(args):
    """Apply database schema."""
    from pipeline.db.connection import init_schema
    init_schema()


def cmd_seed_channels(args):
    """Seed channel config from YAML."""
    from pipeline.setup.channel_seeder import seed_channels
    seed_channels()


def cmd_resolve_channels(args):
    """Resolve channel handles to IDs."""
    from pipeline.crawl.channel_resolver import resolve_channels
    resolve_channels()


def cmd_crawl(args):
    """Crawl YouTube channels for video metadata."""
    from pipeline.crawl.crawl_videos import crawl_all
    crawl_all(channel_handle=args.channel, refresh=args.refresh)


def cmd_screen(args):
    """Screen crawled videos for overlay + OTB suitability."""
    from pipeline.screen.screen_pipeline import screen_all
    screen_all(channel_handle=args.channel, limit=args.limit)


def cmd_download(args):
    """Download approved videos."""
    from pipeline.download.video_downloader import download_approved_videos
    download_approved_videos(limit=args.limit)


def cmd_calibrate(args):
    """Set layout calibration for a channel."""
    from pipeline.overlay.calibration import LayoutCalibration, set_calibration

    def parse_bbox(s):
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected x,y,w,h but got: {s}")
        return tuple(parts)

    overlay = parse_bbox(args.overlay)
    camera = parse_bbox(args.camera)

    if args.resolution:
        ref = tuple(int(x) for x in args.resolution.split("x"))
    else:
        ref = (1920, 1080)

    delay = args.delay if args.delay is not None else 2.0

    cal = LayoutCalibration(
        overlay=overlay,
        camera=camera,
        ref_resolution=ref,
        board_flipped=args.flipped,
        board_theme=args.theme or "lichess_default",
        move_delay_seconds=delay,
    )

    set_calibration(args.channel, cal)
    print(f"Calibration saved for {args.channel}")
    print(f"  Overlay: {overlay}")
    print(f"  Camera:  {camera}")
    print(f"  Ref resolution: {ref}")
    print(f"  Move delay: {delay}s")


def cmd_generate_clips(args):
    """Generate training clips from approved overlay videos."""
    from pipeline.db.connection import get_conn
    from pipeline.download.video_downloader import get_video_path
    from pipeline.overlay.overlay_clip_generator import generate_from_video

    if args.video_id:
        # Single video mode — look up channel handle from DB
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT channel_handle FROM youtube_videos WHERE video_id = %s",
                    (args.video_id,),
                )
                row = cur.fetchone()
        if not row:
            print(f"Video {args.video_id} not found in DB.")
            return
        channel_handle = row[0] or ""
        video_path = get_video_path(args.video_id)
        if video_path is None:
            print(f"Video {args.video_id} not downloaded.")
            return
        results = generate_from_video(
            video_path,
            channel_handle=channel_handle,
            min_moves_per_segment=args.min_moves,
        )
        for r in results:
            print(f"  {r['filepath']} ({r['num_moves']} moves, {r['num_frames']} frames)")
        print(f"\nGenerated {len(results)} clip(s)")
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT video_id, channel_handle
                FROM youtube_videos
                WHERE screening_status = 'approved'
                ORDER BY published_at DESC
            """
            params: list = []

            if args.channel:
                query = query.replace(
                    "ORDER BY",
                    "AND channel_handle = %s ORDER BY",
                )
                params.append(args.channel)

            if args.limit:
                query += " LIMIT %s"
                params.append(args.limit)

            cur.execute(query, params)
            videos = cur.fetchall()

    if not videos:
        print("No approved videos found.")
        return

    total_clips = 0
    for video_id, channel_handle in videos:
        video_path = get_video_path(video_id)
        if video_path is None:
            print(f"  Skipping {video_id}: not downloaded")
            continue

        results = generate_from_video(
            video_path,
            channel_handle=channel_handle,
            min_moves_per_segment=args.min_moves,
        )
        if results:
            total_clips += len(results)
            for r in results:
                print(
                    f"  {r['filepath']} "
                    f"({r['num_moves']} moves, {r['num_frames']} frames)"
                )

    print(f"\nGenerated {total_clips} clip(s) total")


def cmd_inspect(args):
    """Inspect videos by extracting frames and detecting overlay + OTB."""
    from pipeline.db.connection import get_conn
    from pipeline.screen.dual_region_detector import (
        overlay_bbox_to_json,
        screen_video,
    )

    # Build query for target videos
    if args.video_id:
        # Single video
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT video_id, channel_handle, title"
                    " FROM youtube_videos WHERE video_id = %s",
                    (args.video_id,),
                )
                videos = cur.fetchall()
    else:
        # Batch by filters
        with get_conn() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT video_id, channel_handle, title
                    FROM youtube_videos
                    WHERE 1=1
                """
                params: list = []

                status_filter = args.status or "approved"
                if status_filter == "unscreened":
                    query += " AND screening_status IS NULL"
                else:
                    query += " AND screening_status = %s"
                    params.append(status_filter)

                if args.channel:
                    query += " AND channel_handle = %s"
                    params.append(args.channel)

                query += " ORDER BY published_at DESC"

                if args.limit:
                    query += " LIMIT %s"
                    params.append(args.limit)

                cur.execute(query, params)
                videos = cur.fetchall()

    if not videos:
        print("No videos found matching criteria.")
        return

    print(f"Inspecting {len(videos)} video(s)...\n")

    approved_count = 0
    rejected_count = 0
    failed_count = 0

    for video_id, channel_handle, title in videos:
        url = f"https://www.youtube.com/watch?v={video_id}"
        short_title = title[:60] + "..." if len(title) > 60 else title
        print(f"  {video_id}  {short_title}")

        try:
            result = screen_video(url)

            status = "approved" if result.approved else "rejected"
            bbox_json = overlay_bbox_to_json(result.overlay_bbox)

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE youtube_videos
                        SET screening_status = %s,
                            screening_confidence = %s,
                            overlay_bbox = %s,
                            has_otb_footage = %s,
                            layout_type = COALESCE(%s, layout_type),
                            updated_at = now()
                        WHERE video_id = %s
                        """,
                        (
                            status,
                            result.otb_confidence,
                            bbox_json,
                            result.has_otb,
                            "overlay" if result.has_overlay else (
                                "otb_only" if result.has_otb else None
                            ),
                            video_id,
                        ),
                    )
                    conn.commit()

            overlay_str = (
                f"overlay={result.overlay_score:.2f}"
                if result.has_overlay else "no overlay"
            )
            otb_str = f"otb={result.otb_confidence:.2f}" if result.has_otb else "no otb"
            print(f"    -> {status.upper()} ({overlay_str}, {otb_str})")

            if result.approved:
                approved_count += 1
            else:
                rejected_count += 1

        except Exception as e:
            failed_count += 1
            print(f"    -> FAILED: {e}")

    print(f"\nDone: {approved_count} approved, {rejected_count} rejected, {failed_count} failed")


def cmd_overlay_test(args):
    """Test overlay detection + reading on a single image."""
    from pipeline.overlay.diagnostics import test_image

    def parse_bbox(s):
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected x,y,w,h but got: {s}")
        return tuple(parts)

    overlay_bbox = parse_bbox(args.overlay) if args.overlay else None

    test_image(
        image_path=args.image,
        overlay_bbox=overlay_bbox,
        flipped=args.flipped,
        theme=args.theme or "lichess_default",
        output_path=args.output,
    )


def cmd_overlay_test_reader(args):
    """Test overlay reader on a specific image region."""
    from pipeline.overlay.diagnostics import test_reader

    parts = [int(x) for x in args.overlay.split(",")]
    if len(parts) != 4:
        print("ERROR: --overlay must be x,y,w,h")
        return

    test_reader(
        image_path=args.image,
        overlay_bbox=tuple(parts),
        flipped=args.flipped,
        theme=args.theme or "lichess_default",
    )


def cmd_overlay_yolo_export(args):
    """Export bbox training annotations as a YOLO dataset."""
    from pathlib import Path

    from pipeline.overlay.yolo_dataset import export_overlay_yolo_dataset

    export = export_overlay_yolo_dataset(
        dataset_dir=Path(args.out_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Dataset: {export.dataset_dir}")
    print(f"  YAML: {export.dataset_yaml}")
    print(f"  Manifest: {export.manifest_path}")
    print(
        f"  Train: {export.train.images} images "
        f"({export.train.positives} pos, {export.train.negatives} neg)"
    )
    print(
        f"  Val:   {export.val.images} images "
        f"({export.val.positives} pos, {export.val.negatives} neg)"
    )
    print(
        f"  Test:  {export.test.images} images "
        f"({export.test.positives} pos, {export.test.negatives} neg)"
    )



def cmd_overlay_yolo_train(args):
    """Train the default YOLO overlay detector."""
    from pathlib import Path

    from pipeline.overlay.yolo_train import train_overlay_yolo

    result = train_overlay_yolo(
        dataset_yaml=Path(args.data),
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=Path(args.project),
        name=args.name,
    )

    print(f"Run dir: {result.save_dir}")
    print(f"Best weights: {result.best_weights}")
    print("Use for ad-hoc evaluation via:")
    print(
        "  ARGUS_OVERLAY_YOLO_WEIGHTS="
        f"{result.best_weights} .venv/bin/python3 scripts/visualize_overlay_tests.py"
    )



def cmd_otb_yolo_export(args):
    """Export cached OTB/reject frames as a YOLO dataset."""
    from pathlib import Path

    from pipeline.screen.otb_yolo_dataset import export_otb_yolo_dataset

    export = export_otb_yolo_dataset(
        dataset_dir=Path(args.out_dir),
        positive_video_limit=args.positive_video_limit,
        negative_video_limit=args.negative_video_limit,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print(f"Dataset: {export.dataset_dir}")
    print(f"  YAML: {export.dataset_yaml}")
    print(f"  Manifest: {export.manifest_path}")
    print(
        f"  Train: {export.train.images} images "
        f"({export.train.positives} pos, {export.train.negatives} neg)"
    )
    print(
        f"  Val:   {export.val.images} images "
        f"({export.val.positives} pos, {export.val.negatives} neg)"
    )
    print(
        f"  Test:  {export.test.images} images "
        f"({export.test.positives} pos, {export.test.negatives} neg)"
    )



def cmd_otb_yolo_train(args):
    """Train the default YOLO OTB-board detector."""
    from pathlib import Path

    from pipeline.screen.otb_yolo_train import train_otb_yolo

    result = train_otb_yolo(
        dataset_yaml=Path(args.data),
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=Path(args.project),
        name=args.name,
    )

    print(f"Run dir: {result.save_dir}")
    print(f"Best weights: {result.best_weights}")
    print("Use for runtime experiments via:")
    print(
        "  ARGUS_OTB_YOLO_WEIGHTS="
        f"{result.best_weights} "
        ".venv/bin/python3 -m pipeline.cli auto-calibrate --video-id <video_id>"
    )



def cmd_inspect_clip(args):
    """Inspect a .pt training clip file."""
    from pipeline.overlay.diagnostics import inspect_clip

    inspect_clip(
        clip_path=args.file,
        save_frames=args.save_frames,
        output_dir=args.output_dir,
    )


def cmd_ai_extract(args):
    """Pre-compute DINOv2 features for labelled videos."""
    from pipeline.screen.ai_train import extract_and_cache_features
    extract_and_cache_features(device=args.device)


def cmd_ai_train(args):
    """Train the AI screening classifier."""
    from pipeline.screen.ai_train import train
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device)


def cmd_ai_eval(args):
    """Evaluate the AI screening classifier."""
    from pipeline.screen.ai_eval import calibrate_threshold, evaluate
    evaluate(checkpoint_path=args.checkpoint)
    print()
    calibrate_threshold(
        checkpoint_path=args.checkpoint,
        target_precision=args.target_precision,
    )


def cmd_ai_screen(args):
    """Run AI screening on unscreened videos."""
    from pipeline.screen.screen_pipeline import screen_with_ai
    screen_with_ai(
        channel_handle=args.channel,
        limit=args.limit,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
    )


def cmd_ai_retrain(args):
    """Full retrain: extract new features, train, evaluate, optionally screen."""
    from pipeline.screen.ai_eval import calibrate_threshold, evaluate
    from pipeline.screen.ai_train import extract_and_cache_features, train

    print("=== Step 1/3: Extract features for new videos ===")
    extract_and_cache_features(device=args.device)

    print("\n=== Step 2/3: Train classifier ===")
    checkpoint = train(
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=args.device,
    )

    print("\n=== Step 3/3: Evaluate and calibrate ===")
    evaluate(checkpoint_path=checkpoint)
    print()
    calibrate_threshold(
        checkpoint_path=checkpoint,
        target_precision=args.target_precision,
    )

    if args.screen:
        from pipeline.screen.screen_pipeline import screen_with_ai
        print("\n=== Screening unscreened videos with new weights ===")
        screen_with_ai(
            checkpoint_path=checkpoint,
            threshold=args.threshold,
            device=args.device,
            limit=args.screen_limit,
        )


def cmd_ai_profile(args):
    """Profile AI screening pipeline for a single video."""
    from pipeline.screen.ai_profiler import format_profile, profile_video
    records, prediction = profile_video(
        video_id=args.video_id,
        checkpoint_path=args.checkpoint,
        device=args.device,
        force_uncached=args.force_uncached,
    )
    format_profile(records, prediction, video_id=args.video_id)


def cmd_auto_calibrate(args):
    """Auto-propose calibration for a channel from screening data."""
    from pipeline.overlay.auto_calibration import (
        propose_calibration,
        propose_calibration_for_channel,
    )
    from pipeline.overlay.calibration import LayoutCalibration, set_calibration

    if args.video_id:
        proposal = propose_calibration(args.video_id)
    else:
        proposal = propose_calibration_for_channel(args.channel)

    if proposal is None:
        print("Could not generate calibration proposal.")
        print("Ensure the channel has approved overlay videos with detected bounding boxes.")
        return

    print(f"\nCalibration proposal for {args.channel}:")
    print(f"  Overlay:      {proposal.overlay}")
    print(f"  Camera:       {proposal.camera}")
    print(f"  Resolution:   {proposal.ref_resolution}")
    print(f"  Theme:        {proposal.board_theme} (confidence={proposal.theme_confidence:.2f})")
    flip_conf = proposal.orientation_confidence
    print(f"  Flipped:      {proposal.board_flipped} (confidence={flip_conf:.2f})")

    if args.apply:
        cal = LayoutCalibration(
            overlay=proposal.overlay,
            camera=proposal.camera,
            ref_resolution=proposal.ref_resolution,
            board_flipped=proposal.board_flipped,
            board_theme=proposal.board_theme,
        )
        set_calibration(args.channel, cal)
        print(f"\n  Calibration saved for {args.channel}")
    else:
        print("\n  Run with --apply to save this calibration.")


def cmd_smoke_test(args):
    """Run quick smoke tests (no DB required)."""
    import chess

    print("=== Smoke Test: Hard Cut Detection ===")
    from pipeline.overlay.overlay_move_detector import count_fen_differences

    full_reset = count_fen_differences(chess.STARTING_BOARD_FEN, "8/8/8/8/8/8/8/8")
    print(f"  Full reset (expect 32): {full_reset}")
    e2e4 = count_fen_differences(
        chess.STARTING_BOARD_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
    )
    print(f"  e2e4 (expect 2): {e2e4}")
    assert full_reset == 32, f"FAIL: expected 32, got {full_reset}"
    assert e2e4 == 2, f"FAIL: expected 2, got {e2e4}"
    print("  PASS")

    print("\n=== Smoke Test: AI Classifier ===")
    import torch

    from pipeline.screen.ai_classifier import ScreeningClassifier

    model = ScreeningClassifier()
    emb = torch.randn(1, 3, 768)
    scan = torch.randn(1, 3)
    otb = torch.randn(1, 3)
    logits = model(emb, scan, otb)
    assert logits.shape == (1, 3), f"FAIL: expected (1, 3), got {logits.shape}"
    print(f"  Logits shape: {logits.shape} (overlay, otb_only, reject)")
    print("  PASS")

    print("\nAll smoke tests passed.")


def cmd_inspect_calibration(args):
    """Inspect saved calibration for a channel."""
    from pipeline.overlay.calibration import get_calibration

    cal = get_calibration(args.channel)
    if cal is None:
        print(f"No calibration found for {args.channel}")
        return

    print(f"Calibration for {args.channel}:")
    print(f"  Overlay:      {cal.overlay}")
    print(f"  Camera:       {cal.camera}")
    print(f"  Resolution:   {cal.ref_resolution}")
    print(f"  Theme:        {cal.board_theme}")
    print(f"  Flipped:      {cal.board_flipped}")
    print(f"  Move delay:   {cal.move_delay_seconds}s")


def cmd_ai_extract_status(args):
    """Report progress of DINOv2 feature caching."""
    import os

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "screening",
        "dataset",
        "torch",
    )

    labels = {0: 0, 1: 0, 2: 0}
    vertical = 0
    errors = 0

    if os.path.isdir(cache_dir):
        import torch

        for f in os.listdir(cache_dir):
            if not f.endswith(".pt"):
                continue
            try:
                data = torch.load(
                    os.path.join(cache_dir, f), map_location="cpu", weights_only=True
                )
                if data.get("vertical"):
                    vertical += 1
                else:
                    lbl = data.get("label", -1)
                    if lbl in labels:
                        labels[lbl] += 1
                    else:
                        errors += 1
            except Exception:
                errors += 1

    cached = sum(labels.values()) + vertical

    # Count labelled videos in DB
    total = 0
    try:
        from pipeline.db.connection import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM youtube_videos "
                    "WHERE screening_status IN ('approved', 'rejected')"
                )
                total = cur.fetchone()[0]
    except Exception as e:
        print(f"  (Could not query DB: {e})")

    pct = round(100 * cached / total, 1) if total > 0 else 0
    remaining = max(0, total - cached)

    print("\nAI feature extraction progress:")
    print(f"  {cached}/{total} ({pct}%) — overlay={labels[0]}, otb={labels[1]}, "
          f"reject={labels[2]}, vertical={vertical}")
    if errors:
        print(f"  Errors: {errors} files could not be loaded")
    print(f"  Cache dir: {cache_dir}")

    if remaining > 0:
        print(f"\n  ~{remaining} videos remaining (~{remaining * 4}s at ~4s/video)")
    else:
        print("\n  All videos cached.")


def cmd_stats(args):
    """Print pipeline statistics."""
    from pipeline.db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            stats = {}

            _ALLOWED_TABLES = frozenset([
                "crawl_channels", "youtube_videos",
                "training_clips", "api_quota_log",
            ])
            for table in _ALLOWED_TABLES:
                try:
                    assert table in _ALLOWED_TABLES, f"Invalid table: {table}"
                    cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table from allowlist
                    stats[table] = cur.fetchone()[0]
                except Exception:
                    stats[table] = "N/A"
                    conn.rollback()

            # Resolved channels
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM crawl_channels WHERE channel_id NOT LIKE 'UNRESOLVED:%%'"
                )
                stats["resolved_channels"] = cur.fetchone()[0]
            except Exception:
                stats["resolved_channels"] = "N/A"
                conn.rollback()

            # Screening stats
            for status in ["approved", "rejected"]:
                try:
                    cur.execute(
                        "SELECT COUNT(*) FROM youtube_videos WHERE screening_status = %s",
                        (status,),
                    )
                    stats[f"screening_{status}"] = cur.fetchone()[0]
                except Exception:
                    stats[f"screening_{status}"] = "N/A"
                    conn.rollback()

            try:
                cur.execute(
                    "SELECT COUNT(*) FROM youtube_videos WHERE screening_status IS NULL"
                )
                stats["unscreened"] = cur.fetchone()[0]
            except Exception:
                stats["unscreened"] = "N/A"
                conn.rollback()

    print("\n" + "=" * 50)
    print("  ARGUS PIPELINE STATISTICS")
    print("=" * 50)
    print(f"  Crawl channels:         {stats['crawl_channels']}")
    print(f"    Resolved:             {stats['resolved_channels']}")
    print(f"  YouTube videos:         {stats['youtube_videos']}")
    print(f"    Unscreened:           {stats['unscreened']}")
    print(f"    Approved:             {stats['screening_approved']}")
    print(f"    Rejected:             {stats['screening_rejected']}")
    print(f"  Training clips:         {stats['training_clips']}")
    print(f"  API quota log entries:  {stats['api_quota_log']}")
    print("=" * 50)


def cmd_fetch_frames(args):
    """Fetch overlay frames (25/50/75%) from YouTube.

    Resolution modes:
    - fullres (default): 1920x1080 via yt-dlp (required for overlay detection)
    - hires: 1280x720 via YouTube thumbnails (fast, but too low for detection)
    - lowres: 480x360 via YouTube thumbnails (screening only)
    """
    from pipeline.db.connection import get_conn
    from pipeline.screen.frame_fetcher import (
        fetch_overlay_frames,
        fetch_overlay_frames_fullres,
        get_video_duration,
    )

    layout = getattr(args, "layout", "overlay")
    randomize = getattr(args, "random", False)

    with get_conn() as conn:
        with conn.cursor() as cur:
            conditions = ["screening_status = 'approved'"]
            params: list = []

            if layout == "overlay":
                # Include overlay + unclassified (NULL) videos
                conditions.append("(layout_type = 'overlay' OR layout_type IS NULL)")
            elif layout != "all":
                conditions.append("layout_type = %s")
                params.append(layout)

            if args.channel:
                conditions.append("channel_handle = %s")
                params.append(args.channel)

            where = " AND ".join(conditions)
            order = "RANDOM()" if randomize else "channel_handle, video_id"
            limit_clause = ""
            if args.limit:
                limit_clause = " LIMIT %s"
                params.append(args.limit)

            cur.execute(
                f"SELECT video_id, COALESCE(channel_handle, '')"  # noqa: S608
                f" FROM youtube_videos WHERE {where}"
                f" ORDER BY {order}{limit_clause}",
                params,
            )
            videos = cur.fetchall()

    if not videos:
        print("No overlay videos found.")
        return

    res_label = {
        "fullres": "1920x1080 (yt-dlp)",
        "hires": "1280x720",
        "lowres": "480x360",
    }
    print(f"Fetching {res_label[args.resolution]} frames for {len(videos)} videos\n")

    total_frames = 0
    for video_id, channel in videos:
        if args.resolution == "fullres":
            duration = get_video_duration(video_id)
            if duration <= 0:
                print(f"  {channel:30s} {video_id}  SKIPPED (no duration)")
                continue
            results = fetch_overlay_frames_fullres(video_id, duration)
        else:
            results = fetch_overlay_frames(
                video_id, hires=(args.resolution == "hires")
            )
        if results:
            res_str = ", ".join(f"{lbl} {w}x{h}" for lbl, w, h in results)
            print(f"  {channel:30s} {video_id}  {res_str}")
            total_frames += len(results)
        else:
            print(f"  {channel:30s} {video_id}  FAILED")

    print(f"\nDone: {total_frames} frames from {len(videos)} videos")


def cmd_analyze_video(args):
    """Analyze a local chess video into PGN and an annotated video."""
    from pathlib import Path

    from pipeline.analysis.config import VideoAnalysisConfig
    from pipeline.analysis.pipeline import VideoAnalysisPipeline

    config = VideoAnalysisConfig(
        fps=args.fps,
        device=args.device,
        reader_backend=args.reader,
        scene_backend=args.scene,
        annotate=not args.no_annotate,
        tts=args.tts,
        output_dir=Path(args.output),
        vlm_model=args.vlm_model,
    )

    result = VideoAnalysisPipeline(config).run(video_path=args.video)

    print(f"\nDone: {result.total_moves} moves detected in {len(result.segments)} game(s)")
    for pgn_path in result.pgn_files:
        print(f"  PGN: {pgn_path}")
    if result.annotated_video:
        print(f"  Video: {result.annotated_video}")


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Argus data pipeline: crawl, screen, download, generate clips",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # db-init
    subparsers.add_parser("db-init", help="Apply database schema")

    # seed-channels
    subparsers.add_parser("seed-channels", help="Seed channels from YAML config")

    # resolve-channels
    subparsers.add_parser("resolve-channels", help="Resolve channel handles to IDs")

    # crawl
    p = subparsers.add_parser("crawl", help="Crawl YouTube channels")
    p.add_argument("--channel", type=str, default=None, help="Crawl specific channel handle")
    p.add_argument("--refresh", action="store_true", help="Only fetch new videos")

    # screen
    p = subparsers.add_parser("screen", help="Screen videos for overlay + OTB suitability")
    p.add_argument("--channel", type=str, default=None, help="Screen specific channel handle")
    p.add_argument("--limit", type=int, default=None, help="Max videos to screen")

    # inspect
    p = subparsers.add_parser(
        "inspect", help="Inspect videos for overlay + OTB via frame extraction",
    )
    p.add_argument("--video-id", type=str, default=None, help="Inspect a single video by ID")
    p.add_argument("--channel", type=str, default=None, help="Filter by channel handle")
    p.add_argument("--status", type=str, default=None, help="Filter by status (default: approved)")
    p.add_argument("--limit", type=int, default=None, help="Max videos to inspect")

    # download
    p = subparsers.add_parser("download", help="Download approved videos")
    p.add_argument("--limit", type=int, default=None, help="Max videos to download")

    # calibrate
    p = subparsers.add_parser("calibrate", help="Set overlay layout calibration")
    p.add_argument("--channel", type=str, required=True, help="Channel handle (e.g. @STLChessClub)")
    p.add_argument("--overlay", type=str, required=True, help="Overlay bbox: x,y,w,h")
    p.add_argument("--camera", type=str, required=True, help="Camera bbox: x,y,w,h")
    p.add_argument(
        "--resolution", type=str, default=None,
        help="Reference resolution: WxH (default: 1920x1080)",
    )
    p.add_argument(
        "--flipped", action="store_true",
        help="Board is flipped (Black at bottom)",
    )
    p.add_argument(
        "--theme", type=str, default=None,
        help="Board theme (lichess_default, chess_com_green)",
    )
    p.add_argument(
        "--delay", type=float, default=None,
        help="Move delay in seconds (default: 2.0)",
    )

    # generate-clips
    p = subparsers.add_parser("generate-clips", help="Generate training clips from approved videos")
    p.add_argument("--channel", type=str, default=None, help="Generate for specific channel")
    p.add_argument("--video-id", type=str, default=None, help="Generate for a single video by ID")
    p.add_argument("--limit", type=int, default=None, help="Max videos to process")
    p.add_argument(
        "--min-moves",
        type=int,
        default=5,
        help="Minimum detected moves required to save a clip",
    )

    # overlay-test
    p = subparsers.add_parser("overlay-test", help="Test overlay pipeline on a screenshot")
    p.add_argument("--image", type=str, required=True, help="Path to screenshot image")
    p.add_argument(
        "--overlay", type=str, default=None,
        help="Manual overlay bbox: x,y,w,h (skip auto-detect)",
    )
    p.add_argument(
        "--flipped", action="store_true",
        help="Board is flipped (Black at bottom)",
    )
    p.add_argument(
        "--theme", type=str, default=None,
        help="Board theme (lichess_default, chess_com_green)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output path for annotated image",
    )

    # overlay-test-reader
    p = subparsers.add_parser(
        "overlay-test-reader",
        help="Test overlay reader on a specific region",
    )
    p.add_argument("--image", type=str, required=True, help="Path to screenshot image")
    p.add_argument("--overlay", type=str, required=True, help="Overlay bbox: x,y,w,h")
    p.add_argument("--flipped", action="store_true", help="Board is flipped (Black at bottom)")
    p.add_argument("--theme", type=str, default=None, help="Board theme")

    # overlay-yolo-export
    p = subparsers.add_parser(
        "overlay-yolo-export",
        help="Export overlay bbox training annotations as a YOLO dataset",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/overlay/yolo",
        help="Output dataset directory",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for non-fixture annotations",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split",
    )

    # overlay-yolo-train
    p = subparsers.add_parser(
        "overlay-yolo-train",
        help="Train the default YOLO overlay detector",
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/overlay/yolo/dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )
    p.add_argument(
        "--model",
        type=str,
        default="weights/yolo_base/yolo11n.pt",
        help="Bootstrap YOLO checkpoint path",
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (auto, cpu, mps, cuda)",
    )
    p.add_argument(
        "--project",
        type=str,
        default="outputs/overlay_yolo",
        help="Ultralytics project output directory",
    )
    p.add_argument(
        "--name",
        type=str,
        default="train",
        help="Ultralytics run name",
    )

    # otb-yolo-export
    p = subparsers.add_parser(
        "otb-yolo-export",
        help="Export OTB-board pseudo-labels as a YOLO dataset",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/screening/otb_yolo",
        help="Output dataset directory",
    )
    p.add_argument(
        "--positive-video-limit",
        type=int,
        default=None,
        help="Optional cap on approved otb_only videos",
    )
    p.add_argument(
        "--negative-video-limit",
        type=int,
        default=None,
        help="Optional cap on rejected videos used as negatives",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction by video",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Test fraction by video",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video-level splits",
    )

    # otb-yolo-train
    p = subparsers.add_parser(
        "otb-yolo-train",
        help="Train the default YOLO OTB-board detector",
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/screening/otb_yolo/dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )
    p.add_argument(
        "--model",
        type=str,
        default="weights/yolo_base/yolo11n.pt",
        help="Bootstrap YOLO checkpoint path",
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (auto, cpu, mps, cuda)",
    )
    p.add_argument(
        "--project",
        type=str,
        default="outputs/otb_yolo",
        help="Ultralytics project output directory",
    )
    p.add_argument(
        "--name",
        type=str,
        default="train",
        help="Ultralytics run name",
    )

    # inspect-clip
    p = subparsers.add_parser("inspect-clip", help="Inspect a .pt training clip file")
    p.add_argument("--file", type=str, required=True, help="Path to .pt clip file")
    p.add_argument("--save-frames", action="store_true", help="Save individual frames as images")
    p.add_argument("--output-dir", type=str, default=None, help="Directory for saved frames")

    # ai-extract
    p = subparsers.add_parser("ai-extract", help="Pre-compute DINOv2 features for labelled videos")
    p.add_argument("--device", type=str, default="cpu", help="Torch device (cpu, cuda, mps)")

    # ai-train
    p = subparsers.add_parser("ai-train", help="Train the AI screening classifier")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--device", type=str, default="cpu", help="Torch device")

    # ai-eval
    p = subparsers.add_parser(
        "ai-eval",
        help="Evaluate AI screening classifier and calibrate threshold",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: best.pt)",
    )
    p.add_argument(
        "--target-precision", type=float, default=0.95,
        help="Target precision for threshold",
    )

    # ai-screen
    p = subparsers.add_parser("ai-screen", help="Run AI screening on unscreened videos")
    p.add_argument("--channel", type=str, default=None, help="Screen specific channel")
    p.add_argument("--limit", type=int, default=None, help="Max videos to screen")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="Confidence threshold for auto-deciding",
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device")

    # ai-retrain
    p = subparsers.add_parser("ai-retrain", help="Full retrain: extract → train → eval (→ screen)")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--device", type=str, default="cpu", help="Torch device")
    p.add_argument(
        "--target-precision", type=float, default=0.95,
        help="Target precision for threshold calibration",
    )
    p.add_argument(
        "--screen", action="store_true",
        help="Also run ai-screen with new weights after training",
    )
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="Confidence threshold (used with --screen)",
    )
    p.add_argument(
        "--screen-limit", type=int, default=None,
        help="Max videos to screen (used with --screen)",
    )

    # ai-profile
    p = subparsers.add_parser(
        "ai-profile",
        help="Profile AI screening pipeline for a single video",
    )
    p.add_argument(
        "--video-id", type=str, required=True,
        help="YouTube video ID to profile",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: best.pt)",
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device")
    p.add_argument(
        "--force-uncached", action="store_true",
        help="Skip feature cache to force full pipeline",
    )

    # auto-calibrate
    p = subparsers.add_parser("auto-calibrate", help="Auto-propose calibration for a channel")
    p.add_argument("--channel", type=str, required=True, help="Channel handle (e.g. @STLChessClub)")
    p.add_argument(
        "--video-id", type=str, default=None,
        help="Use a specific video instead of multi-video consensus",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Save the proposed calibration (otherwise just print)",
    )

    # smoke-test
    subparsers.add_parser("smoke-test", help="Run quick smoke tests (no DB required)")

    # inspect-calibration
    p = subparsers.add_parser("inspect-calibration", help="Inspect saved calibration for a channel")
    p.add_argument("--channel", type=str, required=True, help="Channel handle (e.g. @STLChessClub)")

    # ai-extract-status
    subparsers.add_parser("ai-extract-status", help="Report AI feature extraction cache progress")

    # fetch-frames
    p = subparsers.add_parser("fetch-frames", help="Fetch overlay frames from YouTube thumbnails")
    p.add_argument("--channel", type=str, default=None, help="Fetch for specific channel handle")
    p.add_argument("--resolution", type=str, default="fullres",
                   choices=["fullres", "hires", "lowres"],
                   help="fullres=1920x1080 via yt-dlp (default), hires=1280x720, lowres=480x360")
    p.add_argument("--limit", type=int, default=None, help="Max videos to fetch")
    p.add_argument("--layout", type=str, default="overlay",
                   choices=["overlay", "otb_only", "all"],
                   help="Layout type filter (default: overlay)")
    p.add_argument("--random", action="store_true",
                   help="Randomize video selection (avoids sampling bias)")

    # analyze-video
    p = subparsers.add_parser(
        "analyze-video",
        help="Analyze a local chess video into PGN and an annotated video",
    )
    p.add_argument("video", type=str, help="Path to the video file")
    p.add_argument(
        "--reader", type=str, default="overlay",
        choices=["overlay", "hybrid"],
        help="Board reader backend (default: overlay)",
    )
    p.add_argument(
        "--scene", type=str, default="none",
        choices=["none", "vlm"],
        help="Scene analysis backend (default: none)",
    )
    p.add_argument(
        "--device", type=str, default="mps",
        help="Torch device for overlay piece classification (default: mps)",
    )
    p.add_argument(
        "--vlm-model", type=str,
        default="gemma4_local",
        help="VLM model ID or local alias used by hybrid/scene backends",
    )
    p.add_argument("--fps", type=float, default=2.0, help="Target FPS for frame extraction")
    p.add_argument(
        "--output", type=str, default="outputs/analysis",
        help="Output directory (default: outputs/analysis)",
    )
    p.add_argument("--tts", action="store_true", help="Announce moves via macOS TTS")
    p.add_argument(
        "--no-annotate", action="store_true",
        help="Skip video annotation (just produce PGN)",
    )

    # stats
    subparsers.add_parser("stats", help="Print pipeline statistics")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Dispatch
    commands = {
        "db-init": cmd_db_init,
        "seed-channels": cmd_seed_channels,
        "resolve-channels": cmd_resolve_channels,
        "crawl": cmd_crawl,
        "screen": cmd_screen,
        "inspect": cmd_inspect,
        "download": cmd_download,
        "calibrate": cmd_calibrate,
        "generate-clips": cmd_generate_clips,
        "overlay-test": cmd_overlay_test,
        "overlay-test-reader": cmd_overlay_test_reader,
        "overlay-yolo-export": cmd_overlay_yolo_export,
        "overlay-yolo-train": cmd_overlay_yolo_train,
        "otb-yolo-export": cmd_otb_yolo_export,
        "otb-yolo-train": cmd_otb_yolo_train,
        "inspect-clip": cmd_inspect_clip,
        "ai-extract": cmd_ai_extract,
        "ai-train": cmd_ai_train,
        "ai-eval": cmd_ai_eval,
        "ai-screen": cmd_ai_screen,
        "ai-retrain": cmd_ai_retrain,
        "ai-profile": cmd_ai_profile,
        "auto-calibrate": cmd_auto_calibrate,
        "smoke-test": cmd_smoke_test,
        "inspect-calibration": cmd_inspect_calibration,
        "ai-extract-status": cmd_ai_extract_status,
        "fetch-frames": cmd_fetch_frames,
        "analyze-video": cmd_analyze_video,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
