"""CLI entry point for the Argus data pipeline."""

import argparse
import logging
import sys

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
    from pipeline.download.video_downloader import get_video_path
    from pipeline.overlay.overlay_clip_generator import generate_from_video
    from pipeline.db.connection import get_conn

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
        video_path = get_video_path(video_id, channel_handle)
        if video_path is None:
            print(f"  Skipping {video_id}: not downloaded")
            continue

        results = generate_from_video(video_path, channel_handle=channel_handle)
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
        screen_video,
        overlay_bbox_to_json,
    )

    # Build query for target videos
    if args.video_id:
        # Single video
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT video_id, channel_handle, title FROM youtube_videos WHERE video_id = %s",
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

                status_filter = args.status or "candidate"
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

            overlay_str = f"overlay={result.overlay_score:.2f}" if result.has_overlay else "no overlay"
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


def cmd_inspect_clip(args):
    """Inspect a .pt training clip file."""
    from pipeline.overlay.diagnostics import inspect_clip

    inspect_clip(
        clip_path=args.file,
        save_frames=args.save_frames,
        output_dir=args.output_dir,
    )


def cmd_stats(args):
    """Print pipeline statistics."""
    from pipeline.db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            stats = {}

            for table in [
                "crawl_channels", "youtube_videos",
                "training_clips", "api_quota_log",
            ]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
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
            for status in ["candidate", "approved", "rejected"]:
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
    print(f"    Candidates:           {stats['screening_candidate']}")
    print(f"    Approved:             {stats['screening_approved']}")
    print(f"    Rejected:             {stats['screening_rejected']}")
    print(f"  Training clips:         {stats['training_clips']}")
    print(f"  API quota log entries:  {stats['api_quota_log']}")
    print("=" * 50)


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
    p = subparsers.add_parser("inspect", help="Inspect videos for overlay + OTB via frame extraction")
    p.add_argument("--video-id", type=str, default=None, help="Inspect a single video by ID")
    p.add_argument("--channel", type=str, default=None, help="Filter by channel handle")
    p.add_argument("--status", type=str, default=None, help="Filter by status (default: candidate)")
    p.add_argument("--limit", type=int, default=None, help="Max videos to inspect")

    # download
    p = subparsers.add_parser("download", help="Download approved videos")
    p.add_argument("--limit", type=int, default=None, help="Max videos to download")

    # calibrate
    p = subparsers.add_parser("calibrate", help="Set overlay layout calibration")
    p.add_argument("--channel", type=str, required=True, help="Channel handle (e.g. @STLChessClub)")
    p.add_argument("--overlay", type=str, required=True, help="Overlay bbox: x,y,w,h")
    p.add_argument("--camera", type=str, required=True, help="Camera bbox: x,y,w,h")
    p.add_argument("--resolution", type=str, default=None, help="Reference resolution: WxH (default: 1920x1080)")
    p.add_argument("--flipped", action="store_true", help="Board is flipped (Black at bottom)")
    p.add_argument("--theme", type=str, default=None, help="Board theme (lichess_default, chess_com_green)")
    p.add_argument("--delay", type=float, default=None, help="Move delay in seconds (default: 2.0)")

    # generate-clips
    p = subparsers.add_parser("generate-clips", help="Generate training clips from approved videos")
    p.add_argument("--channel", type=str, default=None, help="Generate for specific channel")
    p.add_argument("--limit", type=int, default=None, help="Max videos to process")

    # overlay-test
    p = subparsers.add_parser("overlay-test", help="Test overlay pipeline on a screenshot")
    p.add_argument("--image", type=str, required=True, help="Path to screenshot image")
    p.add_argument("--overlay", type=str, default=None, help="Manual overlay bbox: x,y,w,h (skip auto-detect)")
    p.add_argument("--flipped", action="store_true", help="Board is flipped (Black at bottom)")
    p.add_argument("--theme", type=str, default=None, help="Board theme (lichess_default, chess_com_green)")
    p.add_argument("--output", type=str, default=None, help="Output path for annotated image")

    # overlay-test-reader
    p = subparsers.add_parser("overlay-test-reader", help="Test overlay reader on a specific region")
    p.add_argument("--image", type=str, required=True, help="Path to screenshot image")
    p.add_argument("--overlay", type=str, required=True, help="Overlay bbox: x,y,w,h")
    p.add_argument("--flipped", action="store_true", help="Board is flipped (Black at bottom)")
    p.add_argument("--theme", type=str, default=None, help="Board theme")

    # inspect-clip
    p = subparsers.add_parser("inspect-clip", help="Inspect a .pt training clip file")
    p.add_argument("--file", type=str, required=True, help="Path to .pt clip file")
    p.add_argument("--save-frames", action="store_true", help="Save individual frames as images")
    p.add_argument("--output-dir", type=str, default=None, help="Directory for saved frames")

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
        "inspect-clip": cmd_inspect_clip,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
