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


def cmd_import_players(args):
    """Import players from players.zip."""
    from pipeline.importers.player_importer import import_players
    import_players()


def cmd_import_pgns(args):
    """Import PGN games from pgns.zip."""
    from pipeline.importers.pgn_importer import import_pgns
    import_pgns(limit=args.limit)


def cmd_seed_channels(args):
    """Seed channel config from YAML."""
    from pipeline.importers.channel_seeder import seed_channels
    seed_channels()


def cmd_resolve_channels(args):
    """Resolve channel handles to IDs."""
    from pipeline.crawl.channel_resolver import resolve_channels
    resolve_channels()


def cmd_crawl(args):
    """Crawl YouTube channels for video metadata."""
    from pipeline.crawl.crawl_videos import crawl_all
    crawl_all(channel_handle=args.channel, refresh=args.refresh)


def cmd_extract(args):
    """Extract metadata from video titles and descriptions."""
    from pipeline.extract.extract_metadata import extract_all
    extract_all(use_claude=args.claude)


def cmd_match(args):
    """Match games to videos."""
    from pipeline.match.match_pipeline import match_all
    match_all(min_confidence=args.min_confidence)


def cmd_download(args):
    """Download matched videos."""
    from pipeline.download.video_downloader import download_matched_videos
    download_matched_videos(
        min_confidence=args.min_confidence,
        limit=args.limit,
    )


def cmd_generate_clips(args):
    """Generate training clips from downloaded videos."""
    from pipeline.clips.clip_generator import generate_all_clips
    generate_all_clips(
        min_confidence=args.min_confidence,
        limit=args.limit,
    )


def cmd_overlay_scan(args):
    """Screen crawled videos for overlay presence."""
    from pipeline.overlay.scanner import scan_crawled_videos
    scan_crawled_videos(channel_handle=args.channel, limit=args.limit)


def cmd_overlay_calibrate(args):
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

    cal = LayoutCalibration(
        overlay=overlay,
        camera=camera,
        ref_resolution=ref,
        board_flipped=args.flipped,
        board_theme=args.theme or "lichess_default",
    )

    set_calibration(args.channel, cal)
    print(f"Calibration saved for {args.channel}")
    print(f"  Overlay: {overlay}")
    print(f"  Camera:  {camera}")
    print(f"  Ref resolution: {ref}")


def cmd_overlay_generate(args):
    """Generate training clips from overlay videos."""
    from pipeline.overlay.overlay_clip_generator import generate_from_video

    if args.video:
        results = generate_from_video(
            args.video,
            channel_handle=args.channel,
        )
        if results:
            print(f"Generated {len(results)} clip(s):")
            for r in results:
                print(f"  {r['filepath']} ({r['num_moves']} moves, {r['num_frames']} frames)")
        else:
            print("No clips generated.")
    else:
        print("--video is required. Provide a local file path or YouTube URL.")


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
                "fide_players", "player_aliases", "games",
                "crawl_channels", "youtube_videos", "video_chapters",
                "game_video_links", "training_clips", "api_quota_log",
            ]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cur.fetchone()[0]
                except Exception:
                    stats[table] = "N/A"
                    conn.rollback()

            # Additional stats
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM crawl_channels WHERE channel_id NOT LIKE 'UNRESOLVED:%%'"
                )
                stats["resolved_channels"] = cur.fetchone()[0]
            except Exception:
                stats["resolved_channels"] = "N/A"
                conn.rollback()

            try:
                cur.execute(
                    "SELECT COUNT(*) FROM game_video_links WHERE match_confidence >= 80"
                )
                stats["high_conf_matches"] = cur.fetchone()[0]
            except Exception:
                stats["high_conf_matches"] = "N/A"
                conn.rollback()

            try:
                cur.execute(
                    "SELECT COUNT(*) FROM youtube_videos WHERE white_fide_id IS NOT NULL"
                )
                stats["videos_with_fide"] = cur.fetchone()[0]
            except Exception:
                stats["videos_with_fide"] = "N/A"
                conn.rollback()

    print("\n" + "=" * 50)
    print("  ARGUS PIPELINE STATISTICS")
    print("=" * 50)
    print(f"  FIDE players:           {stats['fide_players']}")
    print(f"  Player aliases:         {stats['player_aliases']}")
    print(f"  Games (PGN):            {stats['games']}")
    print(f"  Crawl channels:         {stats['crawl_channels']}")
    print(f"    Resolved:             {stats['resolved_channels']}")
    print(f"  YouTube videos:         {stats['youtube_videos']}")
    print(f"    With FIDE IDs:        {stats['videos_with_fide']}")
    print(f"  Video chapters:         {stats['video_chapters']}")
    print(f"  Game-video links:       {stats['game_video_links']}")
    print(f"    High confidence:      {stats['high_conf_matches']}")
    print(f"  Training clips:         {stats['training_clips']}")
    print(f"  API quota log entries:  {stats['api_quota_log']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Argus data pipeline: YouTube crawl, extraction, matching, clips",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # db-init
    subparsers.add_parser("db-init", help="Apply database schema")

    # import-players
    subparsers.add_parser("import-players", help="Import players from players.zip")

    # import-pgns
    p = subparsers.add_parser("import-pgns", help="Import PGN games")
    p.add_argument("--limit", type=int, default=None, help="Max games to import")

    # seed-channels
    subparsers.add_parser("seed-channels", help="Seed channels from YAML config")

    # resolve-channels
    subparsers.add_parser("resolve-channels", help="Resolve channel handles to IDs")

    # crawl
    p = subparsers.add_parser("crawl", help="Crawl YouTube channels")
    p.add_argument("--channel", type=str, default=None, help="Crawl specific channel handle")
    p.add_argument("--refresh", action="store_true", help="Only fetch new videos")

    # extract
    p = subparsers.add_parser("extract", help="Extract metadata from titles")
    p.add_argument("--claude", action="store_true", help="Use Claude API for low-confidence titles")

    # match
    p = subparsers.add_parser("match", help="Match games to videos")
    p.add_argument("--min-confidence", type=float, default=60.0, help="Minimum confidence threshold")

    # download
    p = subparsers.add_parser("download", help="Download matched videos")
    p.add_argument("--min-confidence", type=float, default=70.0, help="Minimum confidence threshold")
    p.add_argument("--limit", type=int, default=None, help="Max videos to download")

    # generate-clips
    p = subparsers.add_parser("generate-clips", help="Generate training clips")
    p.add_argument("--min-confidence", type=float, default=70.0, help="Minimum match confidence")
    p.add_argument("--limit", type=int, default=None, help="Max clips to generate")

    # overlay-scan
    p = subparsers.add_parser("overlay-scan", help="Screen videos for 2D overlay presence")
    p.add_argument("--channel", type=str, default=None, help="Scan specific channel handle")
    p.add_argument("--limit", type=int, default=None, help="Max videos to scan")

    # overlay-calibrate
    p = subparsers.add_parser("overlay-calibrate", help="Set overlay layout calibration")
    p.add_argument("--channel", type=str, required=True, help="Channel handle (e.g. @STLChessClub)")
    p.add_argument("--overlay", type=str, required=True, help="Overlay bbox: x,y,w,h")
    p.add_argument("--camera", type=str, required=True, help="Camera bbox: x,y,w,h")
    p.add_argument("--resolution", type=str, default=None, help="Reference resolution: WxH (default: 1920x1080)")
    p.add_argument("--flipped", action="store_true", help="Board is flipped (Black at bottom)")
    p.add_argument("--theme", type=str, default=None, help="Board theme (lichess_default, chess_com_green)")

    # overlay-generate
    p = subparsers.add_parser("overlay-generate", help="Generate clips from overlay videos")
    p.add_argument("--video", type=str, default=None, help="Video file path or YouTube URL")
    p.add_argument("--channel", type=str, required=True, help="Channel handle for calibration lookup")
    p.add_argument("--limit", type=int, default=None, help="Max clips to generate")

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
        "import-players": cmd_import_players,
        "import-pgns": cmd_import_pgns,
        "seed-channels": cmd_seed_channels,
        "resolve-channels": cmd_resolve_channels,
        "crawl": cmd_crawl,
        "extract": cmd_extract,
        "match": cmd_match,
        "download": cmd_download,
        "generate-clips": cmd_generate_clips,
        "overlay-scan": cmd_overlay_scan,
        "overlay-calibrate": cmd_overlay_calibrate,
        "overlay-generate": cmd_overlay_generate,
        "overlay-test": cmd_overlay_test,
        "overlay-test-reader": cmd_overlay_test_reader,
        "inspect-clip": cmd_inspect_clip,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
