"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import {
  readOverlayFrame,
  getDetectVideoMovesJobStatus,
  startDetectVideoMovesJob,
  videoFrameUrl,
} from "@/lib/api";
import type {
  VideoClip,
  FrameOverlayResponse,
  VideoMoveDetectionResponse,
  GameSegmentResponse,
  VideoMoveDetectionJobStatus,
  VideoDetectedMove,
} from "@/lib/types";
import { ChessBoard } from "@/components/ChessBoard";
import { fmtTime } from "@/lib/format";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useVideoWorkbench } from "../_context";

/* ── Pulsing board skeleton ─────────────────────────────────── */

function PulsingBoardSkeleton({ size = 200 }: { size?: number }) {
  return (
    <div
      className="rounded border bg-muted animate-pulse"
      style={{ width: size, height: size }}
    />
  );
}

/* ── Move Replay component ──────────────────────────────────── */

function MoveReplay({
  moves,
  sessionId,
  clipId,
}: {
  moves: VideoDetectedMove[];
  sessionId: string;
  clipId: number;
}) {
  const [idx, setIdx] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const move = moves[idx];
  const canPrev = idx > 0;
  const canNext = idx < moves.length - 1;

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" && canPrev) setIdx((i) => i - 1);
      if (e.key === "ArrowRight" && canNext) setIdx((i) => i + 1);
    },
    [canPrev, canNext]
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.focus();
    el.addEventListener("keydown", handleKey);
    return () => el.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  if (!move) return null;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      className="outline-none space-y-2 border rounded-lg p-3 bg-muted/10"
    >
      <div className="flex items-center gap-2 text-xs">
        <button
          onClick={() => setIdx((i) => i - 1)}
          disabled={!canPrev}
          className="px-2 py-1 rounded border text-xs disabled:opacity-30 hover:bg-muted"
        >
          &larr;
        </button>
        <span className="font-medium tabular-nums">
          Move {idx + 1} / {moves.length}
        </span>
        <button
          onClick={() => setIdx((i) => i + 1)}
          disabled={!canNext}
          className="px-2 py-1 rounded border text-xs disabled:opacity-30 hover:bg-muted"
        >
          &rarr;
        </button>
        <span className="text-muted-foreground ml-2 font-mono">
          {move.move_san || move.move_uci}
        </span>
        <span className="text-muted-foreground tabular-nums">
          {move.timestamp_seconds.toFixed(1)}s
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {/* Overlay + OTB frames from the video at this timestamp */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1">Video frame</label>
          <img
            src={videoFrameUrl(
              sessionId,
              Math.round(move.timestamp_seconds * 30) // approx frame
            )}
            alt={`Frame at ${move.timestamp_seconds}s`}
            className="w-full rounded border"
          />
        </div>
        {/* Chess board position */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1">
            Position after {move.move_san || move.move_uci}
          </label>
          <ChessBoard fen={move.fen_after} size={200} />
          <code className="text-[10px] text-muted-foreground break-all block mt-1">
            {move.fen_after}
          </code>
        </div>
        {/* Move info */}
        <div className="space-y-2 text-xs">
          <div>
            <span className="text-muted-foreground">UCI:</span>{" "}
            <span className="font-mono">{move.move_uci}</span>
          </div>
          <div>
            <span className="text-muted-foreground">SAN:</span>{" "}
            <span className="font-medium">{move.move_san}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Time:</span>{" "}
            <span className="tabular-nums">{move.timestamp_seconds.toFixed(1)}s</span>
          </div>
          <div>
            <span className="text-muted-foreground">Move #:</span>{" "}
            <span className="tabular-nums">{move.move_index + 1}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Segment Card ───────────────────────────────────────────── */

function SegmentCard({
  segment,
  expanded,
  onToggle,
  sessionId,
  clipId,
}: {
  segment: GameSegmentResponse;
  expanded: boolean;
  onToggle: () => void;
  sessionId: string;
  clipId: number;
}) {
  return (
    <div className="border rounded-lg">
      <button onClick={onToggle} className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/30 transition-colors">
        <span className="text-xs font-medium">Game {segment.game_index + 1}</span>
        <span className="text-xs text-muted-foreground">{segment.num_moves} moves</span>
        <span className="text-xs text-muted-foreground flex-1 text-right font-mono truncate max-w-xs">
          {segment.pgn_moves.slice(0, 60)}{segment.pgn_moves.length > 60 ? "..." : ""}
        </span>
        <span className="text-xs">{expanded ? "\u25B2" : "\u25BC"}</span>
      </button>

      {expanded && (
        <div className="border-t px-3 py-2 space-y-3">
          {/* Move replay with chess board */}
          {segment.moves.length > 0 && (
            <MoveReplay
              moves={segment.moves}
              sessionId={sessionId}
              clipId={clipId}
            />
          )}

          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">PGN</label>
            <pre className="text-xs bg-muted/30 rounded p-2 whitespace-pre-wrap font-mono max-h-32 overflow-auto">
              {segment.pgn_moves}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Clip Extract Card ──────────────────────────────────────── */

function ClipExtractCard({ clip }: { clip: VideoClip }) {
  const { session } = useVideoWorkbench();
  if (!session) return null;

  const fps = session.fps;
  const startFrame = Math.round(clip.start_time * fps);
  const endFrame = clip.end_time != null ? Math.round(clip.end_time * fps) : session.total_frames - 1;

  const [frameIdx, setFrameIdx] = useState(startFrame);
  const [overlayResult, setOverlayResult] = useState<FrameOverlayResponse | null>(null);
  const [readingFrame, setReadingFrame] = useState(false);
  const [detection, setDetection] = useState<VideoMoveDetectionResponse | null>(null);
  const [detectionJob, setDetectionJob] = useState<VideoMoveDetectionJobStatus | null>(null);
  const [expandedSegment, setExpandedSegment] = useState<number | null>(null);
  const [readerBackend, setReaderBackend] = useState<"overlay" | "hybrid">("overlay");
  const detectingMoves = detectionJob?.status === "running";
  const progressPct = detectionJob && detectionJob.total_samples > 0
    ? Math.round((detectionJob.completed_samples / detectionJob.total_samples) * 100)
    : 0;

  // Auto-read overlay on frame change (debounced)
  const debouncedFrame = useDebouncedValue(frameIdx, 500);

  useEffect(() => {
    let cancelled = false;
    setReadingFrame(true);
    readOverlayFrame(session.session_id, debouncedFrame, clip.id, readerBackend)
      .then((result) => {
        if (!cancelled) setOverlayResult(result);
      })
      .catch((e) => {
        if (!cancelled) toast.error(e instanceof Error ? e.message : "Failed to read overlay");
      })
      .finally(() => {
        if (!cancelled) setReadingFrame(false);
      });
    return () => { cancelled = true; };
  }, [debouncedFrame, session.session_id, clip.id, readerBackend]);

  // Poll detection job
  useEffect(() => {
    if (!detectionJob || detectionJob.status !== "running") return;

    let cancelled = false;
    const pollJob = async () => {
      try {
        const job = await getDetectVideoMovesJobStatus(session.session_id, detectionJob.job_id);
        if (cancelled) return;

        setDetectionJob(job);

        if (job.status === "done" && job.result) {
          setDetection(job.result);
          if (job.result.segments.length > 0) setExpandedSegment(0);
        } else if (job.status === "failed") {
          toast.error(job.error ?? "Move detection failed");
        }
      } catch (e) {
        if (cancelled) return;
        const message = e instanceof Error ? e.message : "Move detection failed";
        setDetectionJob((current) => current ? { ...current, status: "failed", error: message } : current);
        toast.error(message);
      }
    };

    void pollJob();
    const intervalId = window.setInterval(() => { void pollJob(); }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [detectionJob, session.session_id]);

  const handleDetectMoves = async () => {
    setDetection(null);
    setDetectionJob(null);
    setExpandedSegment(null);
    try {
      const job = await startDetectVideoMovesJob(session.session_id, 2.0, clip.id, readerBackend);
      setDetectionJob(job);
      if (job.status === "done" && job.result) {
        setDetection(job.result);
        if (job.result.segments.length > 0) setExpandedSegment(0);
      } else if (job.status === "failed") {
        toast.error(job.error ?? "Move detection failed");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Move detection failed");
    }
  };

  return (
    <div className="border rounded-lg p-3 space-y-3">
      <div className="flex items-center gap-3">
        <span className="text-xs font-medium">{clip.label || `Clip ${clip.clip_index + 1}`}</span>
        <span className="text-[10px] text-muted-foreground">
          {fmtTime(clip.start_time)} &mdash; {clip.end_time != null ? fmtTime(clip.end_time) : "end"}
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <label className="text-xs text-muted-foreground flex flex-col gap-1">
            Reader
            <select
              value={readerBackend}
              onChange={(e) => setReaderBackend(e.target.value as "overlay" | "hybrid")}
              className="h-8 rounded-md border bg-background px-2 text-xs"
            >
              <option value="overlay">overlay</option>
              <option value="hybrid">hybrid (adaptive fallback)</option>
            </select>
          </label>
          <div className="flex-1">
            <div className="text-xs text-muted-foreground mb-1">
              Frame {frameIdx} ({fps > 0 ? (frameIdx / fps).toFixed(1) : 0}s)
            </div>
            <input
              type="range"
              min={startFrame}
              max={endFrame}
              step={Math.max(1, Math.round(fps))}
              value={frameIdx}
              onChange={(e) => setFrameIdx(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Live preview: overlay + OTB board + chess board */}
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Overlay Crop</label>
            {overlayResult?.overlay_crop_b64 ? (
              <img src={`data:image/jpeg;base64,${overlayResult.overlay_crop_b64}`} alt="Overlay" className="w-full rounded border" />
            ) : (
              <div className="w-full aspect-square rounded border bg-muted animate-pulse" />
            )}
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">OTB Board Crop</label>
            {overlayResult?.camera_crop_b64 ? (
              <img src={`data:image/jpeg;base64,${overlayResult.camera_crop_b64}`} alt="OTB Board" className="w-full rounded border" />
            ) : (
              <div className="w-full aspect-square rounded border bg-muted animate-pulse" />
            )}
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">
              Extracted FEN {overlayResult && !overlayResult.fen ? "(failed)" : ""}
            </label>
            {readingFrame && !overlayResult?.fen ? (
              <PulsingBoardSkeleton size={200} />
            ) : overlayResult?.fen ? (
              <div className="space-y-1">
                <ChessBoard fen={overlayResult.fen} size={200} />
                <code className="text-[10px] text-muted-foreground break-all block">{overlayResult.fen}</code>
                {overlayResult.read_method && (
                  <p className="text-[10px] text-muted-foreground">method: {overlayResult.read_method}</p>
                )}
              </div>
            ) : overlayResult ? (
              <p className="text-xs text-muted-foreground">Could not read board</p>
            ) : (
              <PulsingBoardSkeleton size={200} />
            )}
          </div>
        </div>
      </div>

      <hr />

      <div className="space-y-2">
        <button
          onClick={handleDetectMoves}
          disabled={detectingMoves}
          className="px-3 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {detectingMoves ? "Detecting..." : "Run Detection"}
        </button>
        {detectingMoves && (
          <div className="pl-2 space-y-1">
            <div className="text-xs text-muted-foreground">
              Processed {detectionJob.completed_samples} / {detectionJob.total_samples} sampled
              {" "}frames, {detectionJob.num_readable} readable ({progressPct}%).
            </div>
            <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary transition-[width] duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}
        {detection && (
          <div className="space-y-2 pl-2">
            <div className="text-xs text-muted-foreground">
              Sampled {detection.num_frames_sampled} frames, {detection.num_readable} readable.
              Found {detection.segments.length} game(s) using {detection.reader_backend}.
            </div>
            {detection.segments.map((seg) => (
              <SegmentCard
                key={seg.game_index}
                segment={seg}
                expanded={expandedSegment === seg.game_index}
                onToggle={() => setExpandedSegment(expandedSegment === seg.game_index ? null : seg.game_index)}
                sessionId={session.session_id}
                clipId={clip.id}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ExtractPage() {
  const { session, activeClips: clips, downloadStatus } = useVideoWorkbench();

  if (!downloadStatus?.downloaded) {
    return <p className="text-sm text-muted-foreground pt-2">Video must be downloaded first.</p>;
  }

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video session...</p>;
  }

  if (clips.length === 0) {
    return <p className="text-sm text-muted-foreground pt-2">No clips found. Go to the Calibrate step first.</p>;
  }

  return (
    <div className="space-y-4 pt-2 max-w-5xl">
      {clips.map((clip) => (
        <ClipExtractCard key={clip.id} clip={clip} />
      ))}
    </div>
  );
}
