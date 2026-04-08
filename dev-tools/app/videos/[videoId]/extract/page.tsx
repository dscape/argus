"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  readOverlayFrame,
  getDetectVideoMovesJobStatus,
  startDetectVideoMovesJob,
} from "@/lib/api";
import type {
  VideoClip,
  FrameOverlayResponse,
  VideoMoveDetectionResponse,
  GameSegmentResponse,
  VideoMoveDetectionJobStatus,
} from "@/lib/types";
import { ChessBoard } from "@/components/ChessBoard";
import { fmtTime } from "@/lib/format";
import { useVideoWorkbench } from "../_context";

function SegmentCard({
  segment,
  expanded,
  onToggle,
}: {
  segment: GameSegmentResponse;
  expanded: boolean;
  onToggle: () => void;
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
        <div className="border-t px-3 py-2 space-y-2">
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">PGN</label>
            <pre className="text-xs bg-muted/30 rounded p-2 whitespace-pre-wrap font-mono max-h-32 overflow-auto">
              {segment.pgn_moves}
            </pre>
          </div>

          <div className="max-h-64 overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-background">
                <tr className="text-left text-muted-foreground">
                  <th className="px-2 py-1">#</th>
                  <th className="px-2 py-1">UCI</th>
                  <th className="px-2 py-1">SAN</th>
                  <th className="px-2 py-1">Time</th>
                  <th className="px-2 py-1">FEN After</th>
                </tr>
              </thead>
              <tbody>
                {segment.moves.map((m) => (
                  <tr key={m.move_index} className="border-t hover:bg-muted/20">
                    <td className="px-2 py-1 tabular-nums">{m.move_index + 1}</td>
                    <td className="px-2 py-1 font-mono">{m.move_uci}</td>
                    <td className="px-2 py-1 font-medium">{m.move_san}</td>
                    <td className="px-2 py-1 tabular-nums">{m.timestamp_seconds.toFixed(1)}s</td>
                    <td className="px-2 py-1 font-mono text-muted-foreground truncate max-w-[200px]">{m.fen_after}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

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
    const intervalId = window.setInterval(() => {
      void pollJob();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [detectionJob, session.session_id]);

  const handleReadFrame = async () => {
    setReadingFrame(true);
    try {
      const result = await readOverlayFrame(session.session_id, frameIdx, clip.id, readerBackend);
      setOverlayResult(result);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to read overlay");
    } finally {
      setReadingFrame(false);
    }
  };

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
          <button
            onClick={handleReadFrame}
            disabled={readingFrame}
            className="px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {readingFrame ? "Reading..." : "Read Overlay"}
          </button>
        </div>

        {overlayResult && (
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Overlay Crop</label>
              <img src={`data:image/jpeg;base64,${overlayResult.overlay_crop_b64}`} alt="Overlay" className="w-full rounded border" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">OTB Board Crop</label>
              <img src={`data:image/jpeg;base64,${overlayResult.camera_crop_b64}`} alt="OTB Board" className="w-full rounded border" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">
                Extracted FEN {overlayResult.fen ? "" : "(failed)"}
              </label>
              {overlayResult.fen ? (
                <div className="space-y-1">
                  <ChessBoard fen={overlayResult.fen} size={200} />
                  <code className="text-[10px] text-muted-foreground break-all block">{overlayResult.fen}</code>
                  {overlayResult.read_method && (
                    <p className="text-[10px] text-muted-foreground">method: {overlayResult.read_method}</p>
                  )}
                </div>
              ) : (
                <p className="text-xs text-destructive">Could not read board</p>
              )}
            </div>
          </div>
        )}
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
        {detectionJob?.status === "failed" && detectionJob.error && (
          <div className="text-xs text-destructive pl-2">{detectionJob.error}</div>
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
