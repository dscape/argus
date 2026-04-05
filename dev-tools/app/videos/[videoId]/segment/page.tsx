"use client";

import { useState, useCallback } from "react";
import { toast } from "sonner";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  videoFrameUrl,
  createVideoClip,
  updateVideoClip,
  deleteVideoClip,
  autoSegmentVideo,
} from "@/lib/api";
import type { VideoClip, AutoSegmentResponse } from "@/lib/types";
import { fmtTime } from "@/lib/format";
import { SegmentTimeline, type DragState } from "@/components/videos/SegmentTimeline";
import { useVideoWorkbench } from "../_context";

export default function SegmentPage() {
  const { video, session, clips, setClips, refreshClips, downloadStatus } = useVideoWorkbench();

  const [frameIdx, setFrameIdx] = useState(0);
  const debouncedFrameIdx = useDebouncedValue(frameIdx, 150);
  const [selectedClipIds, setSelectedClipIds] = useState<Set<number>>(new Set());
  const [dragging, setDragging] = useState<DragState | null>(null);
  const [segmenting, setSegmenting] = useState(false);
  const [segmentResult, setSegmentResult] = useState<AutoSegmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── Handlers ──────────────────────────────────────────────

  const handleAutoSegment = async (replace: boolean) => {
    if (!video) return;
    setSegmenting(true);
    setError(null);
    try {
      const result = await autoSegmentVideo(video.video_id, { replaceExisting: replace });
      setSegmentResult(result);
      if (result.error) setError(result.error);
      await refreshClips();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Segmentation failed");
    } finally {
      setSegmenting(false);
    }
  };

  const handleAddClip = async () => {
    if (!session || !video) return;
    setError(null);
    const fps = session.fps;
    const currentTime = fps > 0 ? frameIdx / fps : 0;
    try {
      const newClip = await createVideoClip(video.video_id, {
        start_time: currentTime,
        end_time: session.duration_seconds,
        label: null,
        overlay_bbox: [0, 0, 100, 100],
        camera_bbox: [0, 0, 100, 100],
        ref_resolution: [session.width, session.height],
        board_flipped: false,
        board_theme: "lichess_default",
        is_gap: false,
      });
      setClips((prev) => [...prev, newClip]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add segment");
    }
  };

  const handleDeleteClips = async (ids: Set<number>) => {
    if (!video) return;
    setError(null);
    try {
      for (const id of ids) {
        await deleteVideoClip(video.video_id, id);
      }
      await refreshClips();
      setSelectedClipIds(new Set());
      toast.success(`Deleted ${ids.size} segment(s)`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete segment");
    }
  };

  const handleSplit = async () => {
    if (!session || !video) return;
    const selectedArr = clips.filter((c) => selectedClipIds.has(c.id));
    if (selectedArr.length !== 1) {
      toast.error("Select exactly one segment to split");
      return;
    }
    const clip = selectedArr[0];
    const fps = session.fps;
    const splitTime = fps > 0 ? frameIdx / fps : 0;

    if (splitTime <= clip.start_time || (clip.end_time && splitTime >= clip.end_time)) {
      toast.error("Playhead must be within the segment to split");
      return;
    }

    setError(null);
    try {
      await updateVideoClip(video.video_id, clip.id, { end_time: splitTime } as Partial<VideoClip>);
      await createVideoClip(video.video_id, {
        start_time: splitTime,
        end_time: clip.end_time ?? session.duration_seconds,
        label: null,
        overlay_bbox: clip.overlay_bbox as [number, number, number, number],
        camera_bbox: clip.camera_bbox as [number, number, number, number],
        ref_resolution: clip.ref_resolution as [number, number],
        board_flipped: clip.board_flipped,
        board_theme: clip.board_theme,
        is_gap: clip.is_gap,
      });
      await refreshClips();
      setSelectedClipIds(new Set());
      toast.success("Segment split");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to split segment");
    }
  };

  const handleMerge = async () => {
    if (!video) return;
    const selected = clips
      .filter((c) => selectedClipIds.has(c.id))
      .sort((a, b) => a.start_time - b.start_time);

    if (selected.length < 2) {
      toast.error("Select at least 2 segments to merge");
      return;
    }

    // Check adjacency (allow up to 1s gap between segments)
    for (let i = 1; i < selected.length; i++) {
      const prevEnd = selected[i - 1].end_time ?? session?.duration_seconds ?? 0;
      if (selected[i].start_time - prevEnd > 1.0) {
        toast.error("Selected segments must be adjacent to merge");
        return;
      }
    }

    setError(null);
    try {
      const keeper = selected[0];
      const lastEnd = selected[selected.length - 1].end_time;

      // Delete others in reverse index order to avoid unique constraint violations during reindex
      const toDelete = selected.slice(1).sort((a, b) => b.clip_index - a.clip_index);
      for (const clip of toDelete) {
        await deleteVideoClip(video.video_id, clip.id);
      }
      await updateVideoClip(video.video_id, keeper.id, { end_time: lastEnd } as Partial<VideoClip>);
      await refreshClips();
      setSelectedClipIds(new Set());
      toast.success(`Merged ${selected.length} segments`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to merge segments");
    }
  };

  const handleToggleGap = async () => {
    if (!video) return;
    const selected = clips.filter((c) => selectedClipIds.has(c.id));
    if (selected.length === 0) return;

    setError(null);
    try {
      for (const clip of selected) {
        await updateVideoClip(video.video_id, clip.id, { is_gap: !clip.is_gap } as Partial<VideoClip>);
      }
      await refreshClips();
      toast.success(`Toggled gap on ${selected.length} segment(s)`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to toggle gap");
    }
  };

  // ── Timeline event handlers ───────────────────────────────

  const handleSelectClip = useCallback((clipId: number) => {
    setSelectedClipIds((prev) => {
      const next = new Set(prev);
      if (next.has(clipId)) {
        next.delete(clipId);
      } else {
        next.add(clipId);
      }
      return next;
    });
  }, []);

  const handleDragStart = useCallback((clipId: number, edge: "start" | "end") => {
    const clip = clips.find((c) => c.id === clipId);
    if (!clip) return;
    const time = edge === "start" ? clip.start_time : (clip.end_time ?? session?.duration_seconds ?? 0);
    setDragging({ clipId, edge, originalTime: time, currentTime: time });
  }, [clips, session?.duration_seconds]);

  const handleDragMove = useCallback((time: number) => {
    setDragging((prev) => prev ? { ...prev, currentTime: Math.max(0, time) } : null);
  }, []);

  const handleDragEnd = useCallback(async () => {
    if (!dragging || !video) {
      setDragging(null);
      return;
    }

    const { clipId, edge, originalTime, currentTime } = dragging;
    setDragging(null);

    // Skip if barely moved
    if (Math.abs(currentTime - originalTime) < 0.5) return;

    try {
      const update = edge === "start"
        ? { start_time: currentTime }
        : { end_time: currentTime };
      await updateVideoClip(video.video_id, clipId, update as Partial<VideoClip>);
      await refreshClips();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to update boundary");
    }
  }, [dragging, video, refreshClips]);

  // ── Rendering ─────────────────────────────────────────────

  if (!downloadStatus?.downloaded) {
    return <p className="text-sm text-muted-foreground pt-2">Video must be downloaded first. Go to the Download step.</p>;
  }

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video...</p>;
  }

  const totalFrames = session.total_frames;
  const fps = session.fps;
  const duration = session.duration_seconds;
  const frameSrc = videoFrameUrl(session.session_id, debouncedFrameIdx);
  const timestamp = fps > 0 ? (frameIdx / fps).toFixed(1) : "0";

  const selectedArr = clips.filter((c) => selectedClipIds.has(c.id));
  const singleSelected = selectedArr.length === 1 ? selectedArr[0] : null;
  const canMerge = selectedArr.length >= 2;
  const allGaps = selectedArr.length > 0 && selectedArr.every((c) => c.is_gap);

  return (
    <div className="space-y-4 pt-2">
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        {selectedArr.length === 0 ? (
          <>
            <button
              onClick={() => handleAutoSegment(clips.length > 0)}
              disabled={segmenting}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
            >
              {segmenting ? "Segmenting..." : clips.length > 0 ? "Re-Segment (replaces existing)" : "Auto-Segment"}
            </button>
            <button
              onClick={handleAddClip}
              className="px-3 py-2 rounded-lg border border-dashed text-xs font-medium text-muted-foreground hover:border-primary hover:text-foreground transition-colors"
            >
              + Add Segment
            </button>
            {segmenting && (
              <span className="text-xs text-muted-foreground animate-pulse">
                Sampling frames and detecting layouts...
              </span>
            )}
          </>
        ) : (
          <>
            {singleSelected && (
              <button
                onClick={handleSplit}
                className="px-3 py-2 rounded-lg border text-xs font-medium hover:bg-muted transition-colors"
              >
                Split at Playhead
              </button>
            )}
            {canMerge && (
              <button
                onClick={handleMerge}
                className="px-3 py-2 rounded-lg border text-xs font-medium hover:bg-muted transition-colors"
              >
                Merge ({selectedArr.length})
              </button>
            )}
            <button
              onClick={handleToggleGap}
              className="px-3 py-2 rounded-lg border text-xs font-medium hover:bg-muted transition-colors"
            >
              {allGaps ? "Mark as Active" : "Mark as Gap"}
            </button>
            <button
              onClick={() => handleDeleteClips(selectedClipIds)}
              className="px-3 py-2 rounded-lg border border-destructive/50 text-xs font-medium text-destructive hover:bg-destructive/10 transition-colors"
            >
              Delete ({selectedArr.length})
            </button>
            <button
              onClick={() => setSelectedClipIds(new Set())}
              className="px-3 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Deselect
            </button>
            <span className="text-xs text-muted-foreground">
              {selectedArr.length} selected
            </span>
          </>
        )}
      </div>

      {/* Result summary */}
      {segmentResult && !segmentResult.error && (
        <div className="text-xs text-muted-foreground">
          Found {segmentResult.segments.length} segment(s)
          {segmentResult.gaps.length > 0 && `, ${segmentResult.gaps.length} gap(s)`}
          {" "}in {segmentResult.processing_time_sec}s
          ({segmentResult.total_frames_sampled} frames sampled)
        </div>
      )}

      {/* Help text */}
      {clips.length > 0 && (
        <p className="text-[11px] text-muted-foreground">
          Click timeline to preview frame. Drag segment edges to resize. Use the legend below to select segments for merge, split, or delete.
        </p>
      )}

      {/* Timeline */}
      {clips.length > 0 && (
        <SegmentTimeline
          clips={clips}
          duration={duration}
          frameIdx={frameIdx}
          totalFrames={totalFrames}
          fps={fps}
          selectedClipIds={selectedClipIds}
          dragging={dragging}
          onFrameChange={setFrameIdx}
          onSelectClip={handleSelectClip}
          onDragStart={handleDragStart}
          onDragMove={handleDragMove}
          onDragEnd={handleDragEnd}
        />
      )}

      {/* Frame info + preview */}
      {clips.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Frame {frameIdx} / {totalFrames}</span>
            <span>({timestamp}s)</span>
          </div>
          <img src={frameSrc} alt={`Frame ${frameIdx}`} className="w-full max-h-[40vh] object-contain rounded border" />
        </div>
      )}

      {clips.length === 0 && !segmenting && (
        <p className="text-sm text-muted-foreground">
          No segments yet. Click &ldquo;Auto-Segment&rdquo; to detect layout regions, or add one manually.
        </p>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
