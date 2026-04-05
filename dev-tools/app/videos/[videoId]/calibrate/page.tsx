"use client";

import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  videoFrameUrl,
  updateVideoClip,
  autoCalibrateClip,
} from "@/lib/api";
import type { VideoClip, AutoCalibrateResponse } from "@/lib/types";
import { BboxDrawer, type Bbox } from "@/components/videos/BboxDrawer";
import { fmtTime } from "@/lib/format";
import { useVideoWorkbench } from "../_context";

export default function CalibratePage() {
  const { video, session, activeClips: clips, setClips, refreshClips, downloadStatus } = useVideoWorkbench();
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const debouncedFrameIdx = useDebouncedValue(frameIdx, 150);
  const [overlayBbox, setOverlayBbox] = useState<Bbox | null>(null);
  const [cameraBbox, setCameraBbox] = useState<Bbox | null>(null);
  const [drawingMode, setDrawingMode] = useState<"overlay" | "camera">("overlay");
  const [boardTheme, setBoardTheme] = useState("lichess_default");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [autoCalClipId, setAutoCalClipId] = useState<number | null>(null);
  const [autoCalAllRunning, setAutoCalAllRunning] = useState(false);
  const [autoCalPreview, setAutoCalPreview] = useState<AutoCalibrateResponse | null>(null);

  const selectClip = useCallback((idx: number) => {
    const clip = clips[idx];
    if (!clip) return;
    setSelectedIdx(idx);
    setOverlayBbox({ x: clip.overlay_bbox[0], y: clip.overlay_bbox[1], w: clip.overlay_bbox[2], h: clip.overlay_bbox[3] });
    setCameraBbox({ x: clip.camera_bbox[0], y: clip.camera_bbox[1], w: clip.camera_bbox[2], h: clip.camera_bbox[3] });
    setBoardTheme(clip.board_theme);
    setAutoCalPreview(null);
    if (session && session.fps > 0) {
      setFrameIdx(Math.round(clip.start_time * session.fps));
    }
  }, [clips, session]);

  useEffect(() => {
    if (clips.length > 0 && selectedIdx === null) selectClip(0);
  }, [clips, selectedIdx, selectClip]);

  const handleSaveClip = async () => {
    if (selectedIdx === null || !overlayBbox || !cameraBbox || !video) return;
    const clip = clips[selectedIdx];
    setSaving(true);
    setError(null);
    try {
      const updated = await updateVideoClip(video.video_id, clip.id, {
        overlay_bbox: [overlayBbox.x, overlayBbox.y, overlayBbox.w, overlayBbox.h],
        camera_bbox: [cameraBbox.x, cameraBbox.y, cameraBbox.w, cameraBbox.h],
        ref_resolution: [session?.width || 1920, session?.height || 1080],
        board_theme: boardTheme,
      } as Partial<VideoClip>);
      const newClips = [...clips];
      newClips[selectedIdx] = updated;
      setClips(newClips);
      setSuccess("Saved");
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleAutoCalibrateSingle = async (clipId: number) => {
    if (!video) return;
    setAutoCalClipId(clipId);
    setError(null);
    try {
      const result = await autoCalibrateClip(video.video_id, clipId);
      setAutoCalPreview(result);
      if (result.proposal) {
        await refreshClips();
        const updated = clips;
        const idx = updated.findIndex((c) => c.id === clipId);
        if (idx >= 0) selectClip(idx);
        setSuccess("Auto-calibrated");
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError("No overlay detected in this clip's time range");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-calibration failed");
    } finally {
      setAutoCalClipId(null);
    }
  };

  const handleAutoCalAll = async () => {
    if (!video) return;
    setAutoCalAllRunning(true);
    setError(null);
    let calibrated = 0;
    for (const clip of clips) {
      try {
        const result = await autoCalibrateClip(video.video_id, clip.id);
        if (result.proposal) calibrated++;
      } catch {
        // continue with next clip
      }
    }
    await refreshClips();
    if (clips.length > 0) selectClip(0);
    setAutoCalAllRunning(false);
    setSuccess(`Auto-calibrated ${calibrated}/${clips.length} clips`);
    setTimeout(() => setSuccess(null), 5000);
  };

  if (!downloadStatus?.downloaded) {
    return <p className="text-sm text-muted-foreground pt-2">Video must be downloaded first. Go to the Download step.</p>;
  }

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video...</p>;
  }

  if (clips.length === 0) {
    return <p className="text-sm text-muted-foreground pt-2">No segments found. Go to the Segment step first to create segments.</p>;
  }

  const totalFrames = session.total_frames;
  const fps = session.fps;
  const duration = session.duration_seconds;
  const frameSrc = videoFrameUrl(session.session_id, debouncedFrameIdx);
  const timestamp = fps > 0 ? (frameIdx / fps).toFixed(1) : "0";
  const selected = selectedIdx !== null ? clips[selectedIdx] : null;

  return (
    <div className="flex gap-4 pt-2" style={{ height: "calc(100vh - 12rem)" }}>
      {/* Left: Clip list */}
      <div className="w-64 flex-shrink-0 flex flex-col gap-2 overflow-auto">
        <button
          onClick={handleAutoCalAll}
          disabled={autoCalAllRunning}
          className="w-full px-3 py-2 rounded-lg border text-xs font-medium bg-green-500/10 border-green-500/30 text-green-700 hover:bg-green-500/20 transition-colors disabled:opacity-50"
        >
          {autoCalAllRunning ? "Calibrating All..." : "Auto-Calibrate All"}
        </button>

        {clips.map((c, i) => (
          <div
            key={c.id}
            onClick={() => selectClip(i)}
            className={`rounded-lg border p-2 cursor-pointer transition-colors ${
              selectedIdx === i ? "border-primary bg-primary/5" : "hover:border-foreground/30"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">{c.label || `Clip ${c.clip_index + 1}`}</span>
              <button
                onClick={(e) => { e.stopPropagation(); handleAutoCalibrateSingle(c.id); }}
                disabled={autoCalClipId === c.id}
                className="text-[10px] text-green-600 hover:text-green-700 disabled:opacity-50"
              >
                {autoCalClipId === c.id ? "..." : "Cal"}
              </button>
            </div>
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {fmtTime(c.start_time)} &mdash; {c.end_time != null ? fmtTime(c.end_time) : "end"}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {c.overlay_bbox[2]}x{c.overlay_bbox[3]} overlay
              {c.camera_bbox[2] > 100 && ` | ${c.camera_bbox[2]}x${c.camera_bbox[3]} cam`}
            </div>
          </div>
        ))}
      </div>

      {/* Right: Calibration editor */}
      <div className="flex-1 space-y-3 overflow-auto">
        {selected === null ? (
          <p className="text-sm text-muted-foreground">Select a clip to calibrate.</p>
        ) : (
          <>
            <div className="flex items-center gap-3 text-xs">
              <span className="font-medium">{selected.label || `Clip ${selected.clip_index + 1}`}</span>
              <span className="text-muted-foreground">
                {fmtTime(selected.start_time)} &mdash; {selected.end_time != null ? fmtTime(selected.end_time) : "end"}
              </span>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Frame {frameIdx} / {totalFrames}</span>
                <span>({timestamp}s)</span>
              </div>
              <input
                type="range"
                min={0}
                max={totalFrames - 1}
                step={Math.max(1, Math.round(fps))}
                value={frameIdx}
                onChange={(e) => setFrameIdx(Number(e.target.value))}
                className="w-full"
              />

              <div className="relative w-full h-3 bg-muted rounded-full overflow-hidden">
                {clips.map((c, i) => {
                  const startPct = duration > 0 ? (c.start_time / duration) * 100 : 0;
                  const endTime = c.end_time ?? duration;
                  const widthPct = duration > 0 ? ((endTime - c.start_time) / duration) * 100 : 0;
                  return (
                    <div
                      key={c.id}
                      onClick={() => selectClip(i)}
                      title={`${c.label || `Clip ${c.clip_index + 1}`}: ${fmtTime(c.start_time)} \u2014 ${c.end_time != null ? fmtTime(c.end_time) : "end"}`}
                      className={`absolute top-0 h-full cursor-pointer transition-opacity ${
                        selectedIdx === i ? "bg-primary opacity-60" : "bg-primary/30 hover:opacity-50"
                      }`}
                      style={{ left: `${startPct}%`, width: `${widthPct}%` }}
                    />
                  );
                })}
                <div
                  className="absolute top-0 w-0.5 h-full bg-foreground"
                  style={{ left: `${totalFrames > 1 ? (frameIdx / (totalFrames - 1)) * 100 : 0}%` }}
                />
              </div>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => setDrawingMode("overlay")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                  drawingMode === "overlay" ? "bg-green-500/10 border-green-500 text-green-700" : "border-muted text-muted-foreground hover:border-foreground"
                }`}
              >
                2D Overlay (green)
              </button>
              <button
                onClick={() => setDrawingMode("camera")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                  drawingMode === "camera" ? "bg-blue-500/10 border-blue-500 text-blue-700" : "border-muted text-muted-foreground hover:border-foreground"
                }`}
              >
                OTB Board (blue)
              </button>
              {overlayBbox && <span className="text-xs text-green-600">Overlay: {overlayBbox.x},{overlayBbox.y} {overlayBbox.w}x{overlayBbox.h}</span>}
              {cameraBbox && <span className="text-xs text-blue-600">OTB Board: {cameraBbox.x},{cameraBbox.y} {cameraBbox.w}x{cameraBbox.h}</span>}
            </div>

            <BboxDrawer
              imageSrc={frameSrc}
              onBboxChange={(bbox) => {
                if (drawingMode === "overlay") setOverlayBbox(bbox);
                else setCameraBbox(bbox);
              }}
              existingBbox={drawingMode === "overlay" ? overlayBbox : cameraBbox}
              secondBbox={drawingMode === "overlay" ? cameraBbox : overlayBbox}
              bboxColor={drawingMode === "overlay" ? "#22c55e" : "#3b82f6"}
              secondBboxColor={drawingMode === "overlay" ? "#3b82f6" : "#22c55e"}
            />

            {autoCalPreview?.preview_frame_b64 && (
              <div className="space-y-3 border rounded-lg p-3">
                <h4 className="text-xs font-medium text-muted-foreground">Auto-Calibration Preview</h4>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Green = Overlay, Red = Camera</p>
                  <img
                    src={`data:image/jpeg;base64,${autoCalPreview.preview_frame_b64}`}
                    alt="Proposal"
                    className="w-full max-w-lg rounded border"
                  />
                </div>
                {autoCalPreview.camera_heatmap_b64 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Camera Motion Heatmap</p>
                    <img
                      src={`data:image/jpeg;base64,${autoCalPreview.camera_heatmap_b64}`}
                      alt="Heatmap"
                      className="w-full max-w-lg rounded border"
                    />
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center gap-2">
              <button
                onClick={handleSaveClip}
                disabled={saving || !overlayBbox || !cameraBbox}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save"}
              </button>
              <button
                onClick={() => handleAutoCalibrateSingle(selected.id)}
                disabled={autoCalClipId !== null}
                className="px-4 py-2 rounded-lg border text-sm font-medium bg-green-500/10 border-green-500/30 text-green-700 hover:bg-green-500/20 disabled:opacity-50"
              >
                {autoCalClipId === selected.id ? "Calibrating..." : "Auto-Calibrate This Clip"}
              </button>
              {error && <span className="text-xs text-destructive">{error}</span>}
              {success && <span className="text-xs text-green-600">{success}</span>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
