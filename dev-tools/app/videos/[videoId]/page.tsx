"use client";

import { useParams, useRouter } from "next/navigation";
import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  getVideo,
  getDownloadStatus,
  downloadVideo,
  openVideo,
  videoFrameUrl,
  readOverlayFrame,
  detectVideoMoves,
  generateClips,
  listCalibrations,
  saveCalibration,
  deleteVideoSession,
  loadClip,
  getClipInfo,
  clipFrameUrl,
  listVideoClips,
  createVideoClip,
  updateVideoClip,
  deleteVideoClip,
  inspectAutoCalibration,
  updateVideoStatus,
} from "@/lib/api";
import type {
  CrawlVideo,
  DownloadStatus,
  VideoSession,
  FrameOverlayResponse,
  VideoMoveDetectionResponse,
  GameSegmentResponse,
  CalibrationEntry,
  VideoClip,
  GeneratedClip,
  ClipInspectResponse,
} from "@/lib/types";
import { BboxDrawer, type Bbox } from "@/components/BboxDrawer";
import { ChessBoard } from "@/components/ChessBoard";
import { statusBadge, scoreColor, youtubeThumb, StatusDropdown } from "@/components/video-shared";
import type { VideoWithReason } from "@/components/video-shared";
import VideoCard, { type InspectResult } from "@/components/VideoCard";

// ── Step definitions ────────────────────────────────────────

const STEPS = [
  { id: "info", label: "Info" },
  { id: "download", label: "Download" },
  { id: "calibrate", label: "Calibrate" },
  { id: "extract", label: "Extract" },
  { id: "generate", label: "Generate" },
  { id: "inspect", label: "Inspect" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

// ── Page ────────────────────────────────────────────────────

export default function VideoWorkbenchPage() {
  const params = useParams();
  const router = useRouter();
  const videoId = params.videoId as string;

  const goBack = () => {
    try {
      const saved = sessionStorage.getItem("videoBrowserState");
      if (saved) {
        const s = JSON.parse(saved);
        const params = new URLSearchParams();
        if (s.status && s.status !== "approved:!otb_only") params.set("status", s.status);
        if (s.sort && s.sort !== "published_at_desc") params.set("sort", s.sort);
        if (s.channel) params.set("channel", s.channel);
        if (s.page > 0) params.set("page", String(s.page));
        const qs = params.toString();
        router.push(`/videos${qs ? `?${qs}` : ""}`);
        return;
      }
    } catch { /* ignore */ }
    router.push("/videos");
  };

  const [video, setVideo] = useState<CrawlVideo | null>(null);
  const [step, setStep] = useState<StepId>("info");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getVideo(videoId)
      .then(setVideo)
      .catch((e) => setError(e.message));
  }, [videoId]);

  const handleStatusChange = async (vid: string, status: string | null, layoutType?: string) => {
    await updateVideoStatus(vid, status, layoutType);
    setVideo((prev) => prev ? { ...prev, screening_status: status, layout_type: layoutType ?? null } : prev);
  };

  if (error) {
    return (
      <div className="p-6">
        <button onClick={() => goBack()} className="text-sm text-primary hover:underline mb-4 block">
          &larr; Back to Videos
        </button>
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  if (!video) {
    return <div className="p-6 text-sm text-muted-foreground">Loading...</div>;
  }

  return (
    <div className="h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-4 pt-3 pb-2">
        <button onClick={() => goBack()} className="text-xs text-muted-foreground hover:text-foreground mb-1 block">
          &larr; Videos
        </button>
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-lg font-semibold line-clamp-1 flex-1">{video.title}</h2>
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${scoreColor(video.title_score)}`}
          />
          <StatusDropdown video={video as VideoWithReason} onStatusChange={handleStatusChange} />
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-1 rounded-2xl border bg-muted/30 p-1">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setStep(s.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-150 ${
                step === s.id
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-background/60"
              }`}
            >
              <span className="w-4 h-4 rounded-full border text-[10px] flex items-center justify-center font-bold tabular-nums">
                {i + 1}
              </span>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Step content */}
      <div className="flex-1 overflow-auto px-4 pb-4">
        {step === "info" && <InfoStep video={video} />}
        {step === "download" && <DownloadStep video={video} onDownloaded={() => setStep("calibrate")} />}
        {step === "calibrate" && <CalibrateStep video={video} />}
        {step === "extract" && <ExtractStep video={video} />}
        {step === "generate" && <GenerateStep video={video} />}
        {step === "inspect" && <InspectStep video={video} />}
      </div>
    </div>
  );
}

// ── Step 1: Info ────────────────────────────────────────────

function InfoStep({ video }: { video: CrawlVideo }) {
  const [dlStatus, setDlStatus] = useState<DownloadStatus | null>(null);
  const [aiResult, setAiResult] = useState<InspectResult | null>(null);
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    getDownloadStatus(video.video_id).then(setDlStatus);
  }, [video.video_id]);

  const fmtDuration = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  async function inspectAi() {
    setAiLoading(true);
    try {
      const res = await fetch("/api/models/ai-screening/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: video.video_id }),
      });
      if (!res.ok) throw new Error(await res.text());
      setAiResult(await res.json());
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "AI inspection failed");
    } finally {
      setAiLoading(false);
    }
  }

  return (
    <div className="space-y-4 pt-2 max-w-4xl">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Field label="Video ID" value={video.video_id} />
          <Field label="Channel" value={video.channel_handle || "—"} />
          <Field label="Published" value={video.published_at ? new Date(video.published_at).toLocaleDateString() : "—"} />
          <Field label="Duration" value={dlStatus?.duration_seconds != null ? fmtDuration(dlStatus.duration_seconds) : "—"} />
          <Field label="File Size" value={dlStatus?.file_size_mb != null ? `${dlStatus.file_size_mb} MB` : "—"} />
          <Field label="Status" value={video.screening_status || "unscreened"} />
          <Field label="Layout" value={video.layout_type || "—"} />
          <Field label="Title Score" value={String(video.title_score)} />
          <div>
            <a
              href={`https://www.youtube.com/watch?v=${video.video_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:underline"
            >
              Open on YouTube &rarr;
            </a>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-1">
          {[0, 1, 2, 3].map((i) => (
            <img
              key={i}
              src={youtubeThumb(video.video_id, i)}
              alt={`Thumbnail ${i}`}
              className="w-full aspect-video object-cover rounded border"
            />
          ))}
        </div>
      </div>

      {/* AI Screening */}
      <div className="border-t pt-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium">Screening</h3>
          <button
            onClick={inspectAi}
            disabled={aiLoading}
            className="px-3 py-1 bg-foreground text-background rounded text-xs disabled:opacity-50"
          >
            {aiLoading ? "Running..." : aiResult ? "Re-run" : "Run AI Screen"}
          </button>
        </div>
        {aiResult && <VideoCard result={aiResult} />}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs text-muted-foreground w-24 flex-shrink-0">{label}</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}

// ── Step 2: Download ────────────────────────────────────────

function DownloadStep({ video, onDownloaded }: { video: CrawlVideo; onDownloaded: () => void }) {
  const [status, setStatus] = useState<DownloadStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const checkStatus = useCallback(() => {
    getDownloadStatus(video.video_id)
      .then(setStatus)
      .catch((e) => setError(e.message));
  }, [video.video_id]);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  // Tick elapsed time while downloading
  useEffect(() => {
    if (!downloading) return;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [downloading]);

  const handleDownload = async () => {
    setDownloading(true);
    setError(null);
    try {
      await downloadVideo(video.video_id);
      checkStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  };

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="space-y-4 pt-2 max-w-xl">
      {status === null ? (
        <p className="text-sm text-muted-foreground">Checking download status...</p>
      ) : status.downloaded ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-sm font-medium">Downloaded</span>
          </div>
          <Field label="Path" value={status.path!} />
          <Field label="Size" value={`${status.file_size_mb} MB`} />
          <button
            onClick={onDownloaded}
            className="text-sm text-primary hover:underline"
          >
            Continue to Calibrate &rarr;
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-yellow-500" />
            <span className="text-sm">Not downloaded yet</span>
          </div>
          {downloading ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: "100%" }} />
                </div>
                <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">{fmtTime(elapsed)}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Downloading via yt-dlp...
              </p>
            </div>
          ) : (
            <button
              onClick={handleDownload}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90"
            >
              Download Video
            </button>
          )}
        </div>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

// ── Step 3: Calibrate ───────────────────────────────────────

function fmtTime(secs: number) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function CalibrateStep({ video }: { video: CrawlVideo }) {
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [channelCal, setChannelCal] = useState<CalibrationEntry | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const debouncedFrameIdx = useDebouncedValue(frameIdx, 150);
  const [overlayBbox, setOverlayBbox] = useState<Bbox | null>(null);
  const [cameraBbox, setCameraBbox] = useState<Bbox | null>(null);
  const [drawingMode, setDrawingMode] = useState<"overlay" | "camera">("overlay");
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [boardTheme, setBoardTheme] = useState("lichess_default");
  const [clipStartTime, setClipStartTime] = useState(0);
  const [clipEndTime, setClipEndTime] = useState<number | null>(null);
  const [clipLabel, setClipLabel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [autoCalLoading, setAutoCalLoading] = useState(false);
  const [autoCalResult, setAutoCalResult] = useState<any>(null);

  const channelHandle = video.channel_handle;

  // Load data on mount
  useEffect(() => {
    getDownloadStatus(video.video_id).then(setDownloadStatus);
    listVideoClips(video.video_id).then(setClips);
    if (channelHandle) {
      listCalibrations().then((cals) => {
        const existing = cals.find((c) => c.channel_handle === channelHandle);
        if (existing) setChannelCal(existing);
      });
    }
  }, [video.video_id, channelHandle]);

  // Open video session once downloaded
  useEffect(() => {
    if (downloadStatus?.downloaded && downloadStatus.path && !session) {
      openVideo(downloadStatus.path, channelHandle || undefined)
        .then(setSession)
        .catch((e) => setError(e.message));
    }
  }, [downloadStatus, channelHandle, session]);

  // Load selected clip into editor state
  const selectClip = useCallback((idx: number) => {
    const clip = clips[idx];
    if (!clip) return;
    setSelectedIdx(idx);
    setOverlayBbox({ x: clip.overlay_bbox[0], y: clip.overlay_bbox[1], w: clip.overlay_bbox[2], h: clip.overlay_bbox[3] });
    setCameraBbox({ x: clip.camera_bbox[0], y: clip.camera_bbox[1], w: clip.camera_bbox[2], h: clip.camera_bbox[3] });
    setBoardFlipped(clip.board_flipped);
    setBoardTheme(clip.board_theme);
    setClipStartTime(clip.start_time);
    setClipEndTime(clip.end_time);
    setClipLabel(clip.label || "");
    // Jump scrubber to clip start
    if (session && session.fps > 0) {
      setFrameIdx(Math.round(clip.start_time * session.fps));
    }
  }, [clips, session]);

  // Auto-select first clip when clips load
  useEffect(() => {
    if (clips.length > 0 && selectedIdx === null) selectClip(0);
  }, [clips, selectedIdx, selectClip]);

  const handleAddClip = async () => {
    if (!session) return;
    setError(null);
    const fps = session.fps;
    const currentTime = fps > 0 ? frameIdx / fps : 0;
    const duration = session.duration_seconds;

    // Default bbox from channel calibration or empty
    const defOverlay: [number, number, number, number] = channelCal
      ? [channelCal.overlay[0], channelCal.overlay[1], channelCal.overlay[2], channelCal.overlay[3]]
      : [0, 0, 100, 100];
    const defCamera: [number, number, number, number] = channelCal
      ? [channelCal.camera[0], channelCal.camera[1], channelCal.camera[2], channelCal.camera[3]]
      : [0, 0, 100, 100];

    try {
      const newClip = await createVideoClip(video.video_id, {
        start_time: currentTime,
        end_time: duration,
        label: null,
        overlay_bbox: defOverlay,
        camera_bbox: defCamera,
        ref_resolution: [session.width, session.height],
        board_flipped: channelCal?.board_flipped ?? false,
        board_theme: channelCal?.board_theme ?? "lichess_default",
      });
      const updated = [...clips, newClip];
      setClips(updated);
      selectClip(updated.length - 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add clip");
    }
  };

  const handleSaveClip = async () => {
    if (selectedIdx === null || !overlayBbox || !cameraBbox) return;
    const clip = clips[selectedIdx];
    setSaving(true);
    setError(null);
    try {
      const updated = await updateVideoClip(video.video_id, clip.id, {
        start_time: clipStartTime,
        end_time: clipEndTime,
        label: clipLabel || null,
        overlay_bbox: [overlayBbox.x, overlayBbox.y, overlayBbox.w, overlayBbox.h],
        camera_bbox: [cameraBbox.x, cameraBbox.y, cameraBbox.w, cameraBbox.h],
        ref_resolution: [session?.width || 1920, session?.height || 1080],
        board_flipped: boardFlipped,
        board_theme: boardTheme,
      });
      const newClips = [...clips];
      newClips[selectedIdx] = updated;
      setClips(newClips);
      setSuccess("Clip saved");
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteClip = async (idx: number) => {
    const clip = clips[idx];
    setError(null);
    try {
      await deleteVideoClip(video.video_id, clip.id);
      const updated = await listVideoClips(video.video_id);
      setClips(updated);
      if (updated.length === 0) {
        setSelectedIdx(null);
      } else {
        selectClip(Math.min(idx, updated.length - 1));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const handleAutoCalibrate = async () => {
    setAutoCalLoading(true);
    setError(null);
    try {
      const result = await inspectAutoCalibration(video.video_id);
      setAutoCalResult(result);
      if (result.proposal) {
        const p = result.proposal;
        setOverlayBbox({ x: p.overlay[0], y: p.overlay[1], w: p.overlay[2], h: p.overlay[3] });
        setCameraBbox({ x: p.camera[0], y: p.camera[1], w: p.camera[2], h: p.camera[3] });
        setBoardTheme(p.theme);
        setBoardFlipped(p.board_flipped);
        setSuccess("Auto-calibration applied");
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError("No overlay detected — could not auto-calibrate");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-calibration failed");
    } finally {
      setAutoCalLoading(false);
    }
  };

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
  const selected = selectedIdx !== null ? clips[selectedIdx] : null;

  return (
    <div className="flex gap-4 pt-2" style={{ height: "calc(100vh - 12rem)" }}>
      {/* Left: Clip list */}
      <div className="w-64 flex-shrink-0 flex flex-col gap-2 overflow-auto">
        <button
          onClick={handleAddClip}
          className="w-full px-3 py-2 rounded-lg border border-dashed text-xs font-medium text-muted-foreground hover:border-primary hover:text-foreground transition-colors"
        >
          + Add Clip
        </button>
        <button
          onClick={handleAutoCalibrate}
          disabled={autoCalLoading}
          className="w-full px-3 py-2 rounded-lg border text-xs font-medium bg-green-500/10 border-green-500/30 text-green-700 hover:bg-green-500/20 transition-colors disabled:opacity-50"
        >
          {autoCalLoading ? "Auto-Calibrating..." : "Auto-Calibrate"}
        </button>

        {clips.length === 0 && (
          <p className="text-xs text-muted-foreground px-1">
            No clips defined. Add a clip to define calibration regions for this video.
          </p>
        )}

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
                onClick={(e) => { e.stopPropagation(); handleDeleteClip(i); }}
                className="text-xs text-muted-foreground hover:text-destructive"
                title="Delete clip"
              >
                &times;
              </button>
            </div>
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {fmtTime(c.start_time)} &mdash; {c.end_time != null ? fmtTime(c.end_time) : "end"}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {c.overlay_bbox[2]}x{c.overlay_bbox[3]} overlay
            </div>
          </div>
        ))}
      </div>

      {/* Right: Calibration editor */}
      <div className="flex-1 space-y-3 overflow-auto">
        {selected === null ? (
          <p className="text-sm text-muted-foreground">Select or add a clip to begin calibrating.</p>
        ) : (
          <>
            {/* Clip time range */}
            <div className="flex items-center gap-3 text-xs">
              <label className="text-muted-foreground">Label:</label>
              <input
                type="text"
                value={clipLabel}
                onChange={(e) => setClipLabel(e.target.value)}
                placeholder={`Clip ${selected.clip_index + 1}`}
                className="h-7 rounded-md border bg-background px-2 text-xs w-40"
              />
              <label className="text-muted-foreground ml-2">Start (s):</label>
              <input
                type="number"
                value={clipStartTime}
                onChange={(e) => setClipStartTime(Number(e.target.value))}
                min={0}
                max={duration}
                step={0.1}
                className="h-7 rounded-md border bg-background px-2 text-xs w-20 tabular-nums"
              />
              <label className="text-muted-foreground">End (s):</label>
              <input
                type="number"
                value={clipEndTime ?? duration}
                onChange={(e) => setClipEndTime(Number(e.target.value))}
                min={clipStartTime}
                max={duration}
                step={0.1}
                className="h-7 rounded-md border bg-background px-2 text-xs w-20 tabular-nums"
              />
            </div>

            {/* Frame scrubber */}
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

              {/* Clip timeline */}
              <div className="relative w-full h-3 bg-muted rounded-full overflow-hidden">
                {clips.map((c, i) => {
                  const startPct = duration > 0 ? (c.start_time / duration) * 100 : 0;
                  const endTime = c.end_time ?? duration;
                  const widthPct = duration > 0 ? ((endTime - c.start_time) / duration) * 100 : 0;
                  return (
                    <div
                      key={c.id}
                      onClick={() => selectClip(i)}
                      title={`${c.label || `Clip ${c.clip_index + 1}`}: ${fmtTime(c.start_time)} — ${c.end_time != null ? fmtTime(c.end_time) : "end"}`}
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

              {/* Set start/end from current frame */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setClipStartTime(parseFloat((fps > 0 ? frameIdx / fps : 0).toFixed(1)))}
                  className="px-3 py-1 rounded-md border text-xs font-medium hover:bg-muted transition-colors"
                >
                  Set Start
                </button>
                <button
                  onClick={() => setClipEndTime(parseFloat((fps > 0 ? frameIdx / fps : 0).toFixed(1)))}
                  className="px-3 py-1 rounded-md border text-xs font-medium hover:bg-muted transition-colors"
                >
                  Set End
                </button>
                <span className="text-xs text-muted-foreground">
                  Current: {fps > 0 ? (frameIdx / fps).toFixed(1) : "0"}s
                </span>
              </div>
            </div>

            {/* Drawing mode controls */}
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

            {/* Bbox drawer */}
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

            {/* Options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={boardFlipped} onChange={(e) => setBoardFlipped(e.target.checked)} className="rounded" />
                Board flipped
              </label>
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted-foreground">Theme:</label>
                <select value={boardTheme} onChange={(e) => setBoardTheme(e.target.value)} className="h-8 rounded-md border bg-background px-2 text-sm">
                  <option value="lichess_default">Lichess Default</option>
                  <option value="chess_com">Chess.com</option>
                </select>
              </div>
            </div>

            {/* Auto-calibration preview */}
            {autoCalResult && !autoCalResult.error && (
              <div className="space-y-3 border rounded-lg p-3">
                <h4 className="text-xs font-medium text-muted-foreground">Auto-Calibration Preview</h4>
                {autoCalResult.proposal_frame_base64 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">
                      Green = Overlay, Red = Camera
                    </p>
                    <img
                      src={`data:image/jpeg;base64,${autoCalResult.proposal_frame_base64}`}
                      alt="Proposal"
                      className="w-full max-w-lg rounded border"
                    />
                  </div>
                )}
                {autoCalResult.camera_motion_heatmap_base64 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">
                      Camera Motion Heatmap — Red = motion (camera), Blue = static
                    </p>
                    <img
                      src={`data:image/jpeg;base64,${autoCalResult.camera_motion_heatmap_base64}`}
                      alt="Heatmap"
                      className="w-full max-w-lg rounded border"
                    />
                  </div>
                )}
              </div>
            )}

            {/* Save */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleSaveClip}
                disabled={saving || !overlayBbox || !cameraBbox}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save Clip"}
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

// ── Step 4: Extract ─────────────────────────────────────────

function ClipExtractCard({
  clip,
  session,
}: {
  clip: VideoClip;
  session: VideoSession;
}) {
  const fps = session.fps;
  const startFrame = Math.round(clip.start_time * fps);
  const endFrame = clip.end_time != null ? Math.round(clip.end_time * fps) : session.total_frames - 1;

  const [frameIdx, setFrameIdx] = useState(startFrame);
  const [overlayResult, setOverlayResult] = useState<FrameOverlayResponse | null>(null);
  const [readingFrame, setReadingFrame] = useState(false);
  const [detection, setDetection] = useState<VideoMoveDetectionResponse | null>(null);
  const [detectingMoves, setDetectingMoves] = useState(false);
  const [expandedSegment, setExpandedSegment] = useState<number | null>(null);

  const handleReadFrame = async () => {
    setReadingFrame(true);
    try {
      const result = await readOverlayFrame(session.session_id, frameIdx, clip.id);
      setOverlayResult(result);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to read overlay");
    } finally {
      setReadingFrame(false);
    }
  };

  const handleDetectMoves = async () => {
    setDetectingMoves(true);
    try {
      const result = await detectVideoMoves(session.session_id, 2.0, clip.id);
      setDetection(result);
      if (result.segments.length > 0) setExpandedSegment(0);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Move detection failed");
    } finally {
      setDetectingMoves(false);
    }
  };

  return (
    <div className="border rounded-lg p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-medium">{clip.label || `Clip ${clip.clip_index + 1}`}</span>
        <span className="text-[10px] text-muted-foreground">
          {fmtTime(clip.start_time)} &mdash; {clip.end_time != null ? fmtTime(clip.end_time) : "end"}
        </span>
      </div>

      {/* Frame sampling */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
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
                </div>
              ) : (
                <p className="text-xs text-destructive">Could not read board</p>
              )}
            </div>
          </div>
        )}
      </div>

      <hr />

      {/* Move detection */}
      <div className="space-y-2">
        <button
          onClick={handleDetectMoves}
          disabled={detectingMoves}
          className="px-3 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {detectingMoves ? "Detecting..." : "Run Detection"}
        </button>
        {detection && (
          <div className="space-y-2 pl-2">
            <div className="text-xs text-muted-foreground">
              Sampled {detection.num_frames_sampled} frames, {detection.num_readable} readable.
              Found {detection.segments.length} game(s).
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

function ExtractStep({ video }: { video: CrawlVideo }) {
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDownloadStatus(video.video_id).then(setDownloadStatus);
    listVideoClips(video.video_id).then(setClips);
  }, [video.video_id]);

  useEffect(() => {
    if (downloadStatus?.downloaded && downloadStatus.path && !session) {
      openVideo(downloadStatus.path, video.channel_handle || undefined)
        .then(setSession)
        .catch((e) => setError(e.message));
    }
  }, [downloadStatus, video.channel_handle, session]);

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
        <ClipExtractCard key={clip.id} clip={clip} session={session} />
      ))}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

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
          {/* PGN */}
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">PGN</label>
            <pre className="text-xs bg-muted/30 rounded p-2 whitespace-pre-wrap font-mono max-h-32 overflow-auto">
              {segment.pgn_moves}
            </pre>
          </div>

          {/* Move table */}
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

// ── Step 5: Generate ────────────────────────────────────────

function GenerateStep({ video }: { video: CrawlVideo }) {
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [videoClips, setVideoClips] = useState<VideoClip[]>([]);
  const [generatingId, setGeneratingId] = useState<number | null>(null);
  // Per-clip generated results: Map<clipId, GeneratedClip[]>
  const [generated, setGenerated] = useState<Map<number, GeneratedClip[]>>(new Map());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDownloadStatus(video.video_id).then(setDownloadStatus);
    listVideoClips(video.video_id).then(setVideoClips);
  }, [video.video_id]);

  useEffect(() => {
    if (downloadStatus?.downloaded && downloadStatus.path && !session) {
      openVideo(downloadStatus.path, video.channel_handle || undefined)
        .then(setSession)
        .catch((e) => setError(e.message));
    }
  }, [downloadStatus, video.channel_handle, session]);

  const handleGenerate = async (clipId?: number) => {
    if (!session) return;
    setGeneratingId(clipId ?? -1);
    setError(null);
    try {
      const result = await generateClips(session.session_id, clipId);
      setGenerated((prev) => new Map(prev).set(clipId ?? -1, result.clips));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGeneratingId(null);
    }
  };

  if (!downloadStatus?.downloaded) {
    return <p className="text-sm text-muted-foreground pt-2">Video must be downloaded first.</p>;
  }

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video session...</p>;
  }

  const hasClips = videoClips.length > 0;
  if (!hasClips && !session.has_calibration) {
    return <p className="text-sm text-muted-foreground pt-2">No clips or calibration found. Go to the Calibrate step first.</p>;
  }

  return (
    <div className="space-y-4 pt-2 max-w-3xl">
      {hasClips ? (
        <div className="space-y-4">
          {videoClips.map((vc) => {
            const gen = generated.get(vc.id) || [];
            const isGenerating = generatingId === vc.id;
            return (
              <div key={vc.id} className="border rounded-lg p-3 space-y-2">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium">{vc.label || `Clip ${vc.clip_index + 1}`}</span>
                  <span className="text-[10px] text-muted-foreground">
                    {fmtTime(vc.start_time)} &mdash; {vc.end_time != null ? fmtTime(vc.end_time) : "end"}
                  </span>
                  <button
                    onClick={() => handleGenerate(vc.id)}
                    disabled={isGenerating}
                    className="px-3 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
                  >
                    {isGenerating ? "Generating..." : "Generate"}
                  </button>
                </div>
                {gen.length > 0 && (
                  <div className="space-y-1 pl-2">
                    <div className="text-xs text-muted-foreground">{gen.length} training clip(s)</div>
                    {gen.map((g) => (
                      <div key={g.game_index} className="text-xs">
                        <span className="font-medium">Game {g.game_index + 1}</span>
                        <span className="text-muted-foreground ml-2">{g.num_moves} moves, {g.num_frames} frames</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="space-y-3">
          <button
            onClick={() => handleGenerate()}
            disabled={generatingId !== null}
            className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {generatingId !== null ? "Generating clips..." : "Generate Training Clips"}
          </button>
          {(() => {
            const gen = generated.get(-1) || [];
            if (gen.length === 0) return null;
            return (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Generated {gen.length} clip(s)</h3>
                {gen.map((clip) => (
                  <div key={clip.game_index} className="border rounded-lg p-3 space-y-1">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-medium">Game {clip.game_index + 1}</span>
                      <span className="text-xs text-muted-foreground">{clip.num_moves} moves, {clip.num_frames} frames</span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground">{clip.filepath}</div>
                  </div>
                ))}
              </div>
            );
          })()}
        </div>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

// ── Step 6: Inspect ─────────────────────────────────────────

function InspectStep({ video }: { video: CrawlVideo }) {
  const [clips, setClips] = useState<GeneratedClip[]>([]);
  const [inspecting, setInspecting] = useState<number | null>(null);
  const [clipInfo, setClipInfo] = useState<Map<number, ClipInspectResponse>>(new Map());
  const [clipSessions, setClipSessions] = useState<Map<number, string>>(new Map());
  const [frameIndices, setFrameIndices] = useState<Map<number, number>>(new Map());
  const [error, setError] = useState<string | null>(null);

  // Try to find existing clips for this video
  useEffect(() => {
    // Check for .pt files in data/training_clips matching this video
    // We'll use the generate endpoint's returned data if available,
    // or the user can generate first then come here
  }, [video.video_id]);

  const handleInspectClip = async (clip: GeneratedClip) => {
    setInspecting(clip.game_index);
    setError(null);
    try {
      // Load clip file via the clips API
      const response = await fetch(clip.filepath);
      const blob = await response.blob();
      const file = new File([blob], clip.filepath.split("/").pop() || "clip.pt");
      const { session_id } = await loadClip(file);
      setClipSessions((prev) => new Map(prev).set(clip.game_index, session_id));

      const info = await getClipInfo(session_id);
      setClipInfo((prev) => new Map(prev).set(clip.game_index, info));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to inspect clip");
    } finally {
      setInspecting(null);
    }
  };

  return (
    <div className="space-y-4 pt-2 max-w-4xl">
      <p className="text-sm text-muted-foreground">
        Generate clips first (Step 5), then inspect them here to verify training data quality.
      </p>

      {clips.length === 0 && (
        <p className="text-xs text-muted-foreground">No clips to inspect yet. Run clip generation first.</p>
      )}

      {clips.map((clip) => {
        const info = clipInfo.get(clip.game_index);
        const sessionId = clipSessions.get(clip.game_index);
        const currentFrame = frameIndices.get(clip.game_index) ?? 0;

        return (
          <div key={clip.game_index} className="border rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium">Game {clip.game_index + 1}</span>
              <span className="text-xs text-muted-foreground">{clip.num_moves} moves, {clip.num_frames} frames</span>
              <button
                onClick={() => handleInspectClip(clip)}
                disabled={inspecting === clip.game_index}
                className="text-xs text-primary hover:underline"
              >
                {inspecting === clip.game_index ? "Loading..." : "Inspect"}
              </button>
            </div>

            {info && sessionId && (
              <div className="space-y-2">
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Replay:</span>{" "}
                    <span className={info.replay_valid ? "text-green-600" : "text-destructive"}>
                      {info.replay_valid ? "Valid" : "Invalid"}
                    </span>
                  </div>
                  <div><span className="text-muted-foreground">Moves:</span> {info.total_moves}</div>
                  <div><span className="text-muted-foreground">No-move frames:</span> {info.no_move_frames}</div>
                  <div><span className="text-muted-foreground">Avg legal:</span> {info.avg_legal_moves?.toFixed(1)}</div>
                </div>
                {info.replay_error && (
                  <p className="text-xs text-destructive">{info.replay_error}</p>
                )}

                {/* Frame scrubber */}
                <div>
                  <input
                    type="range"
                    min={0}
                    max={info.num_frames - 1}
                    value={currentFrame}
                    onChange={(e) => setFrameIndices((prev) => new Map(prev).set(clip.game_index, Number(e.target.value)))}
                    className="w-full"
                  />
                  <div className="flex gap-3">
                    <img
                      src={clipFrameUrl(sessionId, currentFrame)}
                      alt={`Frame ${currentFrame}`}
                      className="w-56 rounded border"
                    />
                    <div className="text-xs space-y-1">
                      <div>Frame {currentFrame} / {info.num_frames}</div>
                      {info.moves.filter((m) => m.frame_index === currentFrame).map((m) => (
                        <div key={m.frame_index} className="font-medium text-primary">
                          Move: {m.san || m.uci}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
