"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import {
  deleteVideoSession,
  getDownloadStatus,
  getVideo,
  listVideoClips,
  openVideo,
  readOverlayFrame,
} from "@/lib/api";
import type { CrawlVideo, VideoClip, VideoSession } from "@/lib/types";
import { ChessBoard } from "@/components/ChessBoard";

interface OverlayFrameSample {
  sampleId: string;
  videoId: string;
  clipId: number;
  frameIndex: number;
  timestampSeconds: number;
  predictedFen: string | null;
  readMethod: string | null;
  overlayCropB64: string;
}

function parseFrameIndices(raw: string): number[] {
  const values = raw
    .split(/[,\s]+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part))
    .filter((value) => Number.isInteger(value) && value >= 0);
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function buildSampleId(clipId: number, frameIndex: number): string {
  return `${clipId}f${frameIndex.toString().padStart(5, "0")}`;
}

function clipLabel(clip: VideoClip): string {
  return clip.label || `Clip ${clip.clip_index + 1}`;
}

function fmtSeconds(value: number): string {
  return `${value.toFixed(2)}s`;
}

export default function OverlayFenAnnotationPage() {
  const searchParams = useSearchParams();
  const initialVideoId = searchParams.get("video") ?? "";
  const initialFrames = searchParams.get("frames") ?? "";
  const initialClipId = searchParams.get("clip") ?? "";

  const [videoId, setVideoId] = useState(initialVideoId);
  const [frameIndicesInput, setFrameIndicesInput] = useState(initialFrames);
  const [video, setVideo] = useState<CrawlVideo | null>(null);
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<number | null>(
    initialClipId ? Number(initialClipId) : null,
  );
  const [session, setSession] = useState<VideoSession | null>(null);
  const [loadingContext, setLoadingContext] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [samples, setSamples] = useState<OverlayFrameSample[]>([]);
  const [editedFens, setEditedFens] = useState<Record<string, string>>({});
  const [rejected, setRejected] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState<Set<string>>(new Set());
  const [savingId, setSavingId] = useState<string | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const sessionRef = useRef<VideoSession | null>(null);
  const autoLoadRef = useRef(false);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    return () => {
      const activeSession = sessionRef.current;
      if (activeSession) {
        void deleteVideoSession(activeSession.session_id).catch(() => undefined);
      }
    };
  }, []);

  const activeClips = useMemo(() => clips.filter((clip) => !clip.is_gap), [clips]);
  const selectedClip = activeClips.find((clip) => clip.id === selectedClipId) ?? null;
  const parsedFrameIndices = useMemo(() => parseFrameIndices(frameIndicesInput), [frameIndicesInput]);

  useEffect(() => {
    if (!selectedClipId && activeClips.length > 0) {
      setSelectedClipId(activeClips[0].id);
    }
  }, [activeClips, selectedClipId]);

  useEffect(() => {
    if (autoLoadRef.current || !initialVideoId || !initialFrames) {
      return;
    }
    autoLoadRef.current = true;
    void loadSamples();
  }, [initialFrames, initialVideoId]);

  async function ensureVideoContext(): Promise<{
    video: CrawlVideo;
    clips: VideoClip[];
    session: VideoSession;
  }> {
    const trimmedVideoId = videoId.trim();
    if (!trimmedVideoId) {
      throw new Error("Enter a video ID");
    }

    if (video && session && video.video_id === trimmedVideoId && activeClips.length > 0) {
      return { video, clips: activeClips, session };
    }

    setLoadingContext(true);
    try {
      const [nextVideo, downloadStatus, nextClips] = await Promise.all([
        getVideo(trimmedVideoId),
        getDownloadStatus(trimmedVideoId),
        listVideoClips(trimmedVideoId),
      ]);

      if (!downloadStatus.downloaded || !downloadStatus.path) {
        throw new Error("Video must be downloaded first");
      }

      const active = nextClips.filter((clip) => !clip.is_gap);
      if (active.length === 0) {
        throw new Error("No calibrated clips found for this video");
      }

      if (sessionRef.current) {
        await deleteVideoSession(sessionRef.current.session_id).catch(() => undefined);
      }

      const nextSession = await openVideo(downloadStatus.path, nextVideo.channel_handle ?? undefined);
      setVideo(nextVideo);
      setClips(nextClips);
      setSession(nextSession);

      const requestedClipId = initialClipId ? Number(initialClipId) : null;
      const nextSelectedClipId =
        requestedClipId && active.some((clip) => clip.id === requestedClipId)
          ? requestedClipId
          : active[0].id;
      setSelectedClipId(nextSelectedClipId);

      return { video: nextVideo, clips: active, session: nextSession };
    } finally {
      setLoadingContext(false);
    }
  }

  async function loadSamples(): Promise<void> {
    if (parsedFrameIndices.length === 0) {
      toast.error("Enter one or more frame indices");
      return;
    }

    try {
      const context = await ensureVideoContext();
      const clipId =
        selectedClipId && context.clips.some((clip) => clip.id === selectedClipId)
          ? selectedClipId
          : context.clips[0].id;
      const currentClip = context.clips.find((clip) => clip.id === clipId) ?? context.clips[0];
      setSelectedClipId(currentClip.id);
      setLoadingSamples(true);
      setProgress({ current: 0, total: parsedFrameIndices.length });
      setSamples([]);
      setEditedFens({});
      setRejected(new Set());
      setSaved(new Set());

      const nextSamples: OverlayFrameSample[] = [];
      const nextEditedFens: Record<string, string> = {};

      for (let index = 0; index < parsedFrameIndices.length; index += 1) {
        const frameIndex = parsedFrameIndices[index];
        const result = await readOverlayFrame(
          context.session.session_id,
          frameIndex,
          currentClip.id,
          "overlay",
        );
        const sampleId = buildSampleId(currentClip.id, frameIndex);
        const sample: OverlayFrameSample = {
          sampleId,
          videoId: trimmedVideoId(context.video.video_id),
          clipId: currentClip.id,
          frameIndex,
          timestampSeconds: result.timestamp_seconds,
          predictedFen: result.fen,
          readMethod: result.read_method,
          overlayCropB64: result.overlay_crop_b64,
        };
        nextSamples.push(sample);
        if (result.fen) {
          nextEditedFens[sampleId] = result.fen;
        }
        setProgress({ current: index + 1, total: parsedFrameIndices.length });
      }

      setSamples(nextSamples);
      setEditedFens(nextEditedFens);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load samples");
    } finally {
      setLoadingSamples(false);
    }
  }

  async function saveOne(sample: OverlayFrameSample): Promise<void> {
    const fen = editedFens[sample.sampleId];
    if (!fen) {
      return;
    }

    setSavingId(sample.sampleId);
    try {
      const response = await fetch("/api/models/overlay-test/extract-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmations: [
            {
              sample_id: sample.sampleId,
              video_id: sample.videoId,
              fen,
              image_b64: sample.overlayCropB64,
            },
          ],
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      if (Array.isArray(data.errors) && data.errors.length > 0) {
        throw new Error(data.errors.join("\n"));
      }
      setSaved((prev) => new Set(prev).add(sample.sampleId));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save sample");
    } finally {
      setSavingId(null);
    }
  }

  const hasSamples = samples.length > 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-lg font-semibold">Overlay FEN Annotation</h2>
        <button
          onClick={() => void loadSamples()}
          disabled={loadingContext || loadingSamples}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loadingContext ? "Opening video..." : loadingSamples ? "Loading frames..." : "Load Samples"}
        </button>
      </div>

      <p className="text-sm text-muted-foreground">
        Enter a video ID and exact frame indices. The page reads the overlay crop, predicts a FEN,
        and lets you correct + save each crop into <code className="text-xs">data/overlay/val_real/</code>.
      </p>

      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <label className="text-sm text-muted-foreground flex flex-col gap-1">
          Video ID
          <input
            type="text"
            value={videoId}
            onChange={(event) => setVideoId(event.target.value)}
            placeholder="RyXsGZckLHQ"
            className="h-9 rounded-md border bg-background px-3 text-sm font-mono"
          />
        </label>
        <label className="text-sm text-muted-foreground flex flex-col gap-1">
          Frame indices
          <input
            type="text"
            value={frameIndicesInput}
            onChange={(event) => setFrameIndicesInput(event.target.value)}
            placeholder="128,250,375,500,625"
            className="h-9 rounded-md border bg-background px-3 text-sm font-mono"
          />
        </label>
      </div>

      {activeClips.length > 0 && (
        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <label className="text-sm text-muted-foreground flex flex-col gap-1">
            Clip
            <select
              value={selectedClipId ?? undefined}
              onChange={(event) => setSelectedClipId(Number(event.target.value))}
              className="h-9 rounded-md border bg-background px-3 text-sm"
            >
              {activeClips.map((clip) => (
                <option key={clip.id} value={clip.id}>
                  {clipLabel(clip)}
                </option>
              ))}
            </select>
          </label>
          {selectedClip && (
            <div className="rounded-md border px-3 py-2 text-sm text-muted-foreground">
              {clipLabel(selectedClip)} · {selectedClip.start_time.toFixed(1)}s — {selectedClip.end_time != null ? `${selectedClip.end_time.toFixed(1)}s` : "end"}
            </div>
          )}
        </div>
      )}

      {loadingSamples && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Loading overlay frames...</span>
            <span>
              {progress.current}/{progress.total}
            </span>
          </div>
          <div className="h-2 bg-muted rounded overflow-hidden">
            <div
              className="h-full bg-foreground rounded transition-all duration-300"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {hasSamples && (
        <div className="flex items-center gap-4 text-sm">
          <span className="text-muted-foreground">
            {samples.length} samples
          </span>
          <span className="text-muted-foreground">
            {saved.size} saved
          </span>
          <span className="text-muted-foreground">
            {rejected.size} rejected
          </span>
        </div>
      )}

      <div className="space-y-4">
        {samples.map((sample) => {
          const currentFen = editedFens[sample.sampleId] ?? "";
          const isSaved = saved.has(sample.sampleId);
          const isRejected = rejected.has(sample.sampleId);
          const locked = isSaved || isRejected;
          return (
            <div
              key={sample.sampleId}
              className={`border rounded-lg p-3 space-y-3 ${isRejected ? "opacity-40" : ""} ${isSaved ? "border-green-400" : ""}`}
            >
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium font-mono">{sample.sampleId}</span>
                  <a
                    href={`/videos/${sample.videoId}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-blue-600 hover:underline font-mono"
                  >
                    {sample.videoId}
                  </a>
                  <span className="text-xs text-muted-foreground">
                    frame {sample.frameIndex} · {fmtSeconds(sample.timestampSeconds)}
                  </span>
                  {sample.readMethod && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                      {sample.readMethod}
                    </span>
                  )}
                  {isSaved && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-medium">
                      Saved
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => void saveOne(sample)}
                    disabled={!currentFen || locked || savingId === sample.sampleId}
                    className="text-xs px-2 py-1 rounded border bg-green-600 text-white border-green-600 disabled:opacity-40"
                  >
                    {savingId === sample.sampleId ? "Saving..." : "Save"}
                  </button>
                  <button
                    onClick={() =>
                      setRejected((prev) => {
                        const next = new Set(prev);
                        if (next.has(sample.sampleId)) {
                          next.delete(sample.sampleId);
                        } else {
                          next.add(sample.sampleId);
                        }
                        return next;
                      })
                    }
                    disabled={isSaved}
                    className={`text-xs px-2 py-1 rounded border ${
                      isRejected
                        ? "bg-red-100 text-red-700 border-red-300"
                        : "text-muted-foreground hover:text-foreground"
                    } disabled:opacity-40`}
                  >
                    {isRejected ? "Rejected" : "Reject"}
                  </button>
                </div>
              </div>

              <div className="grid gap-3 lg:grid-cols-3">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Overlay Crop</label>
                  <img
                    src={`data:image/jpeg;base64,${sample.overlayCropB64}`}
                    alt={sample.sampleId}
                    className="w-full max-w-[260px] rounded border"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Board</label>
                  {currentFen ? (
                    <ChessBoard fen={currentFen} size={240} />
                  ) : (
                    <div className="w-[240px] h-[240px] rounded border border-dashed flex items-center justify-center text-xs text-muted-foreground">
                      No FEN
                    </div>
                  )}
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">FEN</label>
                  <textarea
                    value={currentFen}
                    onChange={(event) =>
                      setEditedFens((prev) => ({
                        ...prev,
                        [sample.sampleId]: event.target.value,
                      }))
                    }
                    rows={4}
                    disabled={locked}
                    className="w-full rounded border bg-background px-3 py-2 text-sm font-mono resize-none disabled:opacity-60"
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function trimmedVideoId(videoId: string): string {
  return videoId.trim();
}
