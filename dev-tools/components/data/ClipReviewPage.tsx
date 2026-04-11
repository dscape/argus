"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  clipFrameUrl,
  clipOverlayFrameUrl,
  clipSourceVideoUrl,
  deleteClipSession,
  getClipAnnotation,
  getClipInfo,
  loadClipFromPath,
  saveClipAnnotation,
} from "@/lib/api";
import type { ClipInspectResponse, DetectedMove } from "@/lib/types";

type DatasetKind = "real" | "synthetic";

interface ClipReviewPageProps {
  dataset: DatasetKind;
  directory: string;
  filename: string;
}

export function ClipReviewPage({ dataset, directory, filename }: ClipReviewPageProps) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<ClipInspectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const sessionRef = useRef<string | null>(null);

  useEffect(() => {
    const filepath = `${directory}/${filename}`;
    let cancelled = false;

    async function load() {
      let createdSessionId: string | null = null;
      setLoading(true);
      setError(null);
      setClipInfo(null);
      setSessionId(null);

      try {
        const { session_id } = await loadClipFromPath(filepath);
        createdSessionId = session_id;
        if (cancelled) {
          await deleteClipSession(session_id).catch(() => {});
          return;
        }

        sessionRef.current = session_id;
        setSessionId(session_id);

        const info = await getClipInfo(session_id);
        if (cancelled) {
          await deleteClipSession(session_id).catch(() => {});
          return;
        }
        setClipInfo(info);
      } catch (e: unknown) {
        if (createdSessionId) {
          if (sessionRef.current === createdSessionId) {
            sessionRef.current = null;
          }
          await deleteClipSession(createdSessionId).catch(() => {});
        }
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load clip");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
      const sid = sessionRef.current;
      sessionRef.current = null;
      if (sid) {
        void deleteClipSession(sid).catch(() => {});
      }
    };
  }, [directory, filename]);

  if (loading) {
    return <ClipReviewSkeleton dataset={dataset} filename={filename} />;
  }

  if (error || !sessionId || !clipInfo) {
    return (
      <div className="space-y-4">
        <Link href={`/data/${dataset}`} className="text-sm underline underline-offset-4">
          ← Back to {datasetLabel(dataset)} clips
        </Link>
        <Card>
          <CardHeader>
            <CardDescription>{datasetLabel(dataset)} clip review</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p className="font-medium">{filename}</p>
            <p className="text-destructive">{error ?? "Failed to load clip"}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <ClipReviewContent
      dataset={dataset}
      filename={filename}
      sessionId={sessionId}
      clipInfo={clipInfo}
    />
  );
}

function ClipReviewContent({
  dataset,
  filename,
  sessionId,
  clipInfo,
}: {
  dataset: DatasetKind;
  filename: string;
  sessionId: string;
  clipInfo: ClipInspectResponse;
}) {
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [overlayUnavailable, setOverlayUnavailable] = useState(false);
  const [videoUnavailable, setVideoUnavailable] = useState(false);
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [sourceVideoLoading, setSourceVideoLoading] = useState(false);
  const [annotationText, setAnnotationText] = useState("");
  const [savedAnnotationText, setSavedAnnotationText] = useState("");
  const [annotationPath, setAnnotationPath] = useState<string | null>(null);
  const [annotationLoading, setAnnotationLoading] = useState(true);
  const [annotationSaving, setAnnotationSaving] = useState(false);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [annotationSavedAt, setAnnotationSavedAt] = useState<string | null>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [videoCurrentTime, setVideoCurrentTime] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const metadata = clipInfo.metadata ?? {};
  const frameCount = clipInfo.num_frames;
  const frameTime = clipInfo.frame_timestamps_seconds[selectedFrame] ?? null;
  const sourceFrame = clipInfo.frame_indices[selectedFrame] ?? selectedFrame;
  const canShowOverlay = dataset === "real" && !overlayUnavailable;
  const canShowSourceVideo = dataset === "real" && !videoUnavailable && sourceVideoUrl !== null;

  useEffect(() => {
    setSelectedFrame(clipInfo.moves[0]?.frame_index ?? 0);
    setOverlayUnavailable(false);
    setVideoUnavailable(false);
  }, [clipInfo]);

  useEffect(() => {
    if (dataset !== "real") {
      setSourceVideoUrl(null);
      return;
    }

    const controller = new AbortController();
    let blobUrl: string | null = null;
    setSourceVideoUrl(null);
    setSourceVideoLoading(true);
    setVideoUnavailable(false);

    fetch(clipSourceVideoUrl(sessionId), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.blob();
      })
      .then((blob) => {
        blobUrl = URL.createObjectURL(blob);
        setSourceVideoUrl(blobUrl);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        console.error(error);
        setVideoUnavailable(true);
        setSourceVideoUrl(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setSourceVideoLoading(false);
        }
      });

    return () => {
      controller.abort();
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [dataset, sessionId]);

  useEffect(() => {
    let cancelled = false;
    setAnnotationLoading(true);
    setAnnotationError(null);
    setAnnotationSavedAt(null);

    getClipAnnotation(filename)
      .then((result) => {
        if (cancelled) {
          return;
        }
        setAnnotationText(result.content);
        setSavedAnnotationText(result.content);
        setAnnotationPath(result.annotation_path);
      })
      .catch((e: unknown) => {
        if (cancelled) {
          return;
        }
        setAnnotationError(e instanceof Error ? e.message : "Failed to load review notes");
      })
      .finally(() => {
        if (!cancelled) {
          setAnnotationLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [filename]);

  const selectedMoveIndex = useMemo(
    () => clipInfo.moves.findIndex((move) => move.frame_index === selectedFrame),
    [clipInfo.moves, selectedFrame]
  );
  const selectedMove = selectedMoveIndex >= 0 ? clipInfo.moves[selectedMoveIndex] : null;
  const moveRows = useMemo(() => groupMoves(clipInfo.moves), [clipInfo.moves]);
  const moveByFrame = useMemo(() => {
    const map = new Map<number, string>();
    clipInfo.moves.forEach((move, index) => {
      map.set(move.frame_index, moveLabel(index, move));
    });
    return map;
  }, [clipInfo.moves]);
  const hasFirstFrameMove = useMemo(
    () => clipInfo.moves.some((move) => move.frame_index == 0),
    [clipInfo.moves]
  );
  const annotationDirty = annotationText !== savedAnnotationText;

  useEffect(() => {
    if (dataset === "real" && frameTime !== null) {
      setVideoCurrentTime(frameTime);
    }
  }, [dataset, frameTime]);

  const syncVideoToTime = useCallback((targetTime: number | null) => {
    if (targetTime === null) {
      return;
    }

    const video = videoRef.current;
    if (!video) {
      return;
    }

    setVideoCurrentTime(targetTime);

    const seek = () => {
      try {
        const duration = Number.isFinite(video.duration) ? video.duration : null;
        const clamped = duration !== null ? Math.min(Math.max(targetTime, 0), duration) : Math.max(targetTime, 0);
        video.currentTime = clamped;
        setVideoCurrentTime(clamped);
      } catch {
        // Ignore transient seek failures while metadata is still loading.
      }
    };

    if (video.readyState >= 1) {
      seek();
      return;
    }

    const handleReady = () => {
      seek();
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("loadeddata", handleReady);
      video.removeEventListener("canplay", handleReady);
    };
    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("loadeddata", handleReady);
    video.addEventListener("canplay", handleReady);
  }, []);

  useEffect(() => {
    if (sourceVideoUrl) {
      videoRef.current?.load();
    }
  }, [sourceVideoUrl]);

  useEffect(() => {
    if (!canShowSourceVideo) {
      return;
    }
    syncVideoToTime(frameTime);
  }, [canShowSourceVideo, frameTime, syncVideoToTime]);

  const handleSaveAnnotation = async () => {
    setAnnotationSaving(true);
    setAnnotationError(null);
    try {
      const result = await saveClipAnnotation(filename, annotationText);
      setSavedAnnotationText(result.content);
      setAnnotationPath(result.annotation_path);
      setAnnotationSavedAt(new Date().toLocaleTimeString());
    } catch (e: unknown) {
      setAnnotationError(e instanceof Error ? e.message : "Failed to save review notes");
    } finally {
      setAnnotationSaving(false);
    }
  };

  const handleJumpVideo = (deltaSeconds: number) => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const nextTime = Math.max(0, video.currentTime + deltaSeconds);
    syncVideoToTime(nextTime);
  };

  const handleTogglePlayback = async () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    if (video.paused) {
      try {
        await video.play();
      } catch {
        // Ignore browser autoplay/playback failures.
      }
      return;
    }
    video.pause();
  };

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <Link href={`/data/${dataset}`} className="text-sm underline underline-offset-4">
          ← Back to {datasetLabel(dataset)} clips
        </Link>
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-2xl font-semibold tracking-tight">{filename}</h3>
          <Badge variant="outline">{clipInfo.file_size_mb} MB</Badge>
          <Badge variant={clipInfo.replay_valid ? "default" : "destructive"}>
            {clipInfo.replay_valid ? "Replay valid" : "Replay invalid"}
          </Badge>
          <Badge variant="secondary">{frameCount} frames</Badge>
          <Badge variant="secondary">{clipInfo.total_moves} moves</Badge>
          {hasFirstFrameMove && <Badge variant="destructive">first frame is a move</Badge>}
        </div>
        <p className="text-sm text-muted-foreground">
          Review the synchronized footage, overlay, move timeline, source video, and frame strip before training.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(300px,380px)]">
        <div className="space-y-4">
          <section className="rounded-3xl border border-amber-500/20 bg-[#171411] p-4 text-stone-100 shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-stone-400">
                  Review console
                </p>
                <h4 className="text-xl font-semibold text-stone-50">
                  Frame {selectedFrame + 1} / {frameCount}
                </h4>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-sm text-stone-300">
                <span>source f{sourceFrame}</span>
                {frameTime !== null && <span>• {formatSeconds(frameTime)}</span>}
                {selectedMove && (
                  <Badge className="border-amber-400/50 bg-amber-500/20 text-amber-50 hover:bg-amber-500/20">
                    {moveLabel(selectedMoveIndex, selectedMove)}
                  </Badge>
                )}
              </div>
            </div>

            {hasFirstFrameMove && selectedFrame === 0 && (
              <div className="mb-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-100">
                The first sampled frame already contains a move. Use the source video player to scrub earlier than this timestamp and annotate whether the clip needs more pre-roll.
              </div>
            )}

            <div className={`grid gap-4 ${canShowOverlay ? "lg:grid-cols-2" : "grid-cols-1"}`}>
              {canShowOverlay && (
                <ReviewFramePanel
                  title="Overlay"
                  caption="Broadcast board state at the same sampled instant"
                  imageUrl={clipOverlayFrameUrl(sessionId, selectedFrame)}
                  alt={`Overlay frame ${selectedFrame}`}
                  onError={() => setOverlayUnavailable(true)}
                />
              )}
              <ReviewFramePanel
                title={dataset === "real" ? "Real footage" : "Synthetic frame"}
                caption={
                  dataset === "real"
                    ? "Stored camera crop written into the training clip"
                    : "Stored synthetic training frame"
                }
                imageUrl={clipFrameUrl(sessionId, selectedFrame)}
                alt={`Clip frame ${selectedFrame}`}
              />
            </div>

            {!canShowOverlay && dataset === "real" && (
              <p className="mt-4 text-sm text-stone-400">
                Overlay preview is unavailable for this clip session.
              </p>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-2">
              <FrameNavButton
                label="Previous"
                disabled={selectedFrame <= 0}
                onClick={() => setSelectedFrame((value) => Math.max(0, value - 1))}
              />
              <FrameNavButton
                label="Next"
                disabled={selectedFrame >= frameCount - 1}
                onClick={() => setSelectedFrame((value) => Math.min(frameCount - 1, value + 1))}
              />
            </div>

            <div className="mt-6 space-y-3 rounded-2xl border border-stone-800 bg-black/20 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-stone-400">
                    Review notes
                  </p>
                  <p className="text-sm text-stone-300">
                    Save reviewer annotations under <code>{annotationPath ?? annotationFileLabel(filename)}</code>
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-stone-400">
                  {annotationSavedAt && <span>saved at {annotationSavedAt}</span>}
                  <button
                    type="button"
                    onClick={() => void handleSaveAnnotation()}
                    disabled={annotationLoading || annotationSaving || !annotationDirty}
                    className="rounded-full border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 font-medium text-amber-50 transition hover:bg-amber-500/25 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {annotationSaving ? "Saving…" : annotationDirty ? "Save notes" : "Saved"}
                  </button>
                </div>
              </div>
              <textarea
                value={annotationText}
                onChange={(event) => setAnnotationText(event.target.value)}
                rows={5}
                placeholder="Write review notes, suspicious moves, frame-zero issues, or fixes to make later."
                className="min-h-[128px] w-full rounded-2xl border border-stone-700 bg-stone-950/60 px-4 py-3 text-sm text-stone-100 outline-none placeholder:text-stone-500 focus:border-amber-400/60"
              />
              {annotationError && <p className="text-sm text-red-300">{annotationError}</p>}
            </div>
          </section>

          <section className="rounded-3xl border bg-card p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h4 className="text-base font-semibold">Frame strip</h4>
                <p className="text-sm text-muted-foreground">
                  Small squares for quick visual validation and move alignment.
                </p>
              </div>
              <p className="text-xs text-muted-foreground">Click any frame to inspect it above</p>
            </div>
            <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8 xl:grid-cols-10">
              {Array.from({ length: frameCount }, (_, frameIndex) => {
                const isSelected = frameIndex === selectedFrame;
                const moveText = moveByFrame.get(frameIndex);
                return (
                  <button
                    key={frameIndex}
                    type="button"
                    onClick={() => setSelectedFrame(frameIndex)}
                    className={[
                      "group overflow-hidden rounded-2xl border text-left transition-all",
                      isSelected
                        ? "border-amber-500 ring-2 ring-amber-500/25"
                        : "border-border hover:border-amber-500/50",
                    ].join(" ")}
                  >
                    <div className="relative aspect-square bg-muted">
                      <img
                        src={clipFrameUrl(sessionId, frameIndex)}
                        alt={`Thumbnail ${frameIndex}`}
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                      <span className="absolute left-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">
                        {frameIndex}
                      </span>
                      {moveText && (
                        <span className="absolute bottom-1 left-1 right-1 truncate rounded bg-amber-500/90 px-1.5 py-0.5 text-[10px] font-medium text-black">
                          {moveText}
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        </div>

        <div className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          <section className="rounded-3xl border border-stone-800 bg-[#171411] p-4 text-stone-100 shadow-[0_16px_60px_rgba(0,0,0,0.25)]">
            <div className="mb-4">
              <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-stone-400">
                Move timeline
              </p>
              <h4 className="text-lg font-semibold text-stone-50">Detected PGN</h4>
            </div>
            <MoveTimeline
              moves={clipInfo.moves}
              selectedMoveIndex={selectedMoveIndex}
              moveRows={moveRows}
              onSelectMove={(frameIndex) => setSelectedFrame(frameIndex)}
            />
          </section>

          {dataset === "real" && (
            <SourceVideoCard
              videoSrc={sourceVideoUrl}
              videoLoading={sourceVideoLoading}
              frameTime={frameTime}
              selectedMove={selectedMove}
              hasFirstFrameMove={hasFirstFrameMove}
              setVideoRef={(node) => {
                videoRef.current = node;
              }}
              videoPlaying={videoPlaying}
              videoCurrentTime={videoCurrentTime}
              onTogglePlayback={() => void handleTogglePlayback()}
              onSync={() => syncVideoToTime(frameTime)}
              onJumpBack={() => handleJumpVideo(-1.0)}
              onJumpForward={() => handleJumpVideo(1.0)}
              onUnavailable={() => setVideoUnavailable(true)}
              onPlay={() => setVideoPlaying(true)}
              onPause={() => setVideoPlaying(false)}
              onTimeUpdate={(value) => setVideoCurrentTime(value)}
              onLoadedMetadata={() => syncVideoToTime(frameTime)}
            />
          )}

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-1">
            <MetadataCard title="Clip summary">
              <MetadataRow label="Dataset" value={datasetLabel(dataset)} />
              <MetadataRow label="Source video" value={metadata.source_video_id} />
              <MetadataRow label="DB clip id" value={metadata.source_db_clip_id} />
              <MetadataRow label="Channel" value={metadata.source_channel_handle} />
              <MetadataRow
                label="Segment start"
                value={formatMaybeSeconds(metadata.segment_start_time_seconds)}
              />
              <MetadataRow
                label="Segment end"
                value={formatMaybeSeconds(metadata.segment_end_time_seconds)}
              />
              <MetadataRow label="Sampled fps" value={formatMaybeNumber(metadata.sampled_video_fps)} />
              <MetadataRow label="Initial FEN" value={metadata.initial_board_fen} mono multiline />
              <MetadataRow label="Final FEN" value={clipInfo.final_fen} mono multiline />
              <MetadataRow label="Stored moves" value={formatMaybeNumber(metadata.num_moves)} />
              <MetadataRow
                label="Average legal moves"
                value={formatMaybeNumber(clipInfo.avg_legal_moves)}
              />
              <MetadataRow
                label="Replay"
                value={clipInfo.replay_valid ? "valid" : clipInfo.replay_error ?? "invalid"}
                multiline
              />
            </MetadataCard>

            <MetadataCard title="Tensor payload">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="py-1 pr-3 font-medium">Name</th>
                      <th className="py-1 pr-3 font-medium">Shape</th>
                      <th className="py-1 font-medium">Dtype</th>
                    </tr>
                  </thead>
                  <tbody>
                    {clipInfo.tensors.map((tensor) => (
                      <tr key={tensor.name} className="border-b last:border-0">
                        <td className="py-1 pr-3 font-mono text-xs">{tensor.name}</td>
                        <td className="py-1 pr-3 font-mono text-xs">[{tensor.shape.join(", ")}]</td>
                        <td className="py-1 text-xs text-muted-foreground">{tensor.dtype}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </MetadataCard>
          </div>
        </div>
      </div>
    </div>
  );
}

function ReviewFramePanel({
  title,
  caption,
  imageUrl,
  alt,
  onError,
}: {
  title: string;
  caption: string;
  imageUrl: string;
  alt: string;
  onError?: () => void;
}) {
  return (
    <div className="space-y-2">
      <div>
        <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-stone-400">{title}</p>
        <p className="text-sm text-stone-300">{caption}</p>
      </div>
      <div className="overflow-hidden rounded-3xl border border-stone-800 bg-black/40">
        <img
          src={imageUrl}
          alt={alt}
          className="aspect-square w-full object-contain"
          onError={onError}
        />
      </div>
    </div>
  );
}

function SourceVideoCard({
  videoSrc,
  videoLoading,
  frameTime,
  selectedMove,
  hasFirstFrameMove,
  setVideoRef,
  videoPlaying,
  videoCurrentTime,
  onTogglePlayback,
  onSync,
  onJumpBack,
  onJumpForward,
  onUnavailable,
  onPlay,
  onPause,
  onTimeUpdate,
  onLoadedMetadata,
}: {
  videoSrc: string | null;
  videoLoading: boolean;
  frameTime: number | null;
  selectedMove: DetectedMove | null;
  hasFirstFrameMove: boolean;
  setVideoRef: (node: HTMLVideoElement | null) => void;
  videoPlaying: boolean;
  videoCurrentTime: number | null;
  onTogglePlayback: () => void;
  onSync: () => void;
  onJumpBack: () => void;
  onJumpForward: () => void;
  onUnavailable: () => void;
  onPlay: () => void;
  onPause: () => void;
  onTimeUpdate: (value: number) => void;
  onLoadedMetadata: () => void;
}) {
  return (
    <section className="rounded-3xl border bg-card p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-muted-foreground">
            Source video
          </p>
          <h4 className="text-base font-semibold">Inline playback</h4>
          <p className="text-sm text-muted-foreground">
            Scrub before or after the sampled frame when you need more context.
          </p>
        </div>
        <div className="text-right text-xs text-muted-foreground">
          {frameTime !== null && <p>target {formatSeconds(frameTime)}</p>}
          {videoCurrentTime !== null && <p>current {formatSeconds(videoCurrentTime)}</p>}
          {selectedMove && <p>{selectedMove.san || selectedMove.uci}</p>}
        </div>
      </div>

      {hasFirstFrameMove && (
        <div className="mb-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-200">
          This clip starts with a move on frame 0. Jump backward in the source video to inspect the lead-in before deciding whether the clip needs regeneration.
        </div>
      )}

      <div className="overflow-hidden rounded-2xl border bg-black">
        {videoSrc ? (
          <video
            ref={setVideoRef}
            src={videoSrc}
            controls
            playsInline
            preload="metadata"
            className="aspect-video w-full bg-black"
            onError={onUnavailable}
            onPlay={onPlay}
            onPause={onPause}
            onTimeUpdate={(event) => onTimeUpdate(event.currentTarget.currentTime)}
            onLoadedMetadata={onLoadedMetadata}
          />
        ) : (
          <div className="flex aspect-video items-center justify-center text-sm text-stone-300">
            {videoLoading ? "Preparing review video…" : "Source video unavailable."}
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <VideoControlButton label={videoPlaying ? "Pause" : "Play"} onClick={onTogglePlayback} />
        <VideoControlButton label="Sync to frame" onClick={onSync} />
        <VideoControlButton label="-1s" onClick={onJumpBack} />
        <VideoControlButton label="+1s" onClick={onJumpForward} />
      </div>
    </section>
  );
}

function FrameNavButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded-full border border-stone-700 px-3 py-1.5 text-sm text-stone-100 transition hover:border-amber-400/60 hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {label}
    </button>
  );
}

function VideoControlButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-full border px-3 py-1.5 text-sm transition hover:border-foreground hover:text-foreground"
    >
      {label}
    </button>
  );
}

function MoveTimeline({
  moves,
  selectedMoveIndex,
  moveRows,
  onSelectMove,
}: {
  moves: DetectedMove[];
  selectedMoveIndex: number;
  moveRows: Array<{ moveNumber: number; white?: MoveEntry; black?: MoveEntry }>;
  onSelectMove: (frameIndex: number) => void;
}) {
  const activeButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    activeButtonRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedMoveIndex]);

  if (moves.length === 0) {
    return <p className="text-sm text-stone-400">No moves stored in this clip.</p>;
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-stone-800">
      {moveRows.map((row) => (
        <div
          key={row.moveNumber}
          className="grid grid-cols-[56px_minmax(0,1fr)_minmax(0,1fr)] border-b border-stone-800/80 last:border-b-0"
        >
          <div className="flex items-center justify-center bg-white/[0.03] px-3 py-3 text-lg font-semibold text-stone-400">
            {row.moveNumber}.
          </div>
          <MoveCell
            entry={row.white}
            selectedMoveIndex={selectedMoveIndex}
            onSelectMove={onSelectMove}
            setActiveButtonRef={(node) => {
              activeButtonRef.current = node;
            }}
          />
          <MoveCell
            entry={row.black}
            selectedMoveIndex={selectedMoveIndex}
            onSelectMove={onSelectMove}
            setActiveButtonRef={(node) => {
              activeButtonRef.current = node;
            }}
          />
        </div>
      ))}
    </div>
  );
}

type MoveEntry = {
  index: number;
  move: DetectedMove;
};

function MoveCell({
  entry,
  selectedMoveIndex,
  onSelectMove,
  setActiveButtonRef,
}: {
  entry?: MoveEntry;
  selectedMoveIndex: number;
  onSelectMove: (frameIndex: number) => void;
  setActiveButtonRef: (node: HTMLButtonElement | null) => void;
}) {
  if (!entry) {
    return <div className="border-l border-stone-800/80 bg-transparent" />;
  }

  const isSelected = entry.index === selectedMoveIndex;
  return (
    <button
      ref={isSelected ? setActiveButtonRef : undefined}
      type="button"
      onClick={() => onSelectMove(entry.move.frame_index)}
      className={[
        "flex min-w-0 flex-col items-start gap-1 border-l border-stone-800/80 px-4 py-3 text-left transition",
        isSelected
          ? "bg-amber-400/22 text-amber-50 ring-1 ring-inset ring-amber-300/70 shadow-[inset_0_0_0_1px_rgba(251,191,36,0.18)]"
          : "bg-transparent text-stone-100 hover:bg-white/[0.04]",
      ].join(" ")}
    >
      <span className="truncate text-xl font-semibold leading-none">
        {entry.move.san || entry.move.uci}
      </span>
      <span className="text-xs text-stone-400">
        f{entry.move.frame_index}
        {entry.move.timestamp_seconds !== null
          ? ` • ${formatSeconds(entry.move.timestamp_seconds)}`
          : ""}
      </span>
    </button>
  );
}

function MetadataCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{title}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">{children}</CardContent>
    </Card>
  );
}

function MetadataRow({
  label,
  value,
  mono = false,
  multiline = false,
}: {
  label: string;
  value: string | number | boolean | null | undefined;
  mono?: boolean;
  multiline?: boolean;
}) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  return (
    <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={[
          mono ? "font-mono text-xs" : "",
          multiline ? "whitespace-pre-wrap break-words" : "truncate",
        ].join(" ")}
        title={typeof value === "string" ? value : undefined}
      >
        {String(value)}
      </span>
    </div>
  );
}

function ClipReviewSkeleton({ dataset, filename }: { dataset: DatasetKind; filename: string }) {
  return (
    <div className="space-y-4">
      <Link href={`/data/${dataset}`} className="text-sm underline underline-offset-4">
        ← Back to {datasetLabel(dataset)} clips
      </Link>
      <div className="space-y-2">
        <h3 className="text-2xl font-semibold tracking-tight">{filename}</h3>
        <p className="text-sm text-muted-foreground">Loading clip review…</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(300px,380px)]">
        <div className="space-y-4">
          <Skeleton className="h-[720px] w-full rounded-3xl" />
          <Skeleton className="h-[360px] w-full rounded-3xl" />
        </div>
        <div className="space-y-4">
          <Skeleton className="h-[300px] w-full rounded-3xl" />
          <Skeleton className="h-[300px] w-full rounded-3xl" />
          <Skeleton className="h-[240px] w-full rounded-3xl" />
        </div>
      </div>
    </div>
  );
}

function groupMoves(moves: DetectedMove[]) {
  const rows: Array<{ moveNumber: number; white?: MoveEntry; black?: MoveEntry }> = [];
  for (let index = 0; index < moves.length; index += 2) {
    rows.push({
      moveNumber: Math.floor(index / 2) + 1,
      white: { index, move: moves[index] },
      black: moves[index + 1] ? { index: index + 1, move: moves[index + 1] } : undefined,
    });
  }
  return rows;
}

function moveLabel(index: number, move: DetectedMove): string {
  const moveNumber = Math.floor(index / 2) + 1;
  const san = move.san || move.uci;
  return index % 2 === 0 ? `${moveNumber}.${san}` : `${moveNumber}...${san}`;
}

function datasetLabel(dataset: DatasetKind): string {
  return dataset === "real" ? "Real footage" : "Synthetic";
}

function annotationFileLabel(filename: string): string {
  const stem = filename.replace(/\.pt$/i, "");
  return `data/clip_annotations/${stem}.txt`;
}

function formatSeconds(value: number): string {
  return `${value.toFixed(1)}s`;
}

function formatMaybeSeconds(value: string | number | boolean | null | undefined): string | null {
  if (typeof value !== "number") {
    return null;
  }
  return formatSeconds(value);
}

function formatMaybeNumber(value: string | number | boolean | null | undefined): string | null {
  if (typeof value !== "number") {
    return null;
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
