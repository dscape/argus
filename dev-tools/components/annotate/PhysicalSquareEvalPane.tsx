"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import {
  clipFrameUrl,
  deleteClipSession,
  getClipInfo,
  getPhysicalEvalAnnotation,
  getPhysicalEvalSummary,
  listPhysicalEvalClips,
  loadClipFromPath,
  rectifyPhysicalEvalFrame,
  savePhysicalEvalAnnotation,
  type PhysicalEvalAnnotation,
  type PhysicalEvalClip,
  type PhysicalEvalSummary,
} from "@/lib/api";
import type { ClipInspectResponse } from "@/lib/types";

const SQUARE_CLASS_NAMES = [
  "empty",
  "P",
  "N",
  "B",
  "R",
  "Q",
  "K",
  "p",
  "n",
  "b",
  "r",
  "q",
  "k",
] as const;

const DEFAULT_OUTPUT_SIZE = 512;

function emptyLabels(): Array<number | null> {
  return Array.from({ length: 64 }, () => null);
}

function normalizeLabels(labels: Array<number | null> | null | undefined): Array<number | null> {
  if (!labels || labels.length !== 64) {
    return emptyLabels();
  }
  return labels.map((label) => (typeof label === "number" ? label : null));
}

function labelToken(label: number | null): string {
  if (label == null) return "";
  if (label === 0) return "·";
  return SQUARE_CLASS_NAMES[label];
}

function markerLabel(index: number): string {
  return ["TL", "TR", "BR", "BL"][index] ?? String(index + 1);
}

function squareName(squareIndex: number): string {
  const row = Math.floor(squareIndex / 8);
  const col = squareIndex % 8;
  return `${String.fromCharCode("a".charCodeAt(0) + col)}${8 - row}`;
}

export default function PhysicalSquareEvalPane() {
  const [clips, setClips] = useState<PhysicalEvalClip[]>([]);
  const [summary, setSummary] = useState<PhysicalEvalSummary | null>(null);
  const [selectedClipPath, setSelectedClipPath] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<ClipInspectResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [corners, setCorners] = useState<Array<{ x: number; y: number }>>([]);
  const [labels, setLabels] = useState<Array<number | null>>(emptyLabels);
  const [activeLabel, setActiveLabel] = useState<number | null>(1);
  const [rectifiedImageB64, setRectifiedImageB64] = useState<string | null>(null);
  const [sourceImageSize, setSourceImageSize] = useState<{ width: number; height: number } | null>(null);
  const [loadingClip, setLoadingClip] = useState(false);
  const [rectifying, setRectifying] = useState(false);
  const [saving, setSaving] = useState(false);
  const [existingAnnotation, setExistingAnnotation] = useState<PhysicalEvalAnnotation | null>(null);

  const loadSummary = useCallback(async () => {
    try {
      setSummary(await getPhysicalEvalSummary());
    } catch (error) {
      console.warn("Failed to load physical eval summary", error);
    }
  }, []);

  const loadClips = useCallback(async () => {
    try {
      const data = await listPhysicalEvalClips();
      setClips(data.clips);
      if (data.clips.length > 0) {
        setSelectedClipPath((previous) => previous || data.clips[0].clip_path);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load real clips");
    }
  }, []);

  useEffect(() => {
    void loadClips();
    void loadSummary();
  }, [loadClips, loadSummary]);

  useEffect(() => {
    if (!selectedClipPath) {
      setClipInfo(null);
      setSessionId(null);
      return;
    }

    let cancelled = false;
    let loadedSessionId: string | null = null;

    setLoadingClip(true);
    setClipInfo(null);
    setFrameIndex(0);
    setCorners([]);
    setLabels(emptyLabels());
    setRectifiedImageB64(null);
    setExistingAnnotation(null);
    setSourceImageSize(null);

    void (async () => {
      try {
        const session = await loadClipFromPath(selectedClipPath);
        loadedSessionId = session.session_id;
        if (cancelled) {
          await deleteClipSession(session.session_id);
          return;
        }

        setSessionId(session.session_id);
        const info = await getClipInfo(session.session_id);
        if (cancelled) {
          await deleteClipSession(session.session_id);
          return;
        }
        setClipInfo(info);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to load clip");
      } finally {
        if (!cancelled) {
          setLoadingClip(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      if (loadedSessionId) {
        void deleteClipSession(loadedSessionId).catch(() => undefined);
      }
    };
  }, [selectedClipPath]);

  const rectifyBoard = useCallback(
    async (nextCorners: Array<{ x: number; y: number }>) => {
      if (!sessionId || nextCorners.length !== 4) {
        return;
      }

      setRectifying(true);
      try {
        const result = await rectifyPhysicalEvalFrame({
          session_id: sessionId,
          frame_index: frameIndex,
          corners: nextCorners.map((point) => [point.x, point.y]),
          output_size: DEFAULT_OUTPUT_SIZE,
        });
        setRectifiedImageB64(result.image_b64);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to rectify board");
      } finally {
        setRectifying(false);
      }
    },
    [frameIndex, sessionId],
  );

  useEffect(() => {
    if (!selectedClipPath) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const annotation = await getPhysicalEvalAnnotation(selectedClipPath, frameIndex);
        if (cancelled) {
          return;
        }
        setExistingAnnotation(annotation);
        if (!annotation) {
          setCorners([]);
          setLabels(emptyLabels());
          setRectifiedImageB64(null);
          return;
        }

        const nextCorners = annotation.corners.map(([x, y]) => ({ x, y }));
        setCorners(nextCorners);
        setLabels(normalizeLabels(annotation.labels));
        if (sessionId) {
          await rectifyBoard(nextCorners);
        }
      } catch (error) {
        if (!cancelled) {
          toast.error(error instanceof Error ? error.message : "Failed to load saved annotation");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [frameIndex, rectifyBoard, selectedClipPath, sessionId]);

  const currentClip = useMemo(
    () => clips.find((clip) => clip.clip_path === selectedClipPath) ?? null,
    [clips, selectedClipPath],
  );

  const frameUrl = sessionId ? clipFrameUrl(sessionId, frameIndex) : null;
  const labeledCount = useMemo(
    () => labels.filter((label) => label !== null).length,
    [labels],
  );
  const canRectify = sessionId != null && corners.length === 4;
  const canSave = canRectify && labeledCount > 0;
  const frameTimestamp = clipInfo?.frame_timestamps_seconds?.[frameIndex] ?? null;

  const handleSourceImageClick = useCallback(
    async (event: React.MouseEvent<HTMLImageElement>) => {
      if (!sourceImageSize || corners.length >= 4) {
        return;
      }

      const rect = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * sourceImageSize.width;
      const y = ((event.clientY - rect.top) / rect.height) * sourceImageSize.height;
      const nextCorners = [...corners, { x, y }];
      setCorners(nextCorners);
      if (nextCorners.length === 4) {
        await rectifyBoard(nextCorners);
      }
    },
    [corners, rectifyBoard, sourceImageSize],
  );

  const handleSquareClick = useCallback((squareIndex: number) => {
    setLabels((previous) => {
      const next = [...previous];
      next[squareIndex] = activeLabel;
      return next;
    });
  }, [activeLabel]);

  const handleResetCurrentFrame = useCallback(() => {
    setCorners([]);
    setLabels(emptyLabels());
    setRectifiedImageB64(null);
    setExistingAnnotation(null);
  }, []);

  const handleSave = useCallback(async () => {
    if (!sessionId || !selectedClipPath || corners.length !== 4) {
      return;
    }

    setSaving(true);
    try {
      const result = await savePhysicalEvalAnnotation({
        session_id: sessionId,
        clip_path: selectedClipPath,
        frame_index: frameIndex,
        corners: corners.map((point) => [point.x, point.y]),
        labels,
        output_size: DEFAULT_OUTPUT_SIZE,
      });
      setExistingAnnotation(result.annotation);
      setSummary(result.summary);
      toast.success(`Saved ${result.annotation.labeled_square_count} labeled squares`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save annotation");
    } finally {
      setSaving(false);
    }
  }, [corners, frameIndex, labels, selectedClipPath, sessionId]);

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-4">
        <div className="rounded border p-4">
          <div className="text-sm font-medium">Held-out eval set</div>
          <div className="mt-2 text-2xl font-semibold">{summary?.square_crop_count ?? 0}</div>
          <div className="text-xs text-muted-foreground">labeled square crops saved under {summary?.dataset_root ?? "data/physical/eval"}</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm font-medium">Annotated boards</div>
          <div className="mt-2 text-2xl font-semibold">{summary?.board_annotation_count ?? 0}</div>
          <div className="text-xs text-muted-foreground">One board frame can yield up to 64 labeled squares.</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm font-medium">Source videos</div>
          <div className="mt-2 text-2xl font-semibold">{summary?.source_video_count ?? 0}</div>
          <div className="text-xs text-muted-foreground">Keep this split held out from all physical-board training.</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm font-medium">Current frame</div>
          <div className="mt-2 text-2xl font-semibold">{labeledCount}</div>
          <div className="text-xs text-muted-foreground">labeled squares on the current rectified board</div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-4 rounded border p-4">
          <div>
            <h2 className="text-lg font-semibold">1. Choose a real clip</h2>
            <p className="text-sm text-muted-foreground">
              Source frames come from physical camera crops in <code>data/argus/train_real</code>.
            </p>
          </div>

          <label className="space-y-2 text-sm block">
            <span className="font-medium">Clip</span>
            <select
              className="w-full rounded border bg-background px-3 py-2"
              value={selectedClipPath}
              onChange={(event) => setSelectedClipPath(event.target.value)}
            >
              <option value="">Select a clip</option>
              {clips.map((clip) => (
                <option key={clip.clip_path} value={clip.clip_path}>
                  {clip.filename}
                </option>
              ))}
            </select>
          </label>

          {currentClip && (
            <div className="rounded border bg-muted/30 p-3 text-sm">
              <div><span className="font-medium">Video:</span> {currentClip.source_video_id ?? "unknown"}</div>
              <div><span className="font-medium">Clip id:</span> {currentClip.clip_id ?? "-"}</div>
              <div><span className="font-medium">Size:</span> {currentClip.size_mb.toFixed(2)} MB</div>
            </div>
          )}

          <div className="space-y-2 text-sm">
            <div className="font-medium">Class histogram</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {SQUARE_CLASS_NAMES.map((label) => (
                <div key={label} className="flex items-center justify-between rounded border px-2 py-1">
                  <span>{label}</span>
                  <span>{summary?.class_counts?.[label] ?? 0}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4 rounded border p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold">2. Mark board corners</h2>
              <p className="text-sm text-muted-foreground">
                Click the source frame in order: top-left, top-right, bottom-right, bottom-left.
              </p>
            </div>
            {existingAnnotation && (
              <div className="rounded border border-green-600/40 bg-green-600/10 px-3 py-1 text-xs text-green-700 dark:text-green-300">
                Saved for this frame
              </div>
            )}
          </div>

          {clipInfo ? (
            <div className="space-y-3">
              <div>
                <input
                  type="range"
                  min={0}
                  max={Math.max(clipInfo.num_frames - 1, 0)}
                  value={frameIndex}
                  onChange={(event) => setFrameIndex(Number(event.target.value))}
                  className="w-full"
                />
                <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                  <span>Frame {frameIndex + 1} / {clipInfo.num_frames}</span>
                  <span>{frameTimestamp !== null ? `${frameTimestamp.toFixed(2)}s` : ""}</span>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  className="rounded border px-3 py-2 text-sm"
                  onClick={handleResetCurrentFrame}
                >
                  Reset frame
                </button>
                <button
                  type="button"
                  className="rounded border px-3 py-2 text-sm disabled:opacity-50"
                  disabled={!canRectify || rectifying}
                  onClick={() => void rectifyBoard(corners)}
                >
                  {rectifying ? "Rectifying…" : "Rectify board"}
                </button>
              </div>

              <div className="relative overflow-hidden rounded border bg-muted/20">
                {frameUrl && (
                  <img
                    src={frameUrl}
                    alt="Physical board candidate"
                    className="block h-auto w-full cursor-crosshair"
                    onLoad={(event) => {
                      setSourceImageSize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      });
                    }}
                    onClick={(event) => void handleSourceImageClick(event)}
                  />
                )}
                {sourceImageSize && corners.map((point, index) => (
                  <div
                    key={`${point.x}-${point.y}-${index}`}
                    className="absolute -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${(point.x / sourceImageSize.width) * 100}%`,
                      top: `${(point.y / sourceImageSize.height) * 100}%`,
                    }}
                  >
                    <div className="rounded-full bg-red-600 px-2 py-1 text-[10px] font-semibold text-white shadow">
                      {markerLabel(index)}
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-xs text-muted-foreground">
                Corner count: {corners.length} / 4
              </div>
            </div>
          ) : (
            <div className="rounded border border-dashed p-6 text-sm text-muted-foreground">
              {loadingClip ? "Loading clip…" : "Select a real clip to begin."}
            </div>
          )}
        </div>

        <div className="space-y-4 rounded border p-4">
          <div>
            <h2 className="text-lg font-semibold">3. Label held-out squares</h2>
            <p className="text-sm text-muted-foreground">
              Label only squares you are sure about. Unset squares are ignored when saving the eval set.
            </p>
          </div>

          <div className="grid grid-cols-4 gap-2 text-xs">
            <button
              type="button"
              className={`rounded border px-2 py-2 ${activeLabel === null ? "border-foreground bg-foreground text-background" : ""}`}
              onClick={() => setActiveLabel(null)}
            >
              unset
            </button>
            {SQUARE_CLASS_NAMES.map((label, index) => (
              <button
                key={label}
                type="button"
                className={`rounded border px-2 py-2 ${activeLabel === index ? "border-foreground bg-foreground text-background" : ""}`}
                onClick={() => setActiveLabel(index)}
              >
                {index === 0 ? "empty" : label}
              </button>
            ))}
          </div>

          <div className="rounded border bg-muted/20 p-3 text-xs text-muted-foreground">
            Active label: <span className="font-medium text-foreground">{activeLabel === null ? "unset" : SQUARE_CLASS_NAMES[activeLabel]}</span>
          </div>

          <div className="relative aspect-square overflow-hidden rounded border bg-muted/20">
            {rectifiedImageB64 ? (
              <>
                <img
                  src={`data:image/png;base64,${rectifiedImageB64}`}
                  alt="Rectified physical board"
                  className="block h-full w-full object-cover"
                />
                <div className="absolute inset-0 grid grid-cols-8 grid-rows-8">
                  {labels.map((label, squareIndex) => (
                    <button
                      key={squareIndex}
                      type="button"
                      className={`relative border border-white/25 text-xs font-semibold text-white hover:bg-blue-500/20 ${label !== null ? "bg-black/20" : ""}`}
                      onClick={() => handleSquareClick(squareIndex)}
                      title={`${squareName(squareIndex)} — ${label === null ? "unset" : SQUARE_CLASS_NAMES[label]}`}
                    >
                      <span className="pointer-events-none absolute left-1 top-1 rounded bg-black/60 px-1 text-[10px]">
                        {squareName(squareIndex)}
                      </span>
                      <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[10px]">
                        {labelToken(label)}
                      </span>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
                {rectifying ? "Rectifying board…" : "Mark four corners to generate the rectified board preview."}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between rounded border bg-muted/20 p-3 text-sm">
            <span>{labeledCount} labeled squares on this frame</span>
            <button
              type="button"
              className="rounded border px-3 py-2 text-sm disabled:opacity-50"
              disabled={!canSave || saving}
              onClick={() => void handleSave()}
            >
              {saving ? "Saving…" : "Save held-out labels"}
            </button>
          </div>
        </div>
      </section>

      <section className="rounded border p-4">
        <h2 className="text-lg font-semibold">Recent saved annotations</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {(summary?.recent_annotations ?? []).map((annotation) => (
            <div key={annotation.annotation_id} className="rounded border bg-muted/20 p-3 text-sm">
              <div className="font-medium">{annotation.annotation_id}</div>
              <div className="text-xs text-muted-foreground">{annotation.clip_path}</div>
              <div className="mt-2 text-xs">frame {annotation.frame_index} · {annotation.labeled_square_count} labeled squares</div>
            </div>
          ))}
          {(summary?.recent_annotations?.length ?? 0) === 0 && (
            <div className="text-sm text-muted-foreground">No saved physical-board annotations yet.</div>
          )}
        </div>
      </section>
    </div>
  );
}
