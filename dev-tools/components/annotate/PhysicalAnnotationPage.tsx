"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ChessBoard } from "@/components/ChessBoard";
import { chessPieceSvg } from "@/components/chess-pieces";
import { Badge } from "@/components/ui/badge";
import {
  clipFrameUrl,
  clipSourceVideoUrl,
  deleteClipSession,
  deletePhysicalEvalAnnotation,
  deletePhysicalTrainAnnotation,
  getClipInfo,
  getPhysicalEvalAnnotation,
  getPhysicalEvalMoveCorrections,
  getPhysicalTrainAnnotation,
  getPhysicalTrainMoveCorrections,
  loadClipFromPath,
  rectifyPhysicalEvalFrame,
  rectifyPhysicalTrainFrame,
  savePhysicalEvalAnnotation,
  savePhysicalTrainAnnotation,
  type PhysicalEvalAnnotation,
  type PhysicalEvalMoveCorrections,
} from "@/lib/api";
import type { ClipInspectResponse, DetectedMove } from "@/lib/types";

const SQUARE_CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"] as const;
const LABEL_COUNT = SQUARE_CLASS_NAMES.length; // 13
const NAV_BUTTON_CLASS = "inline-flex h-9 items-center justify-center rounded border px-2 text-xs whitespace-nowrap disabled:opacity-40";
const NAV_MOVE_BUTTON_CLASS = `${NAV_BUTTON_CLASS} w-24`;
const NAV_STEP_BUTTON_CLASS = `${NAV_BUTTON_CLASS} w-10 font-medium`;
const NAV_COUNTER_CLASS = "inline-flex h-9 w-20 items-center justify-center text-xs font-medium tabular-nums";
const PANEL_BUTTON_CLASS = "inline-flex h-5 items-center justify-center rounded border px-1.5 text-[9px] leading-none whitespace-nowrap disabled:opacity-40";
const PANEL_STATUS_CLASS = "inline-flex h-5 min-w-[5rem] items-center justify-center rounded border px-1.5 text-[9px] leading-none whitespace-nowrap";
const PRIMARY_SAVE_BUTTON_CLASS = "inline-flex h-6 min-w-[8.75rem] items-center justify-center rounded border bg-primary px-2.5 text-[9px] font-medium text-primary-foreground whitespace-nowrap transition-colors hover:bg-primary/90 disabled:opacity-40";
const LIVE_RECTIFY_DELAY_MS = 75;

function fenToLabels(fen: string): Array<number | null> {
  const placement = fen.split(" ", 1)[0];
  const map = new Map<string, number>(SQUARE_CLASS_NAMES.map((l, i) => [l, i]));
  const labels: Array<number | null> = [];
  for (const rank of placement.split("/")) {
    for (const ch of rank) {
      if (/^[1-8]$/.test(ch)) { for (let i = 0; i < Number(ch); i++) labels.push(0); continue; }
      const idx = map.get(ch);
      if (idx == null) throw new Error(`Unsupported FEN piece: ${ch}`);
      labels.push(idx);
    }
  }
  if (labels.length !== 64) throw new Error(`Expected 64 squares, got ${labels.length}`);
  return labels;
}

function squareName(idx: number): string {
  return `${String.fromCharCode(97 + (idx % 8))}${8 - Math.floor(idx / 8)}`;
}

function cornerLabel(idx: number): string {
  return ["a8", "h8", "h1", "a1"][idx] ?? String(idx + 1);
}

function labelToken(label: number | null): string {
  if (label == null) return "";
  if (label === 0) return "\u00B7";
  return SQUARE_CLASS_NAMES[label];
}

function emptyLabels(): Array<number | null> {
  return Array.from({ length: 64 }, () => null);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function cycleLabel(current: number | null): number | null {
  if (current === null) return 0;
  const next = current + 1;
  return next >= LABEL_COUNT ? null : next;
}

function moveLabel(index: number, move: DetectedMove): string {
  return `#${index + 1} ${move.san || move.uci}`;
}

function normalizeAnnotationLabels(labels: Array<number | null>): Array<number | null> {
  return labels.map((label) => (typeof label === "number" ? label : null));
}

function matchesSavedCorners(
  corners: Array<{ x: number; y: number }>,
  savedCorners: number[][],
): boolean {
  return (
    corners.length === savedCorners.length
    && corners.every(
      (corner, index) => corner.x === savedCorners[index]?.[0] && corner.y === savedCorners[index]?.[1],
    )
  );
}

function matchesSavedLabels(labels: Array<number | null>, savedLabels: Array<number | null>): boolean {
  return labels.length === savedLabels.length && labels.every((label, index) => label === savedLabels[index]);
}

type PhysicalAnnotationSplit = "val" | "train";

interface PhysicalAnnotationSplitConfig {
  indexHref: string;
  indexLabel: string;
  splitLabel: string;
  getAnnotation: typeof getPhysicalEvalAnnotation;
  getMoveCorrections: typeof getPhysicalEvalMoveCorrections;
  rectifyFrame: typeof rectifyPhysicalEvalFrame;
  saveAnnotation: typeof savePhysicalEvalAnnotation;
  deleteAnnotation: typeof deletePhysicalEvalAnnotation;
}

const SPLIT_CONFIG: Record<PhysicalAnnotationSplit, PhysicalAnnotationSplitConfig> = {
  val: {
    indexHref: "/annotate/physical?split=val",
    indexLabel: "All validation clips",
    splitLabel: "Validation",
    getAnnotation: getPhysicalEvalAnnotation,
    getMoveCorrections: getPhysicalEvalMoveCorrections,
    rectifyFrame: rectifyPhysicalEvalFrame,
    saveAnnotation: savePhysicalEvalAnnotation,
    deleteAnnotation: deletePhysicalEvalAnnotation,
  },
  train: {
    indexHref: "/annotate/physical?split=train",
    indexLabel: "All training clips",
    splitLabel: "Train",
    getAnnotation: getPhysicalTrainAnnotation,
    getMoveCorrections: getPhysicalTrainMoveCorrections,
    rectifyFrame: rectifyPhysicalTrainFrame,
    saveAnnotation: savePhysicalTrainAnnotation,
    deleteAnnotation: deletePhysicalTrainAnnotation,
  },
};

interface Props {
  filename: string;
  split?: PhysicalAnnotationSplit;
}

export function PhysicalAnnotationPage({ filename, split = "val" }: Props) {
  const clipPath = `data/argus/train_real/${filename}`;
  const splitConfig = SPLIT_CONFIG[split];
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<ClipInspectResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let sid: string | null = null;
    setLoading(true);
    setError(null);

    void (async () => {
      try {
        const session = await loadClipFromPath(clipPath);
        sid = session.session_id;
        if (cancelled) { await deleteClipSession(sid).catch(() => {}); return; }
        setSessionId(sid);
        const info = await getClipInfo(sid);
        if (cancelled) { await deleteClipSession(sid).catch(() => {}); return; }
        setClipInfo(info);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load clip");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      if (sid) void deleteClipSession(sid).catch(() => {});
    };
  }, [clipPath]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Link href={splitConfig.indexHref} className="text-sm text-blue-500 hover:underline">
          &larr; {splitConfig.indexLabel}
        </Link>
        <p className="text-sm text-muted-foreground">Loading {filename}&hellip;</p>
      </div>
    );
  }

  if (error || !clipInfo || !sessionId) {
    return (
      <div className="space-y-4">
        <Link href={splitConfig.indexHref} className="text-sm text-blue-500 hover:underline">
          &larr; {splitConfig.indexLabel}
        </Link>
        <p className="text-sm text-destructive">{error ?? "Failed to load clip"}</p>
      </div>
    );
  }

  return (
    <AnnotationContent
      filename={filename}
      clipPath={clipPath}
      sessionId={sessionId}
      clipInfo={clipInfo}
      splitConfig={splitConfig}
    />
  );
}

function AnnotationContent({
  filename,
  clipPath,
  sessionId,
  clipInfo,
  splitConfig,
}: {
  filename: string;
  clipPath: string;
  sessionId: string;
  clipInfo: ClipInspectResponse;
  splitConfig: PhysicalAnnotationSplitConfig;
}) {
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [corners, setCorners] = useState<Array<{ x: number; y: number }>>([]);
  const [boardLabels, setBoardLabels] = useState<Array<number | null>>(emptyLabels);
  const [rectifiedImageB64, setRectifiedImageB64] = useState<string | null>(null);
  const [sourceImageSize, setSourceImageSize] = useState<{ width: number; height: number } | null>(null);
  const [rectifying, setRectifying] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [existingAnnotation, setExistingAnnotation] = useState<PhysicalEvalAnnotation | null>(null);
  const [moveCorrections, setMoveCorrections] = useState<PhysicalEvalMoveCorrections | null>(null);

  // Source video
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [videoUnavailable, setVideoUnavailable] = useState(false);
  const [draggedCornerIndex, setDraggedCornerIndex] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const cornersRef = useRef<Array<{ x: number; y: number }>>([]);
  const draggedCornerIndexRef = useRef<number | null>(null);
  const latestRectifyRequestIdRef = useRef(0);
  const liveRectifyTimerRef = useRef<number | null>(null);
  const queuedRectifyCornersRef = useRef<Array<{ x: number; y: number }> | null>(null);
  const rectifyInFlightRef = useRef(false);

  const frameCount = clipInfo.num_frames;
  const frameTime = clipInfo.frame_timestamps_seconds[selectedFrame] ?? null;

  const refreshMoveCorrections = useCallback(async () => {
    try {
      const nextCorrections = await splitConfig.getMoveCorrections(sessionId, clipPath);
      setMoveCorrections(nextCorrections);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load manual move corrections");
    }
  }, [clipPath, sessionId]);

  useEffect(() => {
    void refreshMoveCorrections();
  }, [refreshMoveCorrections]);

  const effectiveFrameReplayFens = moveCorrections?.frame_replay_fens ?? clipInfo.frame_replay_fens;
  const effectiveMoves = moveCorrections?.moves ?? clipInfo.moves;
  const effectiveTotalMoves = moveCorrections?.total_moves ?? clipInfo.total_moves;
  const replayFen = effectiveFrameReplayFens[selectedFrame] ?? null;

  // Load source video
  useEffect(() => {
    const controller = new AbortController();
    let blobUrl: string | null = null;
    setSourceVideoUrl(null);
    setVideoUnavailable(false);

    fetch(clipSourceVideoUrl(sessionId), { signal: controller.signal })
      .then(async (res) => { if (!res.ok) throw new Error(); return res.blob(); })
      .then((blob) => { blobUrl = URL.createObjectURL(blob); setSourceVideoUrl(blobUrl); })
      .catch(() => { if (!controller.signal.aborted) setVideoUnavailable(true); });

    return () => { controller.abort(); if (blobUrl) URL.revokeObjectURL(blobUrl); };
  }, [sessionId]);

  // Sync video to frame time
  useEffect(() => {
    const video = videoRef.current;
    if (!video || frameTime === null) return;
    video.currentTime = frameTime;
  }, [frameTime]);

  const selectedMoveIndex = useMemo(
    () => effectiveMoves.findIndex((m) => m.frame_index === selectedFrame),
    [effectiveMoves, selectedFrame],
  );
  const selectedMove = selectedMoveIndex >= 0 ? effectiveMoves[selectedMoveIndex] : null;
  const displayedReplayFen = selectedMove?.fen_after ?? replayFen;
  const previousFrameDisplayedFen = useMemo(() => {
    if (selectedFrame <= 0) return null;
    const previousMove = effectiveMoves.find((move) => move.frame_index === selectedFrame - 1);
    return previousMove?.fen_after ?? effectiveFrameReplayFens[selectedFrame - 1] ?? null;
  }, [effectiveFrameReplayFens, effectiveMoves, selectedFrame]);
  const nextFrameDisplayedFen = useMemo(() => {
    if (selectedFrame >= frameCount - 1) return null;
    const nextMove = effectiveMoves.find((move) => move.frame_index === selectedFrame + 1);
    return nextMove?.fen_after ?? effectiveFrameReplayFens[selectedFrame + 1] ?? null;
  }, [effectiveFrameReplayFens, effectiveMoves, frameCount, selectedFrame]);

  const previousMoveFrame = useMemo(() => {
    for (let i = effectiveMoves.length - 1; i >= 0; i--) {
      if (effectiveMoves[i].frame_index < selectedFrame) return effectiveMoves[i].frame_index;
    }
    return null;
  }, [effectiveMoves, selectedFrame]);

  const nextMoveFrame = useMemo(() => {
    for (const m of effectiveMoves) {
      if (m.frame_index > selectedFrame) return m.frame_index;
    }
    return null;
  }, [effectiveMoves, selectedFrame]);

  const boardLabeledCount = boardLabels.filter((l) => l !== null).length;
  const canSave = corners.length === 4 && boardLabeledCount > 0;

  useEffect(() => {
    cornersRef.current = corners;
  }, [corners]);

  useEffect(() => {
    latestRectifyRequestIdRef.current += 1;
    draggedCornerIndexRef.current = null;
    setDraggedCornerIndex(null);
    queuedRectifyCornersRef.current = null;
    rectifyInFlightRef.current = false;
    if (liveRectifyTimerRef.current !== null) {
      window.clearTimeout(liveRectifyTimerRef.current);
      liveRectifyTimerRef.current = null;
    }
  }, [clipPath, selectedFrame]);

  useEffect(() => {
    return () => {
      if (liveRectifyTimerRef.current !== null) {
        window.clearTimeout(liveRectifyTimerRef.current);
      }
    };
  }, []);
  const hasUnsavedChanges = useMemo(() => {
    if (!existingAnnotation) return canSave;
    return (
      !matchesSavedCorners(corners, existingAnnotation.corners)
      || !matchesSavedLabels(boardLabels, normalizeAnnotationLabels(existingAnnotation.labels))
    );
  }, [boardLabels, canSave, corners, existingAnnotation]);

  // ── Rectify ──

  const rectifyBoard = useCallback(
    async (nextCorners: Array<{ x: number; y: number }>) => {
      if (nextCorners.length !== 4) return;
      const requestId = latestRectifyRequestIdRef.current + 1;
      latestRectifyRequestIdRef.current = requestId;
      setRectifying(true);
      try {
        const result = await splitConfig.rectifyFrame({
          session_id: sessionId,
          frame_index: selectedFrame,
          corners: nextCorners.map((p) => [p.x, p.y]),
          output_size: 512,
        });
        if (latestRectifyRequestIdRef.current !== requestId) return;
        setRectifiedImageB64(result.image_b64);
      } catch (e) {
        if (latestRectifyRequestIdRef.current === requestId) {
          toast.error(e instanceof Error ? e.message : "Rectify failed");
        }
      } finally {
        if (latestRectifyRequestIdRef.current === requestId) {
          setRectifying(false);
        }
      }
    },
    [selectedFrame, sessionId],
  );

  const flushQueuedRectify = useCallback(async () => {
    if (rectifyInFlightRef.current) return;
    const nextCorners = queuedRectifyCornersRef.current;
    if (!nextCorners || nextCorners.length !== 4) return;

    queuedRectifyCornersRef.current = null;
    rectifyInFlightRef.current = true;
    try {
      await rectifyBoard(nextCorners);
    } finally {
      rectifyInFlightRef.current = false;
      if (queuedRectifyCornersRef.current) {
        void flushQueuedRectify();
      }
    }
  }, [rectifyBoard]);

  const scheduleRectify = useCallback((nextCorners: Array<{ x: number; y: number }>, immediate = false) => {
    queuedRectifyCornersRef.current = nextCorners;
    if (liveRectifyTimerRef.current !== null) {
      window.clearTimeout(liveRectifyTimerRef.current);
      liveRectifyTimerRef.current = null;
    }
    if (immediate) {
      void flushQueuedRectify();
      return;
    }
    liveRectifyTimerRef.current = window.setTimeout(() => {
      liveRectifyTimerRef.current = null;
      void flushQueuedRectify();
    }, LIVE_RECTIFY_DELAY_MS);
  }, [flushQueuedRectify]);

  const pointFromClientCoordinates = useCallback((clientX: number, clientY: number) => {
    const image = imageRef.current;
    if (!image || !sourceImageSize) return null;
    const rect = image.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;

    return {
      x: clamp(((clientX - rect.left) / rect.width) * sourceImageSize.width, 0, sourceImageSize.width - 1),
      y: clamp(((clientY - rect.top) / rect.height) * sourceImageSize.height, 0, sourceImageSize.height - 1),
    };
  }, [sourceImageSize]);

  // ── Image click → corner ──

  const handleImageClick = useCallback(
    async (event: React.MouseEvent<HTMLImageElement>) => {
      if (corners.length >= 4) return;
      const point = pointFromClientCoordinates(event.clientX, event.clientY);
      if (!point) return;
      const next = [...corners, point];
      setCorners(next);
      if (next.length === 4) await rectifyBoard(next);
    },
    [corners, pointFromClientCoordinates, rectifyBoard],
  );

  const updateDraggedCorner = useCallback((clientX: number, clientY: number, immediate = false) => {
    const cornerIndex = draggedCornerIndexRef.current;
    if (cornerIndex === null) return;

    const point = pointFromClientCoordinates(clientX, clientY);
    if (!point) return;

    const previousCorners = cornersRef.current;
    if (previousCorners.length !== 4) return;

    const nextCorners = previousCorners.map((corner, index) => (
      index === cornerIndex ? point : corner
    ));

    cornersRef.current = nextCorners;
    setCorners(nextCorners);
    scheduleRectify(nextCorners, immediate);
  }, [pointFromClientCoordinates, scheduleRectify]);

  const handleCornerPointerDown = useCallback((cornerIndex: number, event: React.PointerEvent<HTMLButtonElement>) => {
    if (cornersRef.current.length !== 4) return;
    event.preventDefault();
    event.stopPropagation();
    draggedCornerIndexRef.current = cornerIndex;
    setDraggedCornerIndex(cornerIndex);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const handleCornerPointerMove = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    if (draggedCornerIndexRef.current === null) return;
    event.preventDefault();
    event.stopPropagation();
    updateDraggedCorner(event.clientX, event.clientY);
  }, [updateDraggedCorner]);

  const finishCornerDrag = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    if (draggedCornerIndexRef.current === null) return;
    event.preventDefault();
    event.stopPropagation();
    updateDraggedCorner(event.clientX, event.clientY, true);
    draggedCornerIndexRef.current = null;
    setDraggedCornerIndex(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, [updateDraggedCorner]);

  // ── Square click → cycle label ──

  const handleSquareClick = useCallback((squareIndex: number) => {
    setBoardLabels((prev) => {
      const next = [...prev];
      next[squareIndex] = cycleLabel(prev[squareIndex]);
      return next;
    });
  }, []);

  // ── Reset / Save / Delete ──

  const prefillFromFen = useCallback(() => {
    if (displayedReplayFen) {
      try { setBoardLabels(fenToLabels(displayedReplayFen)); } catch { setBoardLabels(emptyLabels()); }
    } else {
      setBoardLabels(emptyLabels());
    }
  }, [displayedReplayFen]);

  const resetCorners = useCallback(() => {
    setCorners([]);
    setRectifiedImageB64(null);
    setExistingAnnotation(null);
    prefillFromFen();
  }, [prefillFromFen]);

  const copyFenFromPreviousFrame = useCallback(() => {
    if (!previousFrameDisplayedFen) return;
    try {
      setBoardLabels(fenToLabels(previousFrameDisplayedFen));
    } catch {
      setBoardLabels(emptyLabels());
    }
  }, [previousFrameDisplayedFen]);

  const copyFenFromNextFrame = useCallback(() => {
    if (!nextFrameDisplayedFen) return;
    try {
      setBoardLabels(fenToLabels(nextFrameDisplayedFen));
    } catch {
      setBoardLabels(emptyLabels());
    }
  }, [nextFrameDisplayedFen]);

  const saveAnnotation = useCallback(async () => {
    if (!canSave) return null;
    setSaving(true);
    try {
      const result = await splitConfig.saveAnnotation({
        session_id: sessionId,
        clip_path: clipPath,
        frame_index: selectedFrame,
        corners: corners.map((p) => [p.x, p.y]),
        labels: boardLabels,
        output_size: 512,
      });
      setExistingAnnotation(result.annotation);
      await refreshMoveCorrections();
      return result.annotation;
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Save failed");
      return null;
    } finally {
      setSaving(false);
    }
  }, [boardLabels, canSave, clipPath, corners, refreshMoveCorrections, selectedFrame, sessionId]);

  const handleSaveAndStepForward = useCallback(async () => {
    const annotation = await saveAnnotation();
    if (!annotation) return;
    setSelectedFrame((frame) => Math.min(frameCount - 1, frame + 1));
  }, [frameCount, saveAnnotation]);

  const handleDelete = useCallback(async () => {
    if (!existingAnnotation) return;
    setDeleting(true);
    try {
      await splitConfig.deleteAnnotation(clipPath, selectedFrame);
      resetCorners();
      await refreshMoveCorrections();
      toast.success("Annotation deleted");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  }, [clipPath, existingAnnotation, refreshMoveCorrections, resetCorners, selectedFrame]);

  // ── Load existing annotation on frame change, keeping corners if set ──

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const ann = await splitConfig.getAnnotation(clipPath, selectedFrame);
        if (cancelled) return;
        setExistingAnnotation(ann);
        if (ann) {
          // Saved annotation for this frame — use its corners and labels
          setCorners(ann.corners.map(([x, y]) => ({ x, y })));
          setBoardLabels(normalizeAnnotationLabels(ann.labels));
          await rectifyBoard(ann.corners.map(([x, y]) => ({ x, y })));
        } else {
          // No saved annotation — keep current corners, re-rectify, prefill FEN
          prefillFromFen();
          setCorners((prev) => {
            if (prev.length === 4) {
              void rectifyBoard(prev);
            } else {
              setRectifiedImageB64(null);
            }
            return prev;
          });
        }
      } catch { /* no annotation */ }
    })();
    return () => { cancelled = true; };
  }, [clipPath, prefillFromFen, rectifyBoard, selectedFrame]);

  const frameUrl = clipFrameUrl(sessionId, selectedFrame);

  return (
    <div className="space-y-2">
      {/* Header row: nav + actions */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Link href={splitConfig.indexHref} className="text-xs text-blue-500 hover:underline">
            &larr; {splitConfig.indexLabel}
          </Link>
          <span className="font-mono text-sm font-medium">{filename}</span>
          <Badge variant="outline" className="text-[10px]">{splitConfig.splitLabel}</Badge>
          <Badge variant="secondary" className="text-[10px]">{frameCount} frames</Badge>
          <Badge variant="secondary" className="text-[10px]">{effectiveTotalMoves} moves</Badge>
        </div>
        <div className="flex items-center justify-end gap-2">
          <div className="flex shrink-0 items-center gap-1">
            <button type="button" className={NAV_MOVE_BUTTON_CLASS} disabled={previousMoveFrame === null} onClick={() => previousMoveFrame !== null && setSelectedFrame(previousMoveFrame)} title="Previous move">&larr; Move</button>
            <button type="button" className={NAV_STEP_BUTTON_CLASS} disabled={selectedFrame <= 0} onClick={() => setSelectedFrame((f) => Math.max(0, f - 1))} title="Step back">&lt;</button>
            <span className={NAV_COUNTER_CLASS}>{selectedFrame + 1}/{frameCount}</span>
            <button type="button" className={NAV_STEP_BUTTON_CLASS} disabled={selectedFrame >= frameCount - 1} onClick={() => setSelectedFrame((f) => Math.min(frameCount - 1, f + 1))} title="Step forward">&gt;</button>
            <button type="button" className={NAV_MOVE_BUTTON_CLASS} disabled={nextMoveFrame === null} onClick={() => nextMoveFrame !== null && setSelectedFrame(nextMoveFrame)} title="Next move">Move &rarr;</button>
          </div>
          <Badge
            variant="outline"
            className={`flex h-9 w-28 shrink-0 items-center justify-center px-2 text-[10px] ${selectedMove ? "" : "invisible"}`}
            title={selectedMove ? moveLabel(selectedMoveIndex, selectedMove) : undefined}
          >
            <span className="block w-full truncate text-center">
              {selectedMove ? moveLabel(selectedMoveIndex, selectedMove) : "\u00A0"}
            </span>
          </Badge>
        </div>
      </div>

      {/* Main 3-column layout: camera crop | replay + moves | rectified annotation */}
      <div className="grid items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_220px_minmax(0,1fr)]">
        {/* Left: Camera crop with corners */}
        <div className="space-y-1">
          <div className="text-[10px] text-muted-foreground">
            {corners.length < 4 ? `Click corners: a8 \u2192 h8 \u2192 h1 \u2192 a1 (${corners.length}/4)` : "Corners set \u2014 drag handles to adjust this frame"}
          </div>
          <div className="relative overflow-hidden rounded border">
            <img
              ref={imageRef}
              src={frameUrl}
              alt={`Frame ${selectedFrame}`}
              className={`block w-full ${corners.length < 4 ? "cursor-crosshair" : ""}`}
              onLoad={(e) => setSourceImageSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
              onClick={(e) => void handleImageClick(e)}
            />
            {sourceImageSize && corners.map((pt, i) => (
              <button
                key={i}
                type="button"
                className={`absolute z-10 -translate-x-1/2 -translate-y-1/2 touch-none ${draggedCornerIndex === i ? "cursor-grabbing" : "cursor-grab"}`}
                style={{ left: `${(pt.x / sourceImageSize.width) * 100}%`, top: `${(pt.y / sourceImageSize.height) * 100}%` }}
                onPointerDown={(event) => handleCornerPointerDown(i, event)}
                onPointerMove={handleCornerPointerMove}
                onPointerUp={finishCornerDrag}
                onPointerCancel={finishCornerDrag}
                title={`Drag ${cornerLabel(i)}`}
              >
                <span className={`block rounded-full border border-white/80 bg-red-600 px-1.5 py-0.5 text-[9px] font-bold text-white shadow transition-transform ${draggedCornerIndex === i ? "scale-105" : ""}`}>
                  {cornerLabel(i)}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Center: Replay board + move timeline */}
        <div className="flex h-full min-h-0 flex-col gap-2">
          {displayedReplayFen && (
            <div className="shrink-0 space-y-1">
              <div className="text-[10px] font-medium">Computed replay</div>
              <div className="overflow-hidden rounded border bg-white w-fit">
                <ChessBoard
                  fen={displayedReplayFen}
                  size={196}
                  highlightedSquares={selectedMove ? [selectedMove.uci.slice(0, 2), selectedMove.uci.slice(2, 4)] : []}
                  arrows={selectedMove ? [{ from: selectedMove.uci.slice(0, 2), to: selectedMove.uci.slice(2, 4) }] : []}
                />
              </div>
              <div className="font-mono text-[9px] break-all text-muted-foreground leading-tight">{displayedReplayFen}</div>
            </div>
          )}
          {effectiveMoves.length > 0 && (
            <div className="flex min-h-0 flex-1 flex-col space-y-1">
              <div className="shrink-0 text-[10px] font-medium">Moves</div>
              <div className="min-h-0 flex-1 overflow-y-auto space-y-0.5 rounded border p-1">
                {effectiveMoves.map((move, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setSelectedFrame(move.frame_index)}
                    className={`w-full text-left rounded px-1.5 py-0.5 text-[11px] ${move.frame_index === selectedFrame ? "bg-primary/10 font-medium" : "hover:bg-muted/50"}`}
                  >
                    #{i + 1} <span className="font-medium">{move.san || move.uci}</span>{move.is_manual && <span className="ml-1 text-[10px] text-amber-600">manual</span>} <span className="text-muted-foreground">f{move.frame_index}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Physical board annotation */}
        <div className="space-y-1">
          <div className="flex flex-wrap items-center justify-between gap-1">
            <div className="flex items-center gap-0.5">
              {hasUnsavedChanges ? (
                <span className={`${PANEL_STATUS_CLASS} border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300`}>
                  Save needed
                </span>
              ) : existingAnnotation ? (
                <span className={`${PANEL_STATUS_CLASS} border-green-600/40 bg-green-600/10 text-green-700 dark:text-green-300`}>
                  Saved
                </span>
              ) : null}
              {existingAnnotation && (
                <button type="button" onClick={() => void handleDelete()} disabled={deleting} className="rounded border border-red-600/40 bg-red-600/10 px-1.5 py-0.5 text-[10px] text-red-700 hover:bg-red-600/20 dark:text-red-300 disabled:opacity-50">
                  {deleting ? "Deleting\u2026" : "Delete"}
                </button>
              )}
              <button type="button" onClick={resetCorners} className={PANEL_BUTTON_CLASS}>Reset</button>
              <button
                type="button"
                onClick={copyFenFromPreviousFrame}
                disabled={!previousFrameDisplayedFen}
                className={PANEL_BUTTON_CLASS}
                title="Copy FEN from previous frame"
              >
                Copy prev FEN
              </button>
              <button
                type="button"
                onClick={copyFenFromNextFrame}
                disabled={!nextFrameDisplayedFen}
                className={PANEL_BUTTON_CLASS}
                title="Copy FEN from next frame"
              >
                Copy next FEN
              </button>
            </div>
            <button
              type="button"
              onClick={() => void handleSaveAndStepForward()}
              disabled={!canSave || saving}
              className={PRIMARY_SAVE_BUTTON_CLASS}
              title={selectedFrame < frameCount - 1 ? "Save and advance one frame" : "Save this frame"}
            >
              {saving ? "Saving\u2026" : selectedFrame < frameCount - 1 ? "Save + Next >" : "Save"}
            </button>
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
                  {boardLabels.map((label, si) => {
                    const pieceLetter = label !== null && label > 0 ? SQUARE_CLASS_NAMES[label] : null;
                    return (
                      <button
                        key={si}
                        type="button"
                        className="relative flex items-center justify-center border border-white/20 hover:bg-blue-500/20"
                        onClick={() => handleSquareClick(si)}
                        title={`${squareName(si)} \u2014 ${label === null ? "unset" : SQUARE_CLASS_NAMES[label]} \u2014 click to cycle`}
                      >
                        {pieceLetter && (
                          <svg viewBox="0 0 45 45" className="pointer-events-none h-[60%] w-[60%] drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
                            {chessPieceSvg(pieceLetter)}
                          </svg>
                        )}
                        {label === 0 && (
                          <span className="pointer-events-none text-white/50 text-[10px] font-bold">{"\u00B7"}</span>
                        )}
                        <span className="pointer-events-none absolute left-0.5 top-0 text-[7px] text-white/70">{squareName(si)}</span>
                      </button>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
                {rectifying ? "Rectifying board\u2026" : "Mark four corners on the camera crop to rectify the board."}
              </div>
            )}
          </div>

          <div className="text-[10px] text-muted-foreground">
            {boardLabeledCount}/64 labeled &middot; Click square to cycle piece
          </div>
        </div>
      </div>

      {/* Frame slider */}
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={0}
          max={Math.max(frameCount - 1, 0)}
          value={selectedFrame}
          onChange={(e) => setSelectedFrame(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs text-muted-foreground whitespace-nowrap">
          {frameTime !== null ? `${frameTime.toFixed(1)}s` : ""}
        </span>
      </div>

      {/* Source video */}
      {sourceVideoUrl && !videoUnavailable && (
        <div className="rounded border p-2">
          <div className="text-[10px] font-medium mb-1">Source video</div>
          <video
            ref={(node) => { videoRef.current = node; }}
            src={sourceVideoUrl}
            className="w-full max-h-48 rounded"
            controls
            muted
            onError={() => setVideoUnavailable(true)}
          />
        </div>
      )}
    </div>
  );
}
