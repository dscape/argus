"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ChessBoard } from "@/components/ChessBoard";
import { chessPieceSvg } from "@/components/chess-pieces";
import { Badge } from "@/components/ui/badge";
import { usePhysicalTransientAnnotations } from "@/hooks/usePhysicalTransientAnnotations";
import {
  clipCameraFrameUrl,
  clipFrameUrl,
  clipSourceVideoUrl,
  deleteClipSession,
  deletePhysicalEvalAnnotation,
  deletePhysicalTrainAnnotation,
  detectPhysicalEvalCorners,
  detectPhysicalTrainCorners,
  trackPhysicalEvalCorners,
  trackPhysicalTrainCorners,
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
  updateVideoClip,
  type PhysicalEvalAnnotation,
  type PhysicalEvalMoveCorrections,
} from "@/lib/api";
import type { ClipInspectResponse, DetectedMove } from "@/lib/types";

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
const LABEL_COUNT = SQUARE_CLASS_NAMES.length; // 13
const NAV_BUTTON_CLASS =
  "inline-flex h-9 items-center justify-center rounded border px-2 text-xs whitespace-nowrap disabled:opacity-40";
const NAV_MOVE_BUTTON_CLASS = `${NAV_BUTTON_CLASS} w-24`;
const NAV_STEP_BUTTON_CLASS = `${NAV_BUTTON_CLASS} w-10 font-medium`;
const NAV_COUNTER_CLASS =
  "inline-flex h-9 w-20 items-center justify-center text-xs font-medium tabular-nums";
const PANEL_BUTTON_CLASS =
  "inline-flex h-5 items-center justify-center rounded border px-1.5 text-[9px] leading-none whitespace-nowrap disabled:opacity-40";
const PANEL_STATUS_CLASS =
  "inline-flex h-5 min-w-[5rem] items-center justify-center rounded border px-1.5 text-[9px] leading-none whitespace-nowrap";
const PRIMARY_SAVE_BUTTON_CLASS =
  "inline-flex h-6 min-w-[8.75rem] items-center justify-center rounded border bg-primary px-2.5 text-[9px] font-medium text-primary-foreground whitespace-nowrap transition-colors hover:bg-primary/90 disabled:opacity-40";
const LIVE_RECTIFY_DELAY_MS = 75;
const MIN_CAMERA_BBOX_AREA_RATIO = 0.02;
const MAX_CAMERA_BBOX_AREA_RATIO = 0.25;

function fenToLabels(fen: string): Array<number | null> {
  const placement = fen.split(" ", 1)[0];
  const map = new Map<string, number>(SQUARE_CLASS_NAMES.map((l, i) => [l, i]));
  const labels: Array<number | null> = [];
  for (const rank of placement.split("/")) {
    for (const ch of rank) {
      if (/^[1-8]$/.test(ch)) {
        for (let i = 0; i < Number(ch); i++) labels.push(0);
        continue;
      }
      const idx = map.get(ch);
      if (idx == null) throw new Error(`Unsupported FEN piece: ${ch}`);
      labels.push(idx);
    }
  }
  if (labels.length !== 64)
    throw new Error(`Expected 64 squares, got ${labels.length}`);
  return labels;
}

function squareName(idx: number): string {
  return `${String.fromCharCode(97 + (idx % 8))}${8 - Math.floor(idx / 8)}`;
}

function cornerLabel(idx: number): string {
  return ["a8", "h8", "h1", "a1"][idx] ?? String(idx + 1);
}

function emptyLabels(): Array<number | null> {
  return Array.from({ length: 64 }, () => null);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function asNumberList(value: unknown, expectedLength: number): number[] | null {
  if (!Array.isArray(value) || value.length !== expectedLength) return null;
  const numbers = value.map((item) => Number(item));
  return numbers.every(Number.isFinite) ? numbers : null;
}

function isUsableCameraBbox(
  bbox: number[] | null,
  refResolution: number[] | null,
): boolean {
  if (!bbox || !refResolution) return false;
  const [x, y, w, h] = bbox;
  const [refW, refH] = refResolution;
  if (w <= 0 || h <= 0 || refW <= 0 || refH <= 0) return false;
  if (x === 0 && y === 0 && w === 100 && h === 100) return false;
  if (x < 0 || y < 0 || x + w > refW || y + h > refH) return false;
  const areaRatio = (w * h) / (refW * refH);
  return (
    areaRatio >= MIN_CAMERA_BBOX_AREA_RATIO &&
    areaRatio <= MAX_CAMERA_BBOX_AREA_RATIO
  );
}

function expandCameraBbox(
  bbox: number[],
  padding: number,
  refResolution: number[],
): [number, number, number, number] {
  const [x, y, w, h] = bbox;
  const [refW, refH] = refResolution;
  const x1 = Math.max(0, x - padding);
  const y1 = Math.max(0, y - padding);
  const x2 = Math.min(refW, x + w + padding);
  const y2 = Math.min(refH, y + h + padding);
  return [x1, y1, x2 - x1, y2 - y1];
}

function cycleLabel(current: number | null): number | null {
  if (current === null) return 0;
  const next = current + 1;
  return next >= LABEL_COUNT ? null : next;
}

function moveLabel(index: number, move: DetectedMove): string {
  return `#${index + 1} ${move.san || move.uci}`;
}

function moveTouchAnnotationComplete(annotation: {
  start_frame_index: number | null;
  end_frame_index: number | null;
}): boolean {
  return (
    annotation.start_frame_index !== null && annotation.end_frame_index !== null
  );
}

function normalizeAnnotationLabels(
  labels: Array<number | null>,
): Array<number | null> {
  return labels.map((label) => (typeof label === "number" ? label : null));
}

function matchesSavedCorners(
  corners: Array<{ x: number; y: number }>,
  savedCorners: number[][],
): boolean {
  return (
    corners.length === savedCorners.length &&
    corners.every(
      (corner, index) =>
        corner.x === savedCorners[index]?.[0] &&
        corner.y === savedCorners[index]?.[1],
    )
  );
}

function matchesSavedLabels(
  labels: Array<number | null>,
  savedLabels: Array<number | null>,
): boolean {
  return (
    labels.length === savedLabels.length &&
    labels.every((label, index) => label === savedLabels[index])
  );
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
  detectCorners: typeof detectPhysicalEvalCorners;
  trackCorners: typeof trackPhysicalEvalCorners;
}

const SPLIT_CONFIG: Record<
  PhysicalAnnotationSplit,
  PhysicalAnnotationSplitConfig
> = {
  val: {
    indexHref: "/annotate/physical?split=val",
    indexLabel: "All validation clips",
    splitLabel: "Validation",
    getAnnotation: getPhysicalEvalAnnotation,
    getMoveCorrections: getPhysicalEvalMoveCorrections,
    rectifyFrame: rectifyPhysicalEvalFrame,
    saveAnnotation: savePhysicalEvalAnnotation,
    deleteAnnotation: deletePhysicalEvalAnnotation,
    detectCorners: detectPhysicalEvalCorners,
    trackCorners: trackPhysicalEvalCorners,
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
    detectCorners: detectPhysicalTrainCorners,
    trackCorners: trackPhysicalTrainCorners,
  },
};

interface PendingCornerTranslation {
  oldSize: { width: number; height: number };
  oldPadding: number;
  newPadding: number;
  corners: Array<{ x: number; y: number }>;
}

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
        if (cancelled) {
          await deleteClipSession(sid).catch(() => {});
          return;
        }
        setSessionId(sid);
        const info = await getClipInfo(sid);
        if (cancelled) {
          await deleteClipSession(sid).catch(() => {});
          return;
        }
        setClipInfo(info);
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Failed to load clip");
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
        <Link
          href={splitConfig.indexHref}
          className="text-sm text-blue-500 hover:underline"
        >
          &larr; {splitConfig.indexLabel}
        </Link>
        <p className="text-sm text-muted-foreground">
          Loading {filename}&hellip;
        </p>
      </div>
    );
  }

  if (error || !clipInfo || !sessionId) {
    return (
      <div className="space-y-4">
        <Link
          href={splitConfig.indexHref}
          className="text-sm text-blue-500 hover:underline"
        >
          &larr; {splitConfig.indexLabel}
        </Link>
        <p className="text-sm text-destructive">
          {error ?? "Failed to load clip"}
        </p>
      </div>
    );
  }

  return (
    <AnnotationContent
      filename={filename}
      clipPath={clipPath}
      sessionId={sessionId}
      clipInfo={clipInfo}
      split={split}
      splitConfig={splitConfig}
    />
  );
}

function AnnotationContent({
  filename,
  clipPath,
  sessionId,
  clipInfo,
  split,
  splitConfig,
}: {
  filename: string;
  clipPath: string;
  sessionId: string;
  clipInfo: ClipInspectResponse;
  split: PhysicalAnnotationSplit;
  splitConfig: PhysicalAnnotationSplitConfig;
}) {
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [corners, setCorners] = useState<Array<{ x: number; y: number }>>([]);
  const [boardLabels, setBoardLabels] =
    useState<Array<number | null>>(emptyLabels);
  const [rectifiedImageB64, setRectifiedImageB64] = useState<string | null>(
    null,
  );
  const [sourceImageSize, setSourceImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [rectifying, setRectifying] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [existingAnnotation, setExistingAnnotation] =
    useState<PhysicalEvalAnnotation | null>(null);
  const [moveCorrections, setMoveCorrections] =
    useState<PhysicalEvalMoveCorrections | null>(null);

  // Auto-detect corners
  const [autoDetect, setAutoDetect] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const autoDetectRef = useRef(false);

  // Camera padding
  const [cameraPadding, setCameraPadding] = useState(0);
  const [savingCameraBbox, setSavingCameraBbox] = useState(false);
  const pendingCornerTranslationRef = useRef<PendingCornerTranslation | null>(
    null,
  );

  // Source video
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [videoUnavailable, setVideoUnavailable] = useState(false);
  const [draggedCornerIndex, setDraggedCornerIndex] = useState<number | null>(
    null,
  );
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const cornersRef = useRef<Array<{ x: number; y: number }>>([]);
  const cornersFrameIndexRef = useRef<number>(0);
  const prevFrameCornersRef = useRef<Array<{ x: number; y: number }>>([]);
  const draggedCornerIndexRef = useRef<number | null>(null);
  const latestRectifyRequestIdRef = useRef(0);
  const liveRectifyTimerRef = useRef<number | null>(null);
  const queuedRectifyCornersRef = useRef<Array<{
    x: number;
    y: number;
  }> | null>(null);
  const rectifyInFlightRef = useRef(false);

  const frameCount = clipInfo.num_frames;
  const frameTime = clipInfo.frame_timestamps_seconds[selectedFrame] ?? null;

  const refreshMoveCorrections = useCallback(async () => {
    try {
      const nextCorrections = await splitConfig.getMoveCorrections(
        sessionId,
        clipPath,
      );
      setMoveCorrections(nextCorrections);
    } catch (e) {
      toast.error(
        e instanceof Error
          ? e.message
          : "Failed to load manual move corrections",
      );
    }
  }, [clipPath, sessionId]);

  useEffect(() => {
    void refreshMoveCorrections();
  }, [refreshMoveCorrections]);

  const effectiveFrameReplayFens =
    moveCorrections?.frame_replay_fens ?? clipInfo.frame_replay_fens;
  const effectiveMoves = moveCorrections?.moves ?? clipInfo.moves;
  const effectiveTotalMoves =
    moveCorrections?.total_moves ?? clipInfo.total_moves;
  const [activeMoveIndex, setActiveMoveIndex] = useState<number | null>(null);
  const {
    saving: transientSaving,
    hasInvalidMoveLabels: hasInvalidTransientMoveLabels,
    moveAnnotations,
    setMoveStartFrame,
    setMoveEndFrame,
    clearMoveStartFrame,
    clearMoveEndFrame,
    isFrameOccluded,
    toggleFrameOccluded,
  } = usePhysicalTransientAnnotations({
    split,
    clipPath,
    effectiveMoves,
    frameCount,
  });
  const replayFen = effectiveFrameReplayFens[selectedFrame] ?? null;

  // Load source video
  useEffect(() => {
    const controller = new AbortController();
    let blobUrl: string | null = null;
    setSourceVideoUrl(null);
    setVideoUnavailable(false);

    fetch(clipSourceVideoUrl(sessionId), { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error();
        return res.blob();
      })
      .then((blob) => {
        blobUrl = URL.createObjectURL(blob);
        setSourceVideoUrl(blobUrl);
      })
      .catch(() => {
        if (!controller.signal.aborted) setVideoUnavailable(true);
      });

    return () => {
      controller.abort();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
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
  const selectedMove =
    selectedMoveIndex >= 0 ? effectiveMoves[selectedMoveIndex] : null;
  const currentFrameOccluded = isFrameOccluded(selectedFrame);

  useEffect(() => {
    if (effectiveMoves.length === 0) {
      setActiveMoveIndex(null);
      return;
    }
    if (selectedMoveIndex >= 0) {
      setActiveMoveIndex(selectedMoveIndex);
      return;
    }
    setActiveMoveIndex((current) => {
      if (current === null) return 0;
      return Math.min(current, effectiveMoves.length - 1);
    });
  }, [effectiveMoves.length, selectedMoveIndex]);

  const displayedReplayFen = selectedMove?.fen_after ?? replayFen;
  const previousFrameDisplayedFen = useMemo(() => {
    if (selectedFrame <= 0) return null;
    const previousMove = effectiveMoves.find(
      (move) => move.frame_index === selectedFrame - 1,
    );
    return (
      previousMove?.fen_after ??
      effectiveFrameReplayFens[selectedFrame - 1] ??
      null
    );
  }, [effectiveFrameReplayFens, effectiveMoves, selectedFrame]);
  const nextFrameDisplayedFen = useMemo(() => {
    if (selectedFrame >= frameCount - 1) return null;
    const nextMove = effectiveMoves.find(
      (move) => move.frame_index === selectedFrame + 1,
    );
    return (
      nextMove?.fen_after ?? effectiveFrameReplayFens[selectedFrame + 1] ?? null
    );
  }, [effectiveFrameReplayFens, effectiveMoves, frameCount, selectedFrame]);

  const previousMoveFrame = useMemo(() => {
    for (let i = effectiveMoves.length - 1; i >= 0; i--) {
      if (effectiveMoves[i].frame_index < selectedFrame)
        return effectiveMoves[i].frame_index;
    }
    return null;
  }, [effectiveMoves, selectedFrame]);

  const nextMoveFrame = useMemo(() => {
    for (const m of effectiveMoves) {
      if (m.frame_index > selectedFrame) return m.frame_index;
    }
    return null;
  }, [effectiveMoves, selectedFrame]);

  const rawMetadata = clipInfo.metadata as Record<string, unknown> | undefined;
  const metadataCameraBbox = asNumberList(rawMetadata?.camera_bbox, 4);
  const metadataRefResolution = asNumberList(rawMetadata?.ref_resolution, 2);
  const boardLabeledCount = boardLabels.filter((l) => l !== null).length;
  const canSave = corners.length === 4 && boardLabeledCount > 0;
  const hasDbClip = typeof rawMetadata?.source_db_clip_id === "number";
  const hasUsableCameraBbox = isUsableCameraBbox(
    metadataCameraBbox,
    metadataRefResolution,
  );
  const calibrateHref =
    typeof rawMetadata?.source_video_id === "string"
      ? `/videos/${rawMetadata.source_video_id}/calibrate`
      : null;

  useEffect(() => {
    cornersRef.current = corners;
    if (corners.length > 0) cornersFrameIndexRef.current = selectedFrame;
  }, [corners, selectedFrame]);

  useEffect(() => {
    autoDetectRef.current = autoDetect;
  }, [autoDetect]);

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
      !matchesSavedCorners(corners, existingAnnotation.corners) ||
      !matchesSavedLabels(
        boardLabels,
        normalizeAnnotationLabels(existingAnnotation.labels),
      )
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
          padding_px: cameraPadding,
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
    [cameraPadding, selectedFrame, sessionId],
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

  const scheduleRectify = useCallback(
    (nextCorners: Array<{ x: number; y: number }>, immediate = false) => {
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
    },
    [flushQueuedRectify],
  );

  const pointFromClientCoordinates = useCallback(
    (clientX: number, clientY: number) => {
      const image = imageRef.current;
      if (!image || !sourceImageSize) return null;
      const rect = image.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;

      return {
        x: clamp(
          ((clientX - rect.left) / rect.width) * sourceImageSize.width,
          0,
          sourceImageSize.width - 1,
        ),
        y: clamp(
          ((clientY - rect.top) / rect.height) * sourceImageSize.height,
          0,
          sourceImageSize.height - 1,
        ),
      };
    },
    [sourceImageSize],
  );

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

  const updateDraggedCorner = useCallback(
    (clientX: number, clientY: number, immediate = false) => {
      const cornerIndex = draggedCornerIndexRef.current;
      if (cornerIndex === null) return;

      const point = pointFromClientCoordinates(clientX, clientY);
      if (!point) return;

      const previousCorners = cornersRef.current;
      if (previousCorners.length !== 4) return;

      const nextCorners = previousCorners.map((corner, index) =>
        index === cornerIndex ? point : corner,
      );

      cornersRef.current = nextCorners;
      setCorners(nextCorners);
      scheduleRectify(nextCorners, immediate);
    },
    [pointFromClientCoordinates, scheduleRectify],
  );

  const handleCornerPointerDown = useCallback(
    (cornerIndex: number, event: React.PointerEvent<HTMLButtonElement>) => {
      if (cornersRef.current.length !== 4) return;
      event.preventDefault();
      event.stopPropagation();
      draggedCornerIndexRef.current = cornerIndex;
      setDraggedCornerIndex(cornerIndex);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const handleCornerPointerMove = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (draggedCornerIndexRef.current === null) return;
      event.preventDefault();
      event.stopPropagation();
      updateDraggedCorner(event.clientX, event.clientY);
    },
    [updateDraggedCorner],
  );

  const finishCornerDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (draggedCornerIndexRef.current === null) return;
      event.preventDefault();
      event.stopPropagation();
      updateDraggedCorner(event.clientX, event.clientY, true);
      draggedCornerIndexRef.current = null;
      setDraggedCornerIndex(null);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [updateDraggedCorner],
  );

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
      try {
        setBoardLabels(fenToLabels(displayedReplayFen));
      } catch {
        setBoardLabels(emptyLabels());
      }
    } else {
      setBoardLabels(emptyLabels());
    }
  }, [displayedReplayFen]);

  const handlePaddingChange = useCallback(
    (newPadding: number) => {
      if (!hasUsableCameraBbox || newPadding === cameraPadding) return;
      // Queue corner translation for when the new image loads
      if (corners.length === 4 && sourceImageSize) {
        pendingCornerTranslationRef.current = {
          oldSize: sourceImageSize,
          oldPadding: cameraPadding,
          newPadding: newPadding,
          corners: [...corners],
        };
      }
      setCameraPadding(newPadding);
    },
    [cameraPadding, corners, hasUsableCameraBbox, sourceImageSize],
  );

  const updateCameraBbox = useCallback(async () => {
    const videoId =
      typeof rawMetadata?.source_video_id === "string"
        ? rawMetadata.source_video_id
        : undefined;
    const clipId =
      typeof rawMetadata?.source_db_clip_id === "number"
        ? rawMetadata.source_db_clip_id
        : undefined;
    if (
      !hasUsableCameraBbox ||
      !videoId ||
      clipId == null ||
      !metadataCameraBbox ||
      !metadataRefResolution ||
      cameraPadding <= 0 ||
      savingCameraBbox
    ) {
      return;
    }

    const expanded = expandCameraBbox(
      metadataCameraBbox,
      cameraPadding,
      metadataRefResolution,
    );
    if (!isUsableCameraBbox(expanded, metadataRefResolution)) {
      toast.error(
        "Expanded camera bbox is invalid; reduce padding or recalibrate the clip",
      );
      return;
    }

    setSavingCameraBbox(true);
    try {
      const updated = await updateVideoClip(videoId, clipId, {
        camera_bbox: expanded,
      });
      if (clipInfo.metadata) {
        const metadata = clipInfo.metadata as Record<string, unknown>;
        metadata.camera_bbox = updated.camera_bbox;
        metadata.ref_resolution = updated.ref_resolution;
        metadata.camera_bbox_usable = true;
      }
      pendingCornerTranslationRef.current = null;
      setCameraPadding(0);
      toast.success("Camera bbox updated in DB");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to update bbox");
    } finally {
      setSavingCameraBbox(false);
    }
  }, [
    cameraPadding,
    clipInfo.metadata,
    hasUsableCameraBbox,
    metadataCameraBbox,
    metadataRefResolution,
    rawMetadata,
    savingCameraBbox,
  ]);

  const runTrackCorners = useCallback(
    async (
      sourceFrameIndex: number,
      targetFrameIndex: number,
      sourceCorners: Array<{ x: number; y: number }>,
    ): Promise<boolean> => {
      if (sourceCorners.length !== 4) return false;
      try {
        const result = await splitConfig.trackCorners({
          session_id: sessionId,
          source_frame_index: sourceFrameIndex,
          target_frame_index: targetFrameIndex,
          corners: sourceCorners.map((p) => [p.x, p.y]),
          padding_px: cameraPadding,
        });
        if (result.tracking) {
          const tracked = result.tracking.corners.map(([x, y]) => ({ x, y }));
          setCorners(tracked);
          cornersRef.current = tracked;
          cornersFrameIndexRef.current = targetFrameIndex;
          if (tracked.length === 4) await rectifyBoard(tracked);
          return true;
        }
      } catch {
        /* tracking failed, caller should fall back */
      }
      return false;
    },
    [cameraPadding, rectifyBoard, sessionId, splitConfig],
  );

  const runDetectCorners = useCallback(async () => {
    setDetecting(true);
    try {
      // When corners already exist, track them to the current frame instead
      // of running cold detection (which returns axis-aligned bboxes and
      // destroys perspective-correct corners).
      const existing = cornersRef.current;
      if (existing.length === 4) {
        const sourceFrame = cornersFrameIndexRef.current;
        if (sourceFrame !== selectedFrame) {
          const tracked = await runTrackCorners(
            sourceFrame,
            selectedFrame,
            existing,
          );
          if (tracked) return;
        } else {
          // Corners are already for this frame — nothing to do
          return;
        }
      }

      // No existing corners or tracking failed — cold detection
      const result = await splitConfig.detectCorners({
        session_id: sessionId,
        frame_index: selectedFrame,
        padding_px: cameraPadding,
      });
      if (result.detection) {
        const detected = result.detection.corners.map(([x, y]) => ({ x, y }));
        setCorners(detected);
        cornersRef.current = detected;
        cornersFrameIndexRef.current = selectedFrame;
        if (detected.length === 4) await rectifyBoard(detected);
      } else {
        toast.error("No board detected in this frame");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setDetecting(false);
    }
  }, [
    cameraPadding,
    rectifyBoard,
    runTrackCorners,
    selectedFrame,
    sessionId,
    splitConfig,
  ]);

  const resetCorners = useCallback(() => {
    setCorners([]);
    setRectifiedImageB64(null);
    setExistingAnnotation(null);
    prefillFromFen();
  }, [prefillFromFen]);

  const usePrevCorners = useCallback(async () => {
    const prev = prevFrameCornersRef.current;
    if (prev.length !== 4) return;
    const sourceFrame = cornersFrameIndexRef.current;
    const tracked = await runTrackCorners(sourceFrame, selectedFrame, prev);
    if (!tracked) {
      setCorners(prev);
      cornersRef.current = prev;
      cornersFrameIndexRef.current = selectedFrame;
      await rectifyBoard(prev);
    }
  }, [rectifyBoard, runTrackCorners, selectedFrame]);

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
        padding_px: cameraPadding,
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
  }, [
    boardLabels,
    cameraPadding,
    canSave,
    clipPath,
    corners,
    refreshMoveCorrections,
    selectedFrame,
    sessionId,
    splitConfig,
  ]);

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
  }, [
    clipPath,
    existingAnnotation,
    refreshMoveCorrections,
    resetCorners,
    selectedFrame,
  ]);

  // ── Load existing annotation on frame change, keeping corners if set ──

  useEffect(() => {
    // Snapshot current corners and their frame index before loading the new frame's data
    const prevCorners = [...cornersRef.current];
    const prevFrame = cornersFrameIndexRef.current;
    prevFrameCornersRef.current = prevCorners;
    let cancelled = false;
    void (async () => {
      try {
        const ann = await splitConfig.getAnnotation(clipPath, selectedFrame, {
          sessionId,
          paddingPx: cameraPadding,
        });
        if (cancelled) return;
        setExistingAnnotation(ann);
        if (ann) {
          // Saved annotation — use its corners and labels
          const loadedCorners = ann.corners.map(([x, y]) => ({ x, y }));
          // Legacy annotations saved in 224x224 space need scaling to native resolution.
          // Detect by checking if all corners fit within 224x224 bounds and the current
          // image is significantly larger.
          const maxCorner = Math.max(
            ...loadedCorners.map((p) => Math.max(p.x, p.y)),
          );
          if (
            maxCorner <= 224 &&
            sourceImageSize &&
            sourceImageSize.width > 300
          ) {
            const scaleX = sourceImageSize.width / 224;
            const scaleY = sourceImageSize.height / 224;
            for (const pt of loadedCorners) {
              pt.x *= scaleX;
              pt.y *= scaleY;
            }
          }
          setCorners(loadedCorners);
          cornersRef.current = loadedCorners;
          cornersFrameIndexRef.current = selectedFrame;
          setBoardLabels(normalizeAnnotationLabels(ann.labels));
          await rectifyBoard(loadedCorners);
        } else {
          // No saved annotation — track from previous corners, detect, or keep
          prefillFromFen();
          if (prevCorners.length === 4) {
            // Try optical flow tracking from the previous frame's corners
            const tracked = await runTrackCorners(
              prevFrame,
              selectedFrame,
              prevCorners,
            );
            if (cancelled) return;
            if (!tracked) {
              // Tracking failed — fall back to detect (if auto) or copy prev corners
              if (autoDetectRef.current) {
                void runDetectCorners();
              } else {
                setCorners(prevCorners);
                cornersRef.current = prevCorners;
                cornersFrameIndexRef.current = selectedFrame;
                await rectifyBoard(prevCorners);
              }
            }
          } else if (autoDetectRef.current) {
            void runDetectCorners();
          } else {
            setRectifiedImageB64(null);
          }
        }
      } catch {
        /* no annotation */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    clipPath,
    prefillFromFen,
    rectifyBoard,
    runDetectCorners,
    runTrackCorners,
    selectedFrame,
  ]);

  const frameUrl = hasUsableCameraBbox
    ? clipCameraFrameUrl(sessionId, selectedFrame, cameraPadding)
    : clipFrameUrl(sessionId, selectedFrame);

  return (
    <div className="space-y-2">
      {/* Header row: nav + actions */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Link
            href={splitConfig.indexHref}
            className="text-xs text-blue-500 hover:underline"
          >
            &larr; {splitConfig.indexLabel}
          </Link>
          <span className="font-mono text-sm font-medium">{filename}</span>
          <Badge variant="outline" className="text-[10px]">
            {splitConfig.splitLabel}
          </Badge>
          <Badge variant="secondary" className="text-[10px]">
            {frameCount} frames
          </Badge>
          <Badge variant="secondary" className="text-[10px]">
            {effectiveTotalMoves} moves
          </Badge>
        </div>
        <div className="flex items-center justify-end gap-2">
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              className={NAV_MOVE_BUTTON_CLASS}
              disabled={previousMoveFrame === null}
              onClick={() =>
                previousMoveFrame !== null &&
                setSelectedFrame(previousMoveFrame)
              }
              title="Previous move"
            >
              &larr; Move
            </button>
            <button
              type="button"
              className={NAV_STEP_BUTTON_CLASS}
              disabled={selectedFrame <= 0}
              onClick={() => setSelectedFrame((f) => Math.max(0, f - 1))}
              title="Step back"
            >
              &lt;
            </button>
            <span className={NAV_COUNTER_CLASS}>
              {selectedFrame + 1}/{frameCount}
            </span>
            <button
              type="button"
              className={NAV_STEP_BUTTON_CLASS}
              disabled={selectedFrame >= frameCount - 1}
              onClick={() =>
                setSelectedFrame((f) => Math.min(frameCount - 1, f + 1))
              }
              title="Step forward"
            >
              &gt;
            </button>
            <button
              type="button"
              className={NAV_MOVE_BUTTON_CLASS}
              disabled={nextMoveFrame === null}
              onClick={() =>
                nextMoveFrame !== null && setSelectedFrame(nextMoveFrame)
              }
              title="Next move"
            >
              Move &rarr;
            </button>
          </div>
          <Badge
            variant="outline"
            className={`flex h-9 w-28 shrink-0 items-center justify-center px-2 text-[10px] ${selectedMove ? "" : "invisible"}`}
            title={
              selectedMove
                ? moveLabel(selectedMoveIndex, selectedMove)
                : undefined
            }
          >
            <span className="block w-full truncate text-center">
              {selectedMove
                ? moveLabel(selectedMoveIndex, selectedMove)
                : "\u00A0"}
            </span>
          </Badge>
        </div>
      </div>

      {/* Main 3-column layout: camera crop | replay + moves | rectified annotation */}
      <div className="grid items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_220px_minmax(0,1fr)]">
        {/* Left: Camera crop with corners */}
        <div className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[10px] text-muted-foreground">
              {corners.length < 4
                ? `Click corners: a8 \u2192 h8 \u2192 h1 \u2192 a1 (${corners.length}/4)`
                : "Corners set \u2014 drag handles to adjust this frame"}
            </div>
            {hasDbClip && (
              <div className="flex items-center gap-1">
                {hasUsableCameraBbox ? (
                  <>
                    <label
                      className="text-[9px] text-muted-foreground whitespace-nowrap"
                      htmlFor="cam-pad"
                    >
                      Pad
                    </label>
                    <input
                      id="cam-pad"
                      type="range"
                      min={0}
                      max={200}
                      step={1}
                      value={cameraPadding}
                      onChange={(e) =>
                        handlePaddingChange(Number(e.target.value))
                      }
                      disabled={savingCameraBbox}
                      className="h-3 w-16"
                    />
                    <span className="text-[9px] tabular-nums w-6 text-right text-muted-foreground">
                      {cameraPadding}
                    </span>
                    <button
                      type="button"
                      onClick={() => void updateCameraBbox()}
                      disabled={cameraPadding <= 0 || savingCameraBbox}
                      className="rounded border border-blue-500/40 bg-blue-500/10 px-1.5 py-0.5 text-[9px] text-blue-700 hover:bg-blue-500/20 dark:text-blue-300 disabled:opacity-40"
                    >
                      {savingCameraBbox ? "Saving..." : "Save bbox"}
                    </button>
                  </>
                ) : (
                  <span className="text-[9px] text-muted-foreground whitespace-nowrap">
                    Padding unavailable —{" "}
                    {calibrateHref ? (
                      <Link
                        href={calibrateHref}
                        className="text-blue-500 hover:underline"
                      >
                        set camera bbox
                      </Link>
                    ) : (
                      "camera bbox missing"
                    )}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="relative overflow-hidden rounded border">
            <button
              type="button"
              onClick={() => toggleFrameOccluded(selectedFrame)}
              className={`absolute right-2 top-2 z-20 rounded border px-2 py-1 text-[10px] font-medium shadow ${currentFrameOccluded ? "border-amber-500/60 bg-amber-500/90 text-black" : "border-black/30 bg-black/60 text-white hover:bg-black/70"}`}
            >
              {currentFrameOccluded ? "Hand occluded" : "Mark hand occluded"}
            </button>
            {currentFrameOccluded && (
              <div className="pointer-events-none absolute inset-0 z-10 border-2 border-amber-500/80" />
            )}
            <img
              ref={imageRef}
              src={frameUrl}
              alt={`Frame ${selectedFrame}`}
              className={`block w-full ${corners.length < 4 ? "cursor-crosshair" : ""}`}
              onLoad={(e) => {
                const newSize = {
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                };
                setSourceImageSize(newSize);
                // Translate corners when padding changes and a new image loads
                const pending = pendingCornerTranslationRef.current;
                if (pending && pending.corners.length === 4) {
                  const { oldSize, oldPadding, newPadding } = pending;
                  const boardW = oldSize.width - 2 * oldPadding;
                  const boardH = oldSize.height - 2 * oldPadding;
                  if (boardW > 0 && boardH > 0) {
                    const translated = pending.corners.map((pt) => ({
                      x:
                        ((pt.x - oldPadding) / boardW) *
                          (newSize.width - 2 * newPadding) +
                        newPadding,
                      y:
                        ((pt.y - oldPadding) / boardH) *
                          (newSize.height - 2 * newPadding) +
                        newPadding,
                    }));
                    setCorners(translated);
                    cornersRef.current = translated;
                    void rectifyBoard(translated);
                  }
                  pendingCornerTranslationRef.current = null;
                }
              }}
              onClick={(e) => void handleImageClick(e)}
            />
            {sourceImageSize &&
              corners.map((pt, i) => {
                const labelDx = i === 0 || i === 3 ? -18 : 18;
                const labelDy = i === 0 || i === 1 ? -14 : 14;
                return (
                  <button
                    key={i}
                    type="button"
                    className={`absolute z-10 -translate-x-1/2 -translate-y-1/2 touch-none ${draggedCornerIndex === i ? "cursor-grabbing" : "cursor-grab"}`}
                    style={{
                      left: `${(pt.x / sourceImageSize.width) * 100}%`,
                      top: `${(pt.y / sourceImageSize.height) * 100}%`,
                    }}
                    onPointerDown={(event) => handleCornerPointerDown(i, event)}
                    onPointerMove={handleCornerPointerMove}
                    onPointerUp={finishCornerDrag}
                    onPointerCancel={finishCornerDrag}
                    title={`Drag ${cornerLabel(i)}`}
                  >
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      className="drop-shadow"
                    >
                      <line
                        x1="12"
                        y1="2"
                        x2="12"
                        y2="22"
                        stroke="#dc2626"
                        strokeWidth="2"
                      />
                      <line
                        x1="2"
                        y1="12"
                        x2="22"
                        y2="12"
                        stroke="#dc2626"
                        strokeWidth="2"
                      />
                      <line
                        x1="12"
                        y1="2"
                        x2="12"
                        y2="22"
                        stroke="white"
                        strokeWidth="0.75"
                      />
                      <line
                        x1="2"
                        y1="12"
                        x2="22"
                        y2="12"
                        stroke="white"
                        strokeWidth="0.75"
                      />
                    </svg>
                    <span
                      className="absolute text-[9px] font-bold text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] select-none"
                      style={{
                        left: labelDx,
                        top: labelDy,
                        transform: "translate(-50%, -50%)",
                      }}
                    >
                      {cornerLabel(i)}
                    </span>
                  </button>
                );
              })}
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
                  highlightedSquares={
                    selectedMove
                      ? [
                          selectedMove.uci.slice(0, 2),
                          selectedMove.uci.slice(2, 4),
                        ]
                      : []
                  }
                  arrows={
                    selectedMove
                      ? [
                          {
                            from: selectedMove.uci.slice(0, 2),
                            to: selectedMove.uci.slice(2, 4),
                          },
                        ]
                      : []
                  }
                />
              </div>
              <div className="font-mono text-[9px] break-all text-muted-foreground leading-tight">
                {displayedReplayFen}
              </div>
            </div>
          )}
          <div className="flex min-h-0 flex-1 flex-col space-y-1">
            <div className="flex flex-wrap items-center justify-between gap-2 rounded border p-2">
              {hasInvalidTransientMoveLabels && (
                <span className="text-[10px] text-amber-700 dark:text-amber-300">
                  Invalid touch range
                </span>
              )}
              <div className="ml-auto flex flex-wrap items-center gap-1">
                {transientSaving && (
                  <span className="px-1 text-[10px] text-muted-foreground">
                    Saving…
                  </span>
                )}
                <button
                  type="button"
                  onClick={() =>
                    activeMoveIndex !== null &&
                    setMoveStartFrame(activeMoveIndex, selectedFrame)
                  }
                  disabled={activeMoveIndex === null}
                  className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px] disabled:opacity-40"
                >
                  Touch start
                </button>
                <button
                  type="button"
                  onClick={() =>
                    activeMoveIndex !== null &&
                    setMoveEndFrame(activeMoveIndex, selectedFrame)
                  }
                  disabled={activeMoveIndex === null}
                  className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px] disabled:opacity-40"
                >
                  Touch end
                </button>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto space-y-0.5 rounded border p-1">
              {effectiveMoves.length === 0 ? (
                <div className="px-1.5 py-2 text-[11px] text-muted-foreground">
                  No replay moves available for this clip.
                </div>
              ) : (
                effectiveMoves.map((move, i) => {
                  const annotation = moveAnnotations[i];
                  const moveComplete = annotation
                    ? moveTouchAnnotationComplete(annotation)
                    : false;
                  return (
                    <div
                      key={i}
                      onClick={() => {
                        setActiveMoveIndex(i);
                        setSelectedFrame(move.frame_index);
                      }}
                      className={`flex items-center justify-between gap-2 rounded px-1.5 py-1 text-[11px] ${activeMoveIndex === i ? "bg-primary/10 font-medium" : "hover:bg-muted/50"}`}
                    >
                      <span className="min-w-0">
                        #{i + 1}{" "}
                        <span className="font-medium">
                          {move.san || move.uci}
                        </span>
                        {move.is_manual && (
                          <span className="ml-1 text-[10px] text-amber-600">
                            manual
                          </span>
                        )}
                      </span>
                      <div className="flex shrink-0 items-center gap-1 font-mono text-[10px]">
                        {!moveComplete && (
                          <span className="text-amber-600">!</span>
                        )}
                        {annotation &&
                          annotation.start_frame_index !== null && (
                            <div className="relative pr-1">
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setSelectedFrame(
                                    annotation.start_frame_index!,
                                  );
                                }}
                                className="rounded border px-1.5 py-0.5 text-muted-foreground hover:bg-muted"
                              >
                                f{annotation.start_frame_index}
                              </button>
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  clearMoveStartFrame(i);
                                }}
                                className="absolute -right-1 -top-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border bg-background text-[9px] leading-none text-muted-foreground hover:text-foreground"
                                title="Clear touch start"
                              >
                                ×
                              </button>
                            </div>
                          )}
                        {moveComplete && (
                          <span className="text-muted-foreground">&gt;</span>
                        )}
                        {moveComplete && (
                          <span className="text-muted-foreground">
                            f{annotation?.move_frame_index}
                          </span>
                        )}
                        {moveComplete && (
                          <span className="text-muted-foreground">&gt;</span>
                        )}
                        {annotation && annotation.end_frame_index !== null && (
                          <div className="relative pr-1">
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                setSelectedFrame(annotation.end_frame_index!);
                              }}
                              className="rounded border px-1.5 py-0.5 text-muted-foreground hover:bg-muted"
                            >
                              f{annotation.end_frame_index}
                            </button>
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                clearMoveEndFrame(i);
                              }}
                              className="absolute -right-1 -top-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border bg-background text-[9px] leading-none text-muted-foreground hover:text-foreground"
                              title="Clear touch end"
                            >
                              ×
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Right: Physical board annotation */}
        <div className="space-y-1">
          <div className="flex flex-wrap items-center justify-between gap-1">
            <div className="flex items-center gap-0.5">
              {hasUnsavedChanges ? (
                <span
                  className={`${PANEL_STATUS_CLASS} border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300`}
                >
                  Save needed
                </span>
              ) : existingAnnotation ? (
                <span
                  className={`${PANEL_STATUS_CLASS} border-green-600/40 bg-green-600/10 text-green-700 dark:text-green-300`}
                >
                  Saved
                </span>
              ) : null}
              {existingAnnotation && (
                <button
                  type="button"
                  onClick={() => void handleDelete()}
                  disabled={deleting}
                  className="rounded border border-red-600/40 bg-red-600/10 px-1.5 py-0.5 text-[10px] text-red-700 hover:bg-red-600/20 dark:text-red-300 disabled:opacity-50"
                >
                  {deleting ? "Deleting\u2026" : "Delete"}
                </button>
              )}
              <button
                type="button"
                onClick={resetCorners}
                className={PANEL_BUTTON_CLASS}
              >
                Reset
              </button>
              <button
                type="button"
                onClick={() => void usePrevCorners()}
                disabled={prevFrameCornersRef.current.length !== 4}
                className={PANEL_BUTTON_CLASS}
              >
                Prev corners
              </button>
              <button
                type="button"
                onClick={() => void runDetectCorners()}
                disabled={detecting}
                className={PANEL_BUTTON_CLASS}
              >
                {detecting ? "Detecting\u2026" : "Detect"}
              </button>
              <label className="inline-flex items-center gap-0.5 text-[9px]">
                <input
                  type="checkbox"
                  checked={autoDetect}
                  onChange={(e) => setAutoDetect(e.target.checked)}
                  className="h-3 w-3"
                />
                Auto
              </label>
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
              title={
                selectedFrame < frameCount - 1
                  ? "Save and advance one frame"
                  : "Save this frame"
              }
            >
              {saving
                ? "Saving\u2026"
                : selectedFrame < frameCount - 1
                  ? "Save + Next >"
                  : "Save"}
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
                    const pieceLetter =
                      label !== null && label > 0
                        ? SQUARE_CLASS_NAMES[label]
                        : null;
                    return (
                      <button
                        key={si}
                        type="button"
                        className="relative flex items-center justify-center border border-white/20 hover:bg-blue-500/20"
                        onClick={() => handleSquareClick(si)}
                        title={`${squareName(si)} \u2014 ${label === null ? "unset" : SQUARE_CLASS_NAMES[label]} \u2014 click to cycle`}
                      >
                        {pieceLetter && (
                          <svg
                            viewBox="0 0 45 45"
                            className="pointer-events-none h-[60%] w-[60%] drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]"
                          >
                            {chessPieceSvg(pieceLetter)}
                          </svg>
                        )}
                        {label === 0 && (
                          <span className="pointer-events-none text-white/50 text-[10px] font-bold">
                            {"\u00B7"}
                          </span>
                        )}
                        <span className="pointer-events-none absolute left-0.5 top-0 text-[7px] text-white/70">
                          {squareName(si)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
                {rectifying
                  ? "Rectifying board\u2026"
                  : "Mark four corners on the camera crop to rectify the board."}
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
            ref={(node) => {
              videoRef.current = node;
            }}
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
