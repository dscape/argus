"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import {
  deletePhysicalEvalTransientAnnotation,
  deletePhysicalTrainTransientAnnotation,
  getPhysicalEvalTransientAnnotation,
  getPhysicalTrainTransientAnnotation,
  savePhysicalEvalTransientAnnotation,
  savePhysicalTrainTransientAnnotation,
  type PhysicalHandOcclusionSpan,
  type PhysicalTransientAnnotation,
  type PhysicalTransientMoveAnnotation,
} from "@/lib/api";
import type { DetectedMove } from "@/lib/types";

type PhysicalAnnotationSplit = "val" | "train";

interface Props {
  split: PhysicalAnnotationSplit;
  clipPath: string;
  effectiveMoves: DetectedMove[];
  selectedFrame: number;
  setSelectedFrame: Dispatch<SetStateAction<number>>;
  frameCount: number;
}

interface BaselineSnapshot {
  clipPath: string | null;
  moveAnnotations: PhysicalTransientMoveAnnotation[];
  handOcclusionSpans: PhysicalHandOcclusionSpan[];
}

function inferredCapture(move: DetectedMove): boolean | null {
  if (!move.san) return null;
  return move.san.includes("x");
}

function defaultMoveAnnotation(
  move: DetectedMove,
  moveIndex: number,
): PhysicalTransientMoveAnnotation {
  return {
    move_index: moveIndex,
    uci: move.uci,
    san: move.san,
    move_frame_index: move.frame_index,
    side_to_move: move.side_to_move,
    fen_before: move.fen_before,
    fen_after: move.fen_after,
    start_frame_index: null,
    end_frame_index: move.frame_index,
    is_capture: inferredCapture(move),
  };
}

function sameMoveIdentity(
  annotation: PhysicalTransientMoveAnnotation,
  move: DetectedMove,
  moveIndex: number,
): boolean {
  if (annotation.move_index !== moveIndex || annotation.uci !== move.uci)
    return false;
  if ((annotation.fen_before ?? null) !== (move.fen_before ?? null))
    return false;
  if ((annotation.fen_after ?? null) !== (move.fen_after ?? null)) return false;
  return true;
}

function pickEditableFields(
  annotation: PhysicalTransientMoveAnnotation,
): Pick<
  PhysicalTransientMoveAnnotation,
  "start_frame_index" | "end_frame_index" | "is_capture"
> {
  return {
    start_frame_index: annotation.start_frame_index,
    end_frame_index: annotation.end_frame_index,
    is_capture: annotation.is_capture,
  };
}

function buildMoveAnnotations(
  moves: DetectedMove[],
  sourceAnnotations: PhysicalTransientMoveAnnotation[],
): PhysicalTransientMoveAnnotation[] {
  return moves.map((move, moveIndex) => {
    const baseline = defaultMoveAnnotation(move, moveIndex);
    const existing =
      sourceAnnotations.find((annotation) =>
        sameMoveIdentity(annotation, move, moveIndex),
      ) ??
      sourceAnnotations.find(
        (annotation) =>
          annotation.move_index === moveIndex && annotation.uci === move.uci,
      );
    return existing
      ? { ...baseline, ...pickEditableFields(existing) }
      : baseline;
  });
}

function mergeDraftMoveAnnotations(
  draftAnnotations: PhysicalTransientMoveAnnotation[],
  baselineAnnotations: PhysicalTransientMoveAnnotation[],
): PhysicalTransientMoveAnnotation[] {
  return baselineAnnotations.map((baseline) => {
    const draft = draftAnnotations.find(
      (annotation) =>
        annotation.move_index === baseline.move_index &&
        annotation.uci === baseline.uci,
    );
    return draft ? { ...baseline, ...pickEditableFields(draft) } : baseline;
  });
}

function sameMoveAnnotation(
  left: PhysicalTransientMoveAnnotation,
  right: PhysicalTransientMoveAnnotation,
): boolean {
  return (
    left.move_index === right.move_index &&
    left.uci === right.uci &&
    left.san === right.san &&
    left.move_frame_index === right.move_frame_index &&
    left.side_to_move === right.side_to_move &&
    left.fen_before === right.fen_before &&
    left.fen_after === right.fen_after &&
    left.start_frame_index === right.start_frame_index &&
    left.end_frame_index === right.end_frame_index &&
    left.is_capture === right.is_capture
  );
}

function sameMoveAnnotationList(
  left: PhysicalTransientMoveAnnotation[],
  right: PhysicalTransientMoveAnnotation[],
): boolean {
  return (
    left.length === right.length &&
    left.every((annotation, index) =>
      sameMoveAnnotation(annotation, right[index]),
    )
  );
}

function sortHandOcclusionSpans(
  spans: PhysicalHandOcclusionSpan[],
): PhysicalHandOcclusionSpan[] {
  return [...spans].sort(
    (left, right) =>
      left.start_frame_index - right.start_frame_index ||
      left.end_frame_index - right.end_frame_index,
  );
}

function sameHandOcclusionSpan(
  left: PhysicalHandOcclusionSpan,
  right: PhysicalHandOcclusionSpan,
): boolean {
  return (
    left.start_frame_index === right.start_frame_index &&
    left.end_frame_index === right.end_frame_index
  );
}

function sameHandOcclusionSpanList(
  left: PhysicalHandOcclusionSpan[],
  right: PhysicalHandOcclusionSpan[],
): boolean {
  const sortedLeft = sortHandOcclusionSpans(left);
  const sortedRight = sortHandOcclusionSpans(right);
  return (
    sortedLeft.length === sortedRight.length &&
    sortedLeft.every((span, index) =>
      sameHandOcclusionSpan(span, sortedRight[index]),
    )
  );
}

function moveLabel(
  moveIndex: number,
  annotation: PhysicalTransientMoveAnnotation,
): string {
  const moveText = annotation.san || annotation.uci;
  return `#${moveIndex + 1} ${moveText} · replay f${annotation.move_frame_index}`;
}

function frameLabel(frameIndex: number | null): string {
  return frameIndex === null ? "—" : `f${frameIndex}`;
}

function clampFrame(frameIndex: number, frameCount: number): number {
  return Math.min(Math.max(frameIndex, 0), Math.max(frameCount - 1, 0));
}

export function PhysicalTransientAnnotationPanel({
  split,
  clipPath,
  effectiveMoves,
  selectedFrame,
  setSelectedFrame,
  frameCount,
}: Props) {
  const api = useMemo(
    () =>
      split === "train"
        ? {
            get: getPhysicalTrainTransientAnnotation,
            save: savePhysicalTrainTransientAnnotation,
            remove: deletePhysicalTrainTransientAnnotation,
          }
        : {
            get: getPhysicalEvalTransientAnnotation,
            save: savePhysicalEvalTransientAnnotation,
            remove: deletePhysicalEvalTransientAnnotation,
          },
    [split],
  );
  const [savedAnnotation, setSavedAnnotation] =
    useState<PhysicalTransientAnnotation | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [activeMoveIndex, setActiveMoveIndex] = useState<number | null>(null);
  const [draftMoveAnnotations, setDraftMoveAnnotations] = useState<
    PhysicalTransientMoveAnnotation[]
  >([]);
  const [draftHandOcclusionSpans, setDraftHandOcclusionSpans] = useState<
    PhysicalHandOcclusionSpan[]
  >([]);
  const lastBaselineRef = useRef<BaselineSnapshot>({
    clipPath: null,
    moveAnnotations: [],
    handOcclusionSpans: [],
  });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const annotation = await api.get(clipPath);
        if (!cancelled) {
          setSavedAnnotation(annotation);
        }
      } catch (error) {
        if (!cancelled) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to load transient labels",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [api, clipPath]);

  const baselineMoveAnnotations = useMemo(
    () =>
      buildMoveAnnotations(
        effectiveMoves,
        savedAnnotation?.move_annotations ?? [],
      ),
    [effectiveMoves, savedAnnotation],
  );
  const baselineHandOcclusionSpans = useMemo(
    () => sortHandOcclusionSpans(savedAnnotation?.hand_occlusion_spans ?? []),
    [savedAnnotation],
  );

  useEffect(() => {
    const lastBaseline = lastBaselineRef.current;
    const clipChanged = lastBaseline.clipPath !== clipPath;

    setDraftMoveAnnotations((currentDraft) => {
      if (
        clipChanged ||
        sameMoveAnnotationList(currentDraft, lastBaseline.moveAnnotations)
      ) {
        return baselineMoveAnnotations;
      }
      return mergeDraftMoveAnnotations(currentDraft, baselineMoveAnnotations);
    });
    setDraftHandOcclusionSpans((currentDraft) => {
      if (
        clipChanged ||
        sameHandOcclusionSpanList(currentDraft, lastBaseline.handOcclusionSpans)
      ) {
        return baselineHandOcclusionSpans;
      }
      return currentDraft;
    });

    lastBaselineRef.current = {
      clipPath,
      moveAnnotations: baselineMoveAnnotations,
      handOcclusionSpans: baselineHandOcclusionSpans,
    };
  }, [baselineHandOcclusionSpans, baselineMoveAnnotations, clipPath]);

  const selectedMoveIndex = useMemo(
    () =>
      effectiveMoves.findIndex((move) => move.frame_index === selectedFrame),
    [effectiveMoves, selectedFrame],
  );

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

  const activeMoveAnnotation =
    activeMoveIndex === null
      ? null
      : (draftMoveAnnotations[activeMoveIndex] ?? null);
  const activeMoveBaseline =
    activeMoveIndex === null
      ? null
      : (baselineMoveAnnotations[activeMoveIndex] ?? null);
  const orderedHandOcclusionSpans = useMemo(
    () => sortHandOcclusionSpans(draftHandOcclusionSpans),
    [draftHandOcclusionSpans],
  );

  const hasInvalidMoveLabels = useMemo(
    () =>
      draftMoveAnnotations.some(
        (annotation) =>
          (annotation.start_frame_index !== null &&
            annotation.start_frame_index >= frameCount) ||
          (annotation.end_frame_index !== null &&
            annotation.end_frame_index >= frameCount) ||
          (annotation.start_frame_index !== null &&
            annotation.end_frame_index !== null &&
            annotation.end_frame_index < annotation.start_frame_index),
      ),
    [draftMoveAnnotations, frameCount],
  );
  const hasInvalidHandOcclusionSpans = useMemo(
    () =>
      orderedHandOcclusionSpans.some(
        (span) =>
          span.start_frame_index >= frameCount ||
          span.end_frame_index >= frameCount ||
          span.end_frame_index < span.start_frame_index,
      ),
    [frameCount, orderedHandOcclusionSpans],
  );
  const hasUnsavedChanges = useMemo(
    () =>
      !sameMoveAnnotationList(draftMoveAnnotations, baselineMoveAnnotations) ||
      !sameHandOcclusionSpanList(
        orderedHandOcclusionSpans,
        baselineHandOcclusionSpans,
      ),
    [
      baselineHandOcclusionSpans,
      baselineMoveAnnotations,
      draftMoveAnnotations,
      orderedHandOcclusionSpans,
    ],
  );

  const moveMismatchWarning = useMemo(() => {
    if (!savedAnnotation) return null;
    if (savedAnnotation.move_annotations.length !== effectiveMoves.length) {
      return "Saved move labels were recorded against a different move count. Review before saving.";
    }
    const hasMismatch = savedAnnotation.move_annotations.some(
      (annotation, index) => {
        const move = effectiveMoves[index];
        return !move || annotation.uci !== move.uci;
      },
    );
    return hasMismatch
      ? "Saved move labels no longer line up with the current replay moves. Review before saving."
      : null;
  }, [effectiveMoves, savedAnnotation]);

  const jumpToFrame = useCallback(
    (frameIndex: number | null) => {
      if (frameIndex === null) return;
      setSelectedFrame(clampFrame(frameIndex, frameCount));
    },
    [frameCount, setSelectedFrame],
  );

  const updateActiveMove = useCallback(
    (
      updater: (
        current: PhysicalTransientMoveAnnotation,
      ) => PhysicalTransientMoveAnnotation,
    ) => {
      setDraftMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) => {
          if (annotation.move_index !== activeMoveIndex) return annotation;
          return updater(annotation);
        }),
      );
    },
    [activeMoveIndex],
  );

  const resetActiveMove = useCallback(() => {
    if (activeMoveIndex === null || !activeMoveBaseline) return;
    setDraftMoveAnnotations((currentDraft) =>
      currentDraft.map((annotation) =>
        annotation.move_index === activeMoveIndex
          ? activeMoveBaseline
          : annotation,
      ),
    );
  }, [activeMoveBaseline, activeMoveIndex]);

  const addHandOcclusionSpan = useCallback(() => {
    setDraftHandOcclusionSpans((current) =>
      sortHandOcclusionSpans([
        ...current,
        { start_frame_index: selectedFrame, end_frame_index: selectedFrame },
      ]),
    );
  }, [selectedFrame]);

  const updateHandOcclusionSpan = useCallback(
    (
      spanIndex: number,
      updater: (
        current: PhysicalHandOcclusionSpan,
      ) => PhysicalHandOcclusionSpan,
    ) => {
      setDraftHandOcclusionSpans((currentDraft) =>
        sortHandOcclusionSpans(
          currentDraft.map((span, index) =>
            index === spanIndex ? updater(span) : span,
          ),
        ),
      );
    },
    [],
  );

  const deleteHandOcclusionSpan = useCallback((spanIndex: number) => {
    setDraftHandOcclusionSpans((currentDraft) =>
      currentDraft.filter((_, index) => index !== spanIndex),
    );
  }, []);

  const handleSave = useCallback(async () => {
    if (hasInvalidMoveLabels || hasInvalidHandOcclusionSpans) {
      toast.error("Fix invalid frame ranges before saving transient labels");
      return;
    }

    setSaving(true);
    try {
      const result = await api.save({
        clip_path: clipPath,
        move_annotations: draftMoveAnnotations,
        hand_occlusion_spans: orderedHandOcclusionSpans,
      });
      setSavedAnnotation(result.annotation);
      toast.success("Transient labels saved");
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to save transient labels",
      );
    } finally {
      setSaving(false);
    }
  }, [
    api,
    clipPath,
    draftMoveAnnotations,
    hasInvalidHandOcclusionSpans,
    hasInvalidMoveLabels,
    orderedHandOcclusionSpans,
  ]);

  const handleDelete = useCallback(async () => {
    if (!savedAnnotation) return;
    setDeleting(true);
    try {
      await api.remove(clipPath);
      setSavedAnnotation(null);
      toast.success("Transient labels deleted");
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to delete transient labels",
      );
    } finally {
      setDeleting(false);
    }
  }, [api, clipPath, savedAnnotation]);

  const startLabeledCount = useMemo(
    () =>
      draftMoveAnnotations.filter(
        (annotation) => annotation.start_frame_index !== null,
      ).length,
    [draftMoveAnnotations],
  );
  const settledLabeledCount = useMemo(
    () =>
      draftMoveAnnotations.filter(
        (annotation) => annotation.end_frame_index !== null,
      ).length,
    [draftMoveAnnotations],
  );
  const captureLabeledCount = useMemo(
    () =>
      draftMoveAnnotations.filter(
        (annotation) => annotation.is_capture !== null,
      ).length,
    [draftMoveAnnotations],
  );

  return (
    <div className="space-y-3 rounded-xl border bg-muted/20 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-medium">Transient move labels</h3>
            <Badge variant="outline" className="text-[10px]">
              Current frame f{selectedFrame}
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              {orderedHandOcclusionSpans.length} occlusion spans
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            Label move start, settled frame, capture/non-capture, and
            hand-occluded spans.
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {hasUnsavedChanges && (
            <Badge
              variant="outline"
              className="border-amber-500/40 bg-amber-500/10 text-[10px] text-amber-700 dark:text-amber-300"
            >
              Unsaved changes
            </Badge>
          )}
          {savedAnnotation && !hasUnsavedChanges && (
            <Badge
              variant="outline"
              className="border-green-600/40 bg-green-600/10 text-[10px] text-green-700 dark:text-green-300"
            >
              Saved
            </Badge>
          )}
          {savedAnnotation && (
            <button
              type="button"
              onClick={() => void handleDelete()}
              disabled={deleting}
              className="inline-flex h-8 items-center justify-center rounded border border-red-600/40 bg-red-600/10 px-3 text-xs text-red-700 hover:bg-red-600/20 dark:text-red-300 disabled:opacity-40"
            >
              {deleting ? "Deleting…" : "Delete labels"}
            </button>
          )}
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={loading || saving || !hasUnsavedChanges}
            className="inline-flex h-8 items-center justify-center rounded border bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
          >
            {saving ? "Saving…" : "Save labels"}
          </button>
        </div>
      </div>

      {(moveMismatchWarning ||
        hasInvalidMoveLabels ||
        hasInvalidHandOcclusionSpans) && (
        <div className="space-y-1 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-xs text-amber-800 dark:text-amber-200">
          {moveMismatchWarning && <div>{moveMismatchWarning}</div>}
          {hasInvalidMoveLabels && (
            <div>One or more move labels have invalid frame ranges.</div>
          )}
          {hasInvalidHandOcclusionSpans && (
            <div>
              One or more hand-occlusion spans have invalid frame ranges.
            </div>
          )}
        </div>
      )}

      <div className="grid gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="space-y-3 rounded-lg border bg-background/70 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-xs font-medium">Active move</div>
            <button
              type="button"
              disabled={activeMoveIndex === null || activeMoveIndex <= 0}
              onClick={() =>
                setActiveMoveIndex((current) =>
                  current === null ? 0 : Math.max(current - 1, 0),
                )
              }
              className="inline-flex h-8 items-center justify-center rounded border px-2 text-xs disabled:opacity-40"
            >
              ← Prev
            </button>
            <select
              value={activeMoveIndex ?? ""}
              onChange={(event) =>
                setActiveMoveIndex(Number(event.target.value))
              }
              disabled={draftMoveAnnotations.length === 0}
              className="h-8 min-w-[18rem] rounded border bg-background px-2 text-xs shadow-sm outline-none disabled:opacity-40"
            >
              {draftMoveAnnotations.length === 0 && (
                <option value="">No moves</option>
              )}
              {draftMoveAnnotations.map((annotation, index) => (
                <option key={`${annotation.uci}-${index}`} value={index}>
                  {moveLabel(index, annotation)}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={
                activeMoveIndex === null ||
                activeMoveIndex >= draftMoveAnnotations.length - 1
              }
              onClick={() =>
                setActiveMoveIndex((current) =>
                  current === null
                    ? 0
                    : Math.min(current + 1, draftMoveAnnotations.length - 1),
                )
              }
              className="inline-flex h-8 items-center justify-center rounded border px-2 text-xs disabled:opacity-40"
            >
              Next →
            </button>
            <button
              type="button"
              disabled={!activeMoveAnnotation}
              onClick={() =>
                jumpToFrame(activeMoveAnnotation?.move_frame_index ?? null)
              }
              className="inline-flex h-8 items-center justify-center rounded border px-2 text-xs disabled:opacity-40"
            >
              Go to replay frame
            </button>
          </div>

          {activeMoveAnnotation ? (
            <>
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="secondary" className="text-[10px]">
                  {activeMoveAnnotation.san || activeMoveAnnotation.uci}
                </Badge>
                <span>
                  Replay frame{" "}
                  {frameLabel(activeMoveAnnotation.move_frame_index)}
                </span>
                <span>
                  Start {frameLabel(activeMoveAnnotation.start_frame_index)}
                </span>
                <span>
                  Settled {frameLabel(activeMoveAnnotation.end_frame_index)}
                </span>
                <span>
                  Capture{" "}
                  {activeMoveAnnotation.is_capture === null
                    ? "—"
                    : activeMoveAnnotation.is_capture
                      ? "yes"
                      : "no"}
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() =>
                    updateActiveMove((current) => ({
                      ...current,
                      start_frame_index: selectedFrame,
                    }))
                  }
                  className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs"
                >
                  Use current as start
                </button>
                <button
                  type="button"
                  onClick={() =>
                    updateActiveMove((current) => ({
                      ...current,
                      end_frame_index: selectedFrame,
                    }))
                  }
                  className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs"
                >
                  Use current as settled
                </button>
                <button
                  type="button"
                  disabled={activeMoveAnnotation.start_frame_index === null}
                  onClick={() =>
                    jumpToFrame(activeMoveAnnotation.start_frame_index)
                  }
                  className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs disabled:opacity-40"
                >
                  Jump to start
                </button>
                <button
                  type="button"
                  disabled={activeMoveAnnotation.end_frame_index === null}
                  onClick={() =>
                    jumpToFrame(activeMoveAnnotation.end_frame_index)
                  }
                  className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs disabled:opacity-40"
                >
                  Jump to settled
                </button>
                <button
                  type="button"
                  onClick={resetActiveMove}
                  className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs"
                >
                  Reset move labels
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-muted-foreground">Capture</span>
                <select
                  value={
                    activeMoveAnnotation.is_capture === null
                      ? "unknown"
                      : activeMoveAnnotation.is_capture
                        ? "capture"
                        : "non-capture"
                  }
                  onChange={(event) => {
                    const nextValue = event.target.value;
                    updateActiveMove((current) => ({
                      ...current,
                      is_capture:
                        nextValue === "unknown"
                          ? null
                          : nextValue === "capture",
                    }));
                  }}
                  className="h-8 rounded border bg-background px-2 text-xs shadow-sm outline-none"
                >
                  <option value="unknown">Unknown</option>
                  <option value="non-capture">Non-capture</option>
                  <option value="capture">Capture</option>
                </select>
              </div>
            </>
          ) : (
            <div className="text-xs text-muted-foreground">
              No replay moves available for this clip.
            </div>
          )}
        </div>

        <div className="space-y-3 rounded-lg border bg-background/70 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-medium">Hand-occluded spans</div>
            <button
              type="button"
              onClick={addHandOcclusionSpan}
              className="inline-flex h-8 items-center justify-center rounded border px-3 text-xs"
            >
              Add span @ current frame
            </button>
          </div>

          {orderedHandOcclusionSpans.length === 0 ? (
            <div className="text-xs text-muted-foreground">
              No hand-occluded spans labeled.
            </div>
          ) : (
            <div className="space-y-2">
              {orderedHandOcclusionSpans.map((span, spanIndex) => (
                <div
                  key={`${span.start_frame_index}-${span.end_frame_index}-${spanIndex}`}
                  className="space-y-2 rounded border p-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <span className="font-medium">
                      Span {spanIndex + 1}: {frameLabel(span.start_frame_index)}{" "}
                      → {frameLabel(span.end_frame_index)}
                    </span>
                    <div className="flex flex-wrap gap-1">
                      <button
                        type="button"
                        onClick={() => jumpToFrame(span.start_frame_index)}
                        className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px]"
                      >
                        Jump start
                      </button>
                      <button
                        type="button"
                        onClick={() => jumpToFrame(span.end_frame_index)}
                        className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px]"
                      >
                        Jump end
                      </button>
                      <button
                        type="button"
                        onClick={() => deleteHandOcclusionSpan(spanIndex)}
                        className="inline-flex h-7 items-center justify-center rounded border border-red-600/40 bg-red-600/10 px-2 text-[11px] text-red-700 hover:bg-red-600/20 dark:text-red-300"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <button
                      type="button"
                      onClick={() =>
                        updateHandOcclusionSpan(spanIndex, (current) => ({
                          ...current,
                          start_frame_index: selectedFrame,
                        }))
                      }
                      className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px]"
                    >
                      Current → start
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        updateHandOcclusionSpan(spanIndex, (current) => ({
                          ...current,
                          end_frame_index: selectedFrame,
                        }))
                      }
                      className="inline-flex h-7 items-center justify-center rounded border px-2 text-[11px]"
                    >
                      Current → end
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
        <span>
          {startLabeledCount}/{draftMoveAnnotations.length} starts labeled
        </span>
        <span>
          {settledLabeledCount}/{draftMoveAnnotations.length} settled frames
          labeled
        </span>
        <span>
          {captureLabeledCount}/{draftMoveAnnotations.length} capture labels set
        </span>
      </div>
    </div>
  );
}
