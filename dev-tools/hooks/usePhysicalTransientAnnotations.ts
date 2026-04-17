import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import {
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

interface BaselineSnapshot {
  clipPath: string | null;
  moveAnnotations: PhysicalTransientMoveAnnotation[];
  occludedFrameIndices: number[];
}

interface Args {
  split: PhysicalAnnotationSplit;
  clipPath: string;
  effectiveMoves: DetectedMove[];
  frameCount: number;
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
    end_frame_index: null,
    is_capture: inferredCapture(move),
  };
}

function sameMoveIdentity(
  annotation: PhysicalTransientMoveAnnotation,
  move: DetectedMove,
  moveIndex: number,
): boolean {
  return (
    annotation.move_index === moveIndex &&
    annotation.uci === move.uci &&
    (annotation.fen_before ?? null) === (move.fen_before ?? null) &&
    (annotation.fen_after ?? null) === (move.fen_after ?? null)
  );
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

function sortNumbers(values: number[]): number[] {
  return [...values].sort((left, right) => left - right);
}

function sameNumberList(left: number[], right: number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function mergeDraftOccludedFrameIndices(
  draftFrameIndices: number[],
  baselineFrameIndices: number[],
  nextBaselineFrameIndices: number[],
): number[] {
  const draftSet = new Set(draftFrameIndices);
  const baselineSet = new Set(baselineFrameIndices);
  const nextBaselineSet = new Set(nextBaselineFrameIndices);
  const changedFrameIndices = new Set<number>([
    ...draftFrameIndices,
    ...baselineFrameIndices,
  ]);

  for (const frameIndex of changedFrameIndices) {
    if (draftSet.has(frameIndex) === baselineSet.has(frameIndex)) {
      continue;
    }
    if (nextBaselineSet.has(frameIndex)) {
      nextBaselineSet.delete(frameIndex);
      continue;
    }
    nextBaselineSet.add(frameIndex);
  }

  return sortNumbers([...nextBaselineSet]);
}

function hasDraftChanges(
  moveAnnotations: PhysicalTransientMoveAnnotation[],
  baselineMoveAnnotations: PhysicalTransientMoveAnnotation[],
  occludedFrameIndices: number[],
  baselineOccludedFrameIndices: number[],
): boolean {
  return (
    !sameMoveAnnotationList(moveAnnotations, baselineMoveAnnotations) ||
    !sameNumberList(occludedFrameIndices, baselineOccludedFrameIndices)
  );
}

function draftPayloadKey(
  moveAnnotations: PhysicalTransientMoveAnnotation[],
  occludedFrameIndices: number[],
): string {
  return JSON.stringify({
    move_annotations: moveAnnotations,
    hand_occlusion_spans: frameIndicesToSpans(occludedFrameIndices),
  });
}

function spansToFrameIndices(
  spans: PhysicalHandOcclusionSpan[],
  frameCount: number,
): number[] {
  const frameIndices = new Set<number>();
  for (const span of spans) {
    for (
      let frameIndex = Math.max(0, span.start_frame_index);
      frameIndex <= Math.min(frameCount - 1, span.end_frame_index);
      frameIndex += 1
    ) {
      frameIndices.add(frameIndex);
    }
  }
  return sortNumbers([...frameIndices]);
}

function frameIndicesToSpans(
  frameIndices: number[],
): PhysicalHandOcclusionSpan[] {
  if (frameIndices.length === 0) return [];

  const sortedFrameIndices = sortNumbers(frameIndices);
  const spans: PhysicalHandOcclusionSpan[] = [];
  let spanStart = sortedFrameIndices[0];
  let previousFrameIndex = sortedFrameIndices[0];

  for (let index = 1; index < sortedFrameIndices.length; index += 1) {
    const frameIndex = sortedFrameIndices[index];
    if (frameIndex === previousFrameIndex + 1) {
      previousFrameIndex = frameIndex;
      continue;
    }

    spans.push({
      start_frame_index: spanStart,
      end_frame_index: previousFrameIndex,
    });
    spanStart = frameIndex;
    previousFrameIndex = frameIndex;
  }

  spans.push({
    start_frame_index: spanStart,
    end_frame_index: previousFrameIndex,
  });
  return spans;
}

export function usePhysicalTransientAnnotations({
  split,
  clipPath,
  effectiveMoves,
  frameCount,
}: Args) {
  const api = useMemo(
    () =>
      split === "train"
        ? {
            get: getPhysicalTrainTransientAnnotation,
            save: savePhysicalTrainTransientAnnotation,
          }
        : {
            get: getPhysicalEvalTransientAnnotation,
            save: savePhysicalEvalTransientAnnotation,
          },
    [split],
  );
  const [savedAnnotation, setSavedAnnotation] =
    useState<PhysicalTransientAnnotation | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [failedPayloadKey, setFailedPayloadKey] = useState<string | null>(null);
  const [draftMoveAnnotations, setDraftMoveAnnotations] = useState<
    PhysicalTransientMoveAnnotation[]
  >([]);
  const [draftOccludedFrameIndices, setDraftOccludedFrameIndices] = useState<
    number[]
  >([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const draftMoveAnnotationsRef = useRef<PhysicalTransientMoveAnnotation[]>([]);
  const draftOccludedFrameIndicesRef = useRef<number[]>([]);
  const hasUnsavedChangesRef = useRef(false);
  const pendingBaselineSyncPayloadKeyRef = useRef<string | null>(null);
  const lastBaselineRef = useRef<BaselineSnapshot>({
    clipPath: null,
    moveAnnotations: [],
    occludedFrameIndices: [],
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
              : "Failed to load touch/occlusion labels",
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
  const baselineOccludedFrameIndices = useMemo(
    () =>
      spansToFrameIndices(
        savedAnnotation?.hand_occlusion_spans ?? [],
        frameCount,
      ),
    [frameCount, savedAnnotation],
  );

  useEffect(() => {
    const lastBaseline = lastBaselineRef.current;
    const clipChanged = lastBaseline.clipPath !== clipPath;
    const currentDraftMoveAnnotations = draftMoveAnnotationsRef.current;
    const currentDraftOccludedFrameIndices =
      draftOccludedFrameIndicesRef.current;
    const keepLocalMoveEdits =
      !clipChanged &&
      hasUnsavedChangesRef.current &&
      !sameMoveAnnotationList(currentDraftMoveAnnotations, lastBaseline.moveAnnotations);
    const keepLocalOcclusionEdits =
      !clipChanged &&
      hasUnsavedChangesRef.current &&
      !sameNumberList(
        currentDraftOccludedFrameIndices,
        lastBaseline.occludedFrameIndices,
      );
    const nextDraftMoveAnnotations = keepLocalMoveEdits
      ? mergeDraftMoveAnnotations(
          currentDraftMoveAnnotations,
          baselineMoveAnnotations,
        )
      : baselineMoveAnnotations;
    const nextDraftOccludedFrameIndices = keepLocalOcclusionEdits
      ? mergeDraftOccludedFrameIndices(
          currentDraftOccludedFrameIndices,
          lastBaseline.occludedFrameIndices,
          baselineOccludedFrameIndices,
        )
      : baselineOccludedFrameIndices;
    const nextHasUnsavedChanges = hasDraftChanges(
      nextDraftMoveAnnotations,
      baselineMoveAnnotations,
      nextDraftOccludedFrameIndices,
      baselineOccludedFrameIndices,
    );

    pendingBaselineSyncPayloadKeyRef.current = draftPayloadKey(
      nextDraftMoveAnnotations,
      nextDraftOccludedFrameIndices,
    );
    draftMoveAnnotationsRef.current = nextDraftMoveAnnotations;
    draftOccludedFrameIndicesRef.current = nextDraftOccludedFrameIndices;
    hasUnsavedChangesRef.current = nextHasUnsavedChanges;
    setDraftMoveAnnotations(nextDraftMoveAnnotations);
    setDraftOccludedFrameIndices(nextDraftOccludedFrameIndices);
    setHasUnsavedChanges(nextHasUnsavedChanges);

    lastBaselineRef.current = {
      clipPath,
      moveAnnotations: baselineMoveAnnotations,
      occludedFrameIndices: baselineOccludedFrameIndices,
    };
  }, [baselineMoveAnnotations, baselineOccludedFrameIndices, clipPath]);

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
  const savePayload = useMemo(
    () => ({
      move_annotations: draftMoveAnnotations,
      hand_occlusion_spans: frameIndicesToSpans(draftOccludedFrameIndices),
    }),
    [draftMoveAnnotations, draftOccludedFrameIndices],
  );
  const savePayloadKey = useMemo(
    () => draftPayloadKey(draftMoveAnnotations, draftOccludedFrameIndices),
    [draftMoveAnnotations, draftOccludedFrameIndices],
  );

  const save = useCallback(async () => {
    if (hasInvalidMoveLabels) {
      return false;
    }

    setSaving(true);
    try {
      const result = await api.save({
        clip_path: clipPath,
        ...savePayload,
      });
      hasUnsavedChangesRef.current = false;
      setHasUnsavedChanges(false);
      setSavedAnnotation(result.annotation);
      setFailedPayloadKey(null);
      return true;
    } catch (error) {
      setFailedPayloadKey(savePayloadKey);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to save touch/occlusion labels",
      );
      return false;
    } finally {
      setSaving(false);
    }
  }, [api, clipPath, hasInvalidMoveLabels, savePayload, savePayloadKey]);

  useEffect(() => {
    if (pendingBaselineSyncPayloadKeyRef.current === savePayloadKey) {
      pendingBaselineSyncPayloadKeyRef.current = null;
    }
  }, [savePayloadKey]);

  useEffect(() => {
    if (
      pendingBaselineSyncPayloadKeyRef.current !== null ||
      loading ||
      saving ||
      hasInvalidMoveLabels ||
      !hasUnsavedChanges ||
      failedPayloadKey === savePayloadKey
    ) {
      return;
    }
    void save();
  }, [
    failedPayloadKey,
    hasInvalidMoveLabels,
    hasUnsavedChanges,
    loading,
    save,
    savePayloadKey,
    saving,
  ]);

  const setMoveStartFrame = useCallback(
    (moveIndex: number, frameIndex: number) => {
      setDraftMoveAnnotations((currentDraft) => {
        const nextDraft = currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, start_frame_index: frameIndex }
            : annotation,
        );
        const nextHasUnsavedChanges = hasDraftChanges(
          nextDraft,
          baselineMoveAnnotations,
          draftOccludedFrameIndicesRef.current,
          baselineOccludedFrameIndices,
        );
        draftMoveAnnotationsRef.current = nextDraft;
        hasUnsavedChangesRef.current = nextHasUnsavedChanges;
        setHasUnsavedChanges(nextHasUnsavedChanges);
        return nextDraft;
      });
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices],
  );

  const setMoveEndFrame = useCallback(
    (moveIndex: number, frameIndex: number) => {
      setDraftMoveAnnotations((currentDraft) => {
        const nextDraft = currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, end_frame_index: frameIndex }
            : annotation,
        );
        const nextHasUnsavedChanges = hasDraftChanges(
          nextDraft,
          baselineMoveAnnotations,
          draftOccludedFrameIndicesRef.current,
          baselineOccludedFrameIndices,
        );
        draftMoveAnnotationsRef.current = nextDraft;
        hasUnsavedChangesRef.current = nextHasUnsavedChanges;
        setHasUnsavedChanges(nextHasUnsavedChanges);
        return nextDraft;
      });
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices],
  );

  const clearMoveStartFrame = useCallback(
    (moveIndex: number) => {
      setDraftMoveAnnotations((currentDraft) => {
        const nextDraft = currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, start_frame_index: null }
            : annotation,
        );
        const nextHasUnsavedChanges = hasDraftChanges(
          nextDraft,
          baselineMoveAnnotations,
          draftOccludedFrameIndicesRef.current,
          baselineOccludedFrameIndices,
        );
        draftMoveAnnotationsRef.current = nextDraft;
        hasUnsavedChangesRef.current = nextHasUnsavedChanges;
        setHasUnsavedChanges(nextHasUnsavedChanges);
        return nextDraft;
      });
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices],
  );

  const clearMoveEndFrame = useCallback(
    (moveIndex: number) => {
      setDraftMoveAnnotations((currentDraft) => {
        const nextDraft = currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, end_frame_index: null }
            : annotation,
        );
        const nextHasUnsavedChanges = hasDraftChanges(
          nextDraft,
          baselineMoveAnnotations,
          draftOccludedFrameIndicesRef.current,
          baselineOccludedFrameIndices,
        );
        draftMoveAnnotationsRef.current = nextDraft;
        hasUnsavedChangesRef.current = nextHasUnsavedChanges;
        setHasUnsavedChanges(nextHasUnsavedChanges);
        return nextDraft;
      });
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices],
  );

  const isFrameOccluded = useCallback(
    (frameIndex: number) => draftOccludedFrameIndices.includes(frameIndex),
    [draftOccludedFrameIndices],
  );

  const toggleFrameOccluded = useCallback(
    (frameIndex: number) => {
      setDraftOccludedFrameIndices((currentDraft) => {
        const nextDraft = currentDraft.includes(frameIndex)
          ? currentDraft.filter((value) => value !== frameIndex)
          : sortNumbers([...currentDraft, frameIndex]);
        const nextHasUnsavedChanges = hasDraftChanges(
          draftMoveAnnotationsRef.current,
          baselineMoveAnnotations,
          nextDraft,
          baselineOccludedFrameIndices,
        );
        draftOccludedFrameIndicesRef.current = nextDraft;
        hasUnsavedChangesRef.current = nextHasUnsavedChanges;
        setHasUnsavedChanges(nextHasUnsavedChanges);
        return nextDraft;
      });
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices],
  );

  return {
    loading,
    saving,
    hasInvalidMoveLabels,
    hasUnsavedChanges,
    moveAnnotations: draftMoveAnnotations,
    save,
    setMoveStartFrame,
    setMoveEndFrame,
    clearMoveStartFrame,
    clearMoveEndFrame,
    isFrameOccluded,
    toggleFrameOccluded,
  };
}
