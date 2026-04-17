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

    setDraftMoveAnnotations((currentDraft) => {
      if (
        clipChanged ||
        sameMoveAnnotationList(currentDraft, lastBaseline.moveAnnotations)
      ) {
        return baselineMoveAnnotations;
      }
      return mergeDraftMoveAnnotations(currentDraft, baselineMoveAnnotations);
    });
    setDraftOccludedFrameIndices((currentDraft) => {
      if (
        clipChanged ||
        sameNumberList(currentDraft, lastBaseline.occludedFrameIndices)
      ) {
        return baselineOccludedFrameIndices;
      }
      return currentDraft;
    });

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
  const hasUnsavedChanges = useMemo(
    () =>
      !sameMoveAnnotationList(draftMoveAnnotations, baselineMoveAnnotations) ||
      !sameNumberList(draftOccludedFrameIndices, baselineOccludedFrameIndices),
    [
      baselineMoveAnnotations,
      baselineOccludedFrameIndices,
      draftMoveAnnotations,
      draftOccludedFrameIndices,
    ],
  );
  const savePayload = useMemo(
    () => ({
      move_annotations: draftMoveAnnotations,
      hand_occlusion_spans: frameIndicesToSpans(draftOccludedFrameIndices),
    }),
    [draftMoveAnnotations, draftOccludedFrameIndices],
  );
  const savePayloadKey = useMemo(
    () => JSON.stringify(savePayload),
    [savePayload],
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
    if (
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
      setDraftMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, start_frame_index: frameIndex }
            : annotation,
        ),
      );
    },
    [],
  );

  const setMoveEndFrame = useCallback(
    (moveIndex: number, frameIndex: number) => {
      setDraftMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, end_frame_index: frameIndex }
            : annotation,
        ),
      );
    },
    [],
  );

  const clearMoveStartFrame = useCallback((moveIndex: number) => {
    setDraftMoveAnnotations((currentDraft) =>
      currentDraft.map((annotation) =>
        annotation.move_index === moveIndex
          ? { ...annotation, start_frame_index: null }
          : annotation,
      ),
    );
  }, []);

  const clearMoveEndFrame = useCallback((moveIndex: number) => {
    setDraftMoveAnnotations((currentDraft) =>
      currentDraft.map((annotation) =>
        annotation.move_index === moveIndex
          ? { ...annotation, end_frame_index: null }
          : annotation,
      ),
    );
  }, []);

  const isFrameOccluded = useCallback(
    (frameIndex: number) => draftOccludedFrameIndices.includes(frameIndex),
    [draftOccludedFrameIndices],
  );

  const toggleFrameOccluded = useCallback((frameIndex: number) => {
    setDraftOccludedFrameIndices((currentDraft) => {
      if (currentDraft.includes(frameIndex)) {
        return currentDraft.filter((value) => value !== frameIndex);
      }
      return sortNumbers([...currentDraft, frameIndex]);
    });
  }, []);

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
