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

function buildSavePayload(
  moveAnnotations: PhysicalTransientMoveAnnotation[],
  occludedFrameIndices: number[],
) {
  return {
    move_annotations: moveAnnotations,
    hand_occlusion_spans: frameIndicesToSpans(occludedFrameIndices),
  };
}

function draftPayloadKey(
  moveAnnotations: PhysicalTransientMoveAnnotation[],
  occludedFrameIndices: number[],
): string {
  return JSON.stringify(buildSavePayload(moveAnnotations, occludedFrameIndices));
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
  const loadingRef = useRef(true);
  const savingRef = useRef(false);
  const failedPayloadKeyRef = useRef<string | null>(null);
  const hasInvalidMoveLabelsRef = useRef(false);
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
  const savePayloadKey = useMemo(
    () => draftPayloadKey(draftMoveAnnotations, draftOccludedFrameIndices),
    [draftMoveAnnotations, draftOccludedFrameIndices],
  );

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  useEffect(() => {
    failedPayloadKeyRef.current = failedPayloadKey;
  }, [failedPayloadKey]);

  useEffect(() => {
    hasInvalidMoveLabelsRef.current = hasInvalidMoveLabels;
  }, [hasInvalidMoveLabels]);

  const save = useCallback(async () => {
    if (
      pendingBaselineSyncPayloadKeyRef.current !== null ||
      loadingRef.current ||
      savingRef.current ||
      hasInvalidMoveLabelsRef.current ||
      !hasUnsavedChangesRef.current
    ) {
      return false;
    }

    const moveAnnotations = draftMoveAnnotationsRef.current;
    const occludedFrameIndices = draftOccludedFrameIndicesRef.current;
    const payloadKey = draftPayloadKey(moveAnnotations, occludedFrameIndices);
    if (failedPayloadKeyRef.current === payloadKey) {
      return false;
    }

    setSaving(true);
    savingRef.current = true;
    try {
      const result = await api.save({
        clip_path: clipPath,
        ...buildSavePayload(moveAnnotations, occludedFrameIndices),
      });
      const latestDraftPayloadKey = draftPayloadKey(
        draftMoveAnnotationsRef.current,
        draftOccludedFrameIndicesRef.current,
      );
      const savedLatestDraft = latestDraftPayloadKey === payloadKey;
      hasUnsavedChangesRef.current = !savedLatestDraft;
      setHasUnsavedChanges(!savedLatestDraft);
      failedPayloadKeyRef.current = null;
      setFailedPayloadKey(null);
      setSavedAnnotation(result.annotation);
      return true;
    } catch (error) {
      failedPayloadKeyRef.current = payloadKey;
      setFailedPayloadKey(payloadKey);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to save touch/occlusion labels",
      );
      return false;
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }, [api, clipPath]);

  useEffect(() => {
    if (pendingBaselineSyncPayloadKeyRef.current === savePayloadKey) {
      pendingBaselineSyncPayloadKeyRef.current = null;
      void save();
    }
  }, [save, savePayloadKey]);

  const updateMoveAnnotations = useCallback(
    (
      updater: (
        annotations: PhysicalTransientMoveAnnotation[],
      ) => PhysicalTransientMoveAnnotation[],
    ) => {
      failedPayloadKeyRef.current = null;
      setFailedPayloadKey(null);
      const nextDraft = updater(draftMoveAnnotationsRef.current);
      const nextHasUnsavedChanges = hasDraftChanges(
        nextDraft,
        baselineMoveAnnotations,
        draftOccludedFrameIndicesRef.current,
        baselineOccludedFrameIndices,
      );
      draftMoveAnnotationsRef.current = nextDraft;
      hasUnsavedChangesRef.current = nextHasUnsavedChanges;
      pendingBaselineSyncPayloadKeyRef.current = null;
      setDraftMoveAnnotations(nextDraft);
      setHasUnsavedChanges(nextHasUnsavedChanges);
      void save();
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices, save],
  );

  const updateOccludedFrameIndices = useCallback(
    (updater: (frameIndices: number[]) => number[]) => {
      failedPayloadKeyRef.current = null;
      setFailedPayloadKey(null);
      const nextDraft = updater(draftOccludedFrameIndicesRef.current);
      const nextHasUnsavedChanges = hasDraftChanges(
        draftMoveAnnotationsRef.current,
        baselineMoveAnnotations,
        nextDraft,
        baselineOccludedFrameIndices,
      );
      draftOccludedFrameIndicesRef.current = nextDraft;
      hasUnsavedChangesRef.current = nextHasUnsavedChanges;
      pendingBaselineSyncPayloadKeyRef.current = null;
      setDraftOccludedFrameIndices(nextDraft);
      setHasUnsavedChanges(nextHasUnsavedChanges);
      void save();
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices, save],
  );

  const setMoveStartFrame = useCallback(
    (moveIndex: number, frameIndex: number) => {
      updateMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, start_frame_index: frameIndex }
            : annotation,
        ),
      );
    },
    [updateMoveAnnotations],
  );

  const setMoveEndFrame = useCallback(
    (moveIndex: number, frameIndex: number) => {
      updateMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, end_frame_index: frameIndex }
            : annotation,
        ),
      );
    },
    [updateMoveAnnotations],
  );

  const clearMoveStartFrame = useCallback(
    (moveIndex: number) => {
      updateMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, start_frame_index: null }
            : annotation,
        ),
      );
    },
    [updateMoveAnnotations],
  );

  const clearMoveEndFrame = useCallback(
    (moveIndex: number) => {
      updateMoveAnnotations((currentDraft) =>
        currentDraft.map((annotation) =>
          annotation.move_index === moveIndex
            ? { ...annotation, end_frame_index: null }
            : annotation,
        ),
      );
    },
    [updateMoveAnnotations],
  );

  const isFrameOccluded = useCallback(
    (frameIndex: number) => draftOccludedFrameIndices.includes(frameIndex),
    [draftOccludedFrameIndices],
  );

  const toggleFrameOccluded = useCallback(
    (frameIndex: number) => {
      updateOccludedFrameIndices((currentDraft) =>
        currentDraft.includes(frameIndex)
          ? currentDraft.filter((value) => value !== frameIndex)
          : sortNumbers([...currentDraft, frameIndex]),
      );
    },
    [updateOccludedFrameIndices],
  );

  const replaceDraftFromAutoLabel = useCallback(
    (
      moveUpdates: Array<{
        move_index: number;
        uci?: string;
        start_frame_index: number | null;
        end_frame_index: number | null;
        is_capture?: boolean | null;
      }>,
      occlusionSpans: PhysicalHandOcclusionSpan[],
    ) => {
      failedPayloadKeyRef.current = null;
      setFailedPayloadKey(null);
      const updatesByIndex = new Map(
        moveUpdates.map((update) => [update.move_index, update]),
      );
      const nextMoveAnnotations = draftMoveAnnotationsRef.current.map(
        (annotation) => {
          const update = updatesByIndex.get(annotation.move_index);
          if (!update) return annotation;
          if (update.uci && update.uci !== annotation.uci) return annotation;
          return {
            ...annotation,
            start_frame_index: update.start_frame_index,
            end_frame_index: update.end_frame_index,
            is_capture:
              update.is_capture !== undefined
                ? update.is_capture
                : annotation.is_capture,
          };
        },
      );
      const nextOccludedFrameIndices = spansToFrameIndices(
        occlusionSpans,
        frameCount,
      );
      const nextHasUnsavedChanges = hasDraftChanges(
        nextMoveAnnotations,
        baselineMoveAnnotations,
        nextOccludedFrameIndices,
        baselineOccludedFrameIndices,
      );
      draftMoveAnnotationsRef.current = nextMoveAnnotations;
      draftOccludedFrameIndicesRef.current = nextOccludedFrameIndices;
      hasUnsavedChangesRef.current = nextHasUnsavedChanges;
      pendingBaselineSyncPayloadKeyRef.current = null;
      setDraftMoveAnnotations(nextMoveAnnotations);
      setDraftOccludedFrameIndices(nextOccludedFrameIndices);
      setHasUnsavedChanges(nextHasUnsavedChanges);
      void save();
    },
    [baselineMoveAnnotations, baselineOccludedFrameIndices, frameCount, save],
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
    replaceDraftFromAutoLabel,
  };
}
