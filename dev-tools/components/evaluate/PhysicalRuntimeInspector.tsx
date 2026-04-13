"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getModelVersions,
  listPhysicalEvalClips,
  renderPhysicalRuntimeVisualization,
  type PhysicalEvalClip,
  type PhysicalRuntimeVisualizationResponse,
} from "@/lib/api";

function parsePositiveInteger(value: string | null, fallback: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const rounded = Math.floor(parsed);
  return rounded > 0 ? rounded : fallback;
}

function parseNonNegativeInteger(value: string | null, fallback: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const rounded = Math.floor(parsed);
  return rounded >= 0 ? rounded : fallback;
}

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatDelta(value: number) {
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
}

export default function PhysicalRuntimeInspector() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialClipPath = searchParams.get("clip");
  const initialFrameStart = parseNonNegativeInteger(searchParams.get("start"), 0);
  const initialFrameCount = parsePositiveInteger(searchParams.get("count"), 8);

  const [clips, setClips] = useState<PhysicalEvalClip[]>([]);
  const [selectedClipPath, setSelectedClipPath] = useState<string | null>(initialClipPath);
  const [frameStart, setFrameStart] = useState(initialFrameStart);
  const [frameCount, setFrameCount] = useState(initialFrameCount);
  const [loadingClips, setLoadingClips] = useState(true);
  const [rendering, setRendering] = useState(false);
  const [result, setResult] = useState<PhysicalRuntimeVisualizationResponse | null>(null);
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const [lastRequestKey, setLastRequestKey] = useState<string | null>(null);

  const selectedClip = useMemo(
    () => clips.find((clip) => clip.clip_path === selectedClipPath) ?? null,
    [clips, selectedClipPath],
  );

  const summary = useMemo(() => {
    if (!result || result.frames.length === 0) return null;
    const statelessErrors = result.frames.map((frame) => frame.stateless_error_count);
    const temporalErrors = result.frames.map((frame) => frame.temporal_error_count);
    const statelessConfidences = result.frames.map((frame) => frame.stateless_mean_confidence);
    const temporalConfidences = result.frames.map((frame) => frame.temporal_mean_confidence);
    const temporalBetterFrames = result.frames.filter(
      (frame) => frame.temporal_error_count < frame.stateless_error_count,
    ).length;
    const temporalWorseFrames = result.frames.filter(
      (frame) => frame.temporal_error_count > frame.stateless_error_count,
    ).length;

    return {
      avgStatelessErrors: average(statelessErrors),
      avgTemporalErrors: average(temporalErrors),
      avgErrorDelta: average(
        result.frames.map(
          (frame) => frame.temporal_error_count - frame.stateless_error_count,
        ),
      ),
      avgStatelessConfidence: average(statelessConfidences),
      avgTemporalConfidence: average(temporalConfidences),
      temporalBetterFrames,
      temporalWorseFrames,
    };
  }, [result]);

  const runRender = useCallback(
    async (overrides?: { clipPath?: string | null; frameStart?: number; frameCount?: number }) => {
      const clipPath = overrides?.clipPath ?? selectedClipPath;
      if (!clipPath) {
        toast.error("Select a held-out eval clip first");
        return;
      }

      const requestedFrameStart = Math.max(0, overrides?.frameStart ?? frameStart);
      const requestedFrameCount = Math.max(1, overrides?.frameCount ?? frameCount);
      const requestKey = `${clipPath}:${requestedFrameStart}:${requestedFrameCount}`;
      setLastRequestKey(requestKey);
      setRendering(true);
      try {
        const nextResult = await renderPhysicalRuntimeVisualization({
          clip_path: clipPath,
          frame_start: requestedFrameStart,
          frame_count: requestedFrameCount,
        });
        setSelectedClipPath(nextResult.clip_path);
        setFrameStart(nextResult.frame_start);
        setFrameCount(nextResult.frame_count);
        setResult(nextResult);

        const params = new URLSearchParams();
        params.set("clip", nextResult.clip_path);
        params.set("start", String(nextResult.frame_start));
        params.set("count", String(nextResult.frame_count));
        router.replace(`/evaluate/physical?${params.toString()}`, { scroll: false });
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to render runtime view");
      } finally {
        setRendering(false);
      }
    },
    [frameCount, frameStart, router, selectedClipPath],
  );

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const [clipData, versions] = await Promise.all([
          listPhysicalEvalClips(),
          getModelVersions(),
        ]);
        if (cancelled) return;

        setClips(clipData.clips);
        setModelVersion(versions.physical ?? null);

        const fallbackClipPath =
          clipData.clips.find((clip) => clip.clip_path === initialClipPath)?.clip_path
          ?? clipData.clips[0]?.clip_path
          ?? null;
        setSelectedClipPath(fallbackClipPath);
      } catch (error) {
        if (!cancelled) {
          toast.error(error instanceof Error ? error.message : "Failed to load held-out clips");
        }
      } finally {
        if (!cancelled) setLoadingClips(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [initialClipPath]);

  useEffect(() => {
    if (loadingClips || !selectedClipPath || result !== null || lastRequestKey !== null) return;
    void runRender({ clipPath: selectedClipPath, frameStart, frameCount });
  }, [frameCount, frameStart, lastRequestKey, loadingClips, result, runRender, selectedClipPath]);

  const canStepBackward = frameStart > 0;
  const canStepForward = selectedClip
    ? frameStart + frameCount < (selectedClip.num_frames ?? result?.available_frame_count ?? 0)
    : true;

  if (loadingClips) {
    return <div className="text-sm text-muted-foreground">Loading held-out physical eval clips…</div>;
  }

  if (clips.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No held-out physical eval clips found in <code>data/argus/train_real</code>.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Physical runtime</CardTitle>
          <CardDescription>
            Inspect the current physical board reader on held-out eval clips frame by frame.
            {modelVersion ? ` Runtime weights: ${modelVersion}.` : ""}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1.75fr)_repeat(2,minmax(0,0.5fr))_auto]">
            <label className="space-y-2 text-sm">
              <span className="font-medium">Clip</span>
              <select
                className="h-10 w-full rounded-md border bg-background px-3"
                value={selectedClipPath ?? ""}
                onChange={(event) => {
                  setSelectedClipPath(event.target.value);
                  setFrameStart(0);
                  setResult(null);
                  setLastRequestKey(null);
                }}
              >
                {clips.map((clip) => (
                  <option key={clip.clip_path} value={clip.clip_path}>
                    {clip.filename}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium">Frame start</span>
              <input
                className="h-10 w-full rounded-md border bg-background px-3"
                min={0}
                step={1}
                type="number"
                value={frameStart}
                onChange={(event) => setFrameStart(parseNonNegativeInteger(event.target.value, 0))}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium">Frame count</span>
              <input
                className="h-10 w-full rounded-md border bg-background px-3"
                min={1}
                step={1}
                type="number"
                value={frameCount}
                onChange={(event) => setFrameCount(parsePositiveInteger(event.target.value, 8))}
              />
            </label>
            <div className="flex items-end">
              <Button className="w-full" disabled={rendering || !selectedClipPath} onClick={() => void runRender()}>
                {rendering ? "Rendering…" : "Render"}
              </Button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <Button
              variant="outline"
              disabled={rendering || !canStepBackward}
              onClick={() => void runRender({ frameStart: Math.max(0, frameStart - frameCount) })}
            >
              Previous window
            </Button>
            <Button
              variant="outline"
              disabled={rendering || !canStepForward}
              onClick={() => void runRender({ frameStart: frameStart + frameCount })}
            >
              Next window
            </Button>
            {selectedClip && (
              <span>
                {selectedClip.filename} · {selectedClip.annotated_frame_count}
                {selectedClip.num_frames ? `/${selectedClip.num_frames}` : ""} annotated frames
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {summary && result && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
          <MetricCard label="Frames rendered" value={`${result.frame_count}`} sublabel={`of ${result.available_frame_count}`} />
          <MetricCard label="Avg stateless errors" value={summary.avgStatelessErrors.toFixed(1)} />
          <MetricCard label="Avg temporal errors" value={summary.avgTemporalErrors.toFixed(1)} />
          <MetricCard label="Temporal - stateless" value={formatDelta(summary.avgErrorDelta)} />
          <MetricCard
            label="Temporal better / worse"
            value={`${summary.temporalBetterFrames}/${summary.temporalWorseFrames}`}
            sublabel="frames"
          />
          <MetricCard
            label="Mean conf single / temp"
            value={`${summary.avgStatelessConfidence.toFixed(2)} / ${summary.avgTemporalConfidence.toFixed(2)}`}
          />
        </div>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Contact sheet</CardTitle>
            <CardDescription>
              Red squares are wrong, yellow borders mark real board changes, blue borders mark
              prediction changes.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <img
              alt="Physical runtime contact sheet"
              className="w-full rounded-md border"
              src={`data:image/png;base64,${result.contact_sheet_b64}`}
            />
          </CardContent>
        </Card>
      )}

      {result && (
        <div className="space-y-4">
          {result.frames.map((frame) => (
            <Card key={frame.annotation_id}>
              <CardHeader>
                <CardTitle className="text-base">Frame {frame.frame_index.toString().padStart(4, "0")}</CardTitle>
                <CardDescription>
                  gtΔ={frame.gt_change_count ?? "-"} · singleΔ={frame.stateless_change_count ?? "-"}
                  {" "}· tempΔ={frame.temporal_change_count ?? "-"} · single err={frame.stateless_error_count}
                  {" "}· temp err={frame.temporal_error_count}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <img
                  alt={`Physical runtime frame ${frame.frame_index}`}
                  className="w-full rounded-md border"
                  loading="lazy"
                  src={`data:image/png;base64,${frame.image_b64}`}
                />
                <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2 xl:grid-cols-4">
                  <div>Stateless confidence: {frame.stateless_mean_confidence.toFixed(2)}</div>
                  <div>Temporal confidence: {frame.temporal_mean_confidence.toFixed(2)}</div>
                  <div>Rectified board: <code>{frame.board_path.split("/").at(-1)}</code></div>
                  <div>Annotation: <code>{frame.annotation_id}</code></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold tracking-tight">{value}</div>
        {sublabel ? <div className="text-xs text-muted-foreground">{sublabel}</div> : null}
      </CardContent>
    </Card>
  );
}
