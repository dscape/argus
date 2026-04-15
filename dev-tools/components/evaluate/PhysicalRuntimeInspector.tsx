"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import PhysicalRuntimeCard from "@/components/evaluate/PhysicalRuntimeCard";
import {
  createPhysicalRuntimeSession,
  inspectPhysicalRuntimeFrames,
  listPhysicalRuntimeModels,
  listPhysicalRuntimeSessions,
  physicalRuntimeSessionImageUrl,
  samplePhysicalRuntimeFrames,
  savePhysicalRuntimeEval,
  updatePhysicalRuntimePins,
  type PhysicalRuntimeEvalResult,
  type PhysicalRuntimeModelOption,
  type PhysicalRuntimeSession,
} from "@/lib/api";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: {
    non_empty_accuracy?: number;
    exact_match_rate?: number;
    elapsed_ms_avg?: number;
    images_per_minute?: number;
    stateless_square_accuracy?: number;
    stateless_non_empty_accuracy?: number;
    stateless_exact_match_rate?: number;
    model_path?: string;
    model_label?: string;
  } | null;
}

interface PhysicalRuntimeInspectorProps {
  initialSession?: {
    id: string;
    results: PhysicalRuntimeEvalResult[];
    square_accuracy: number | null;
    non_empty_accuracy: number | null;
    exact_match_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
    model_label?: string | null;
    model_path?: string | null;
  };
}

interface BatchSummary {
  totalFrames: number;
  totalSquares: number;
  temporalCorrectSquares: number;
  statelessCorrectSquares: number;
  temporalErrorCount: number;
  statelessErrorCount: number;
  nonEmptySquares: number;
  temporalNonEmptyCorrect: number;
  statelessNonEmptyCorrect: number;
  temporalSquareAccuracy: number;
  statelessSquareAccuracy: number;
  temporalNonEmptyAccuracy: number | null;
  statelessNonEmptyAccuracy: number | null;
  temporalExactFrames: number;
  statelessExactFrames: number;
  temporalBetterFrames: number;
  temporalWorseFrames: number;
  avgElapsedMs: number;
}

function parsePositiveInteger(value: string, fallback: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const rounded = Math.floor(parsed);
  return rounded > 0 ? rounded : fallback;
}

function resultKey(result: PhysicalRuntimeEvalResult): string {
  return result.annotation_id;
}

function formatPercent(value: number | null | undefined) {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatModelLabel(value: string | null | undefined) {
  if (!value) return null;
  if (!value.includes("/")) return value;

  const parts = value.split("/");
  const filename = parts.at(-1) ?? value;
  if (filename === "best.pt" && parts.at(-2) === "physical") {
    return "default";
  }
  if (filename === "board_probe.pt" || filename === "linear_probe.pt") {
    return parts.at(-2) ?? value;
  }
  return filename;
}

function thumbnailSrc(
  result: PhysicalRuntimeEvalResult,
  sessionId: string | null,
): string | null {
  if (result.thumbnail_b64) return `data:image/png;base64,${result.thumbnail_b64}`;
  if (sessionId && result.thumbnail_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.thumbnail_filename);
  }
  return null;
}

function computeSummary(results: PhysicalRuntimeEvalResult[]): BatchSummary | null {
  if (results.length === 0) return null;

  const totalFrames = results.length;
  const totalSquares = totalFrames * 64;
  const temporalCorrectSquares = results.reduce(
    (sum, result) => sum + (64 - result.temporal_error_count),
    0,
  );
  const statelessCorrectSquares = results.reduce(
    (sum, result) => sum + (64 - result.stateless_error_count),
    0,
  );
  const temporalErrorCount = results.reduce(
    (sum, result) => sum + result.temporal_error_count,
    0,
  );
  const statelessErrorCount = results.reduce(
    (sum, result) => sum + result.stateless_error_count,
    0,
  );
  const nonEmptySquares = results.reduce(
    (sum, result) => sum + result.non_empty_square_count,
    0,
  );
  const temporalNonEmptyCorrect = results.reduce(
    (sum, result) => sum + result.temporal_non_empty_correct_count,
    0,
  );
  const statelessNonEmptyCorrect = results.reduce(
    (sum, result) => sum + result.stateless_non_empty_correct_count,
    0,
  );
  const temporalExactFrames = results.filter((result) => result.temporal_exact_match).length;
  const statelessExactFrames = results.filter((result) => result.stateless_exact_match).length;
  const temporalBetterFrames = results.filter(
    (result) => result.temporal_error_count < result.stateless_error_count,
  ).length;
  const temporalWorseFrames = results.filter(
    (result) => result.temporal_error_count > result.stateless_error_count,
  ).length;
  const avgElapsedMs =
    results.reduce((sum, result) => sum + result.elapsed_ms, 0) / totalFrames;

  return {
    totalFrames,
    totalSquares,
    temporalCorrectSquares,
    statelessCorrectSquares,
    temporalErrorCount,
    statelessErrorCount,
    nonEmptySquares,
    temporalNonEmptyCorrect,
    statelessNonEmptyCorrect,
    temporalSquareAccuracy: temporalCorrectSquares / totalSquares,
    statelessSquareAccuracy: statelessCorrectSquares / totalSquares,
    temporalNonEmptyAccuracy:
      nonEmptySquares > 0 ? temporalNonEmptyCorrect / nonEmptySquares : null,
    statelessNonEmptyAccuracy:
      nonEmptySquares > 0 ? statelessNonEmptyCorrect / nonEmptySquares : null,
    temporalExactFrames,
    statelessExactFrames,
    temporalBetterFrames,
    temporalWorseFrames,
    avgElapsedMs,
  };
}

function InfoIcon({ tip }: { tip: string }) {
  return (
    <span className="relative group inline-flex items-center ml-1 cursor-default">
      <span className="inline-flex items-center justify-center w-4 h-4 rounded-full border text-[10px] text-muted-foreground select-none">
        i
      </span>
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1 w-56 rounded bg-popover border text-popover-foreground text-xs px-2 py-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity z-50 whitespace-normal">
        {tip}
      </span>
    </span>
  );
}

export default function PhysicalRuntimeInspector({
  initialSession,
}: PhysicalRuntimeInspectorProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [sampleSize, setSampleSize] = useState(8);
  const [results, setResults] = useState<PhysicalRuntimeEvalResult[]>(
    initialSession?.results ?? [],
  );
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [evalHistory, setEvalHistory] = useState<EvalPoint[]>([]);
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(
    new Set(
      initialSession?.pin_state
        ? Object.entries(initialSession.pin_state)
            .filter(([, value]) => value)
            .map(([key]) => key)
        : [],
    ),
  );
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [sessionId, setSessionId] = useState<string | null>(
    initialSession?.id ?? null,
  );
  const [recentSessions, setRecentSessions] = useState<PhysicalRuntimeSession[]>([]);
  const [showSessionList, setShowSessionList] = useState(false);
  const [emptyMessage, setEmptyMessage] = useState<string | null>(null);
  const [modelOptions, setModelOptions] = useState<PhysicalRuntimeModelOption[]>([]);
  const [modelPath, setModelPath] = useState<string>(initialSession?.model_path ?? "");

  const inspectedIds = useRef<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const sessionListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHistory();
    listPhysicalRuntimeModels()
      .then(({ models }) => {
        setModelOptions(models);
        setModelPath((current) => {
          if (current) return current;
          return models.find((model) => model.is_default)?.path ?? models[0]?.path ?? "";
        });
      })
      .catch((error) => {
        console.warn("Failed to load physical runtime models:", error);
      });
  }, []);

  useEffect(() => {
    if (initialSession || !searchParams.toString()) return;
    if (!["clip", "start", "count"].some((key) => searchParams.has(key))) return;
    router.replace("/evaluate/physical", { scroll: false });
  }, [initialSession, router, searchParams]);

  useEffect(() => {
    if (!showSessionList) return;
    const handleClick = (event: MouseEvent) => {
      if (
        sessionListRef.current &&
        !sessionListRef.current.contains(event.target as Node)
      ) {
        setShowSessionList(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showSessionList]);

  async function fetchHistory() {
    try {
      const res = await fetch("/api/models/evaluations?model_name=physical");
      if (res.ok) {
        const data = await res.json();
        setEvalHistory(data.evaluations);
      }
    } catch (error) {
      console.warn("Failed to fetch evaluation history:", error);
    }
  }

  async function fetchRecentSessions() {
    try {
      const { sessions } = await listPhysicalRuntimeSessions(20);
      setRecentSessions(sessions);
      setShowSessionList(true);
    } catch (error) {
      console.warn("Failed to fetch sessions:", error);
    }
  }

  const selectedModel = useMemo(
    () => modelOptions.find((option) => option.path === modelPath) ?? null,
    [modelOptions, modelPath],
  );
  const selectedModelLabel =
    selectedModel?.label ?? formatModelLabel(initialSession?.model_label ?? modelPath) ?? "default";

  const togglePin = useCallback(
    (annotationId: string) => {
      setPinnedIds((prev) => {
        const next = new Set(prev);
        if (next.has(annotationId)) next.delete(annotationId);
        else next.add(annotationId);

        if (sessionId) {
          updatePhysicalRuntimePins(sessionId, {
            [annotationId]: !prev.has(annotationId),
          }).catch(() => {});
        }
        return next;
      });
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(annotationId);
        return next;
      });
    },
    [sessionId],
  );

  const toggleExpand = useCallback((annotationId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(annotationId)) next.delete(annotationId);
      else next.add(annotationId);
      return next;
    });
  }, []);

  async function runBatch() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setResults([]);
    setProgress({ current: 0, total: 0 });
    setPinnedIds(new Set());
    setExpandedIds(new Set());
    setSessionId(null);
    setEmptyMessage(null);

    const collected: PhysicalRuntimeEvalResult[] = [];
    const autoPinned = new Set<string>();
    const batchStartTime = performance.now();

    try {
      const exclude = Array.from(inspectedIds.current);
      const { frames } = await samplePhysicalRuntimeFrames(
        sampleSize,
        exclude,
        controller.signal,
      );

      if (frames.length === 0) {
        setEmptyMessage(
          exclude.length > 0
            ? "No more held-out physical validation frames are available in this browser session. Reload to start over."
            : "No held-out physical validation frames are available. Annotate frames under data/physical/val first.",
        );
        return;
      }

      setProgress({ current: 0, total: frames.length });

      const batchResults = await inspectPhysicalRuntimeFrames(
        frames.map((frame) => frame.annotation_id),
        {
          model_path: modelPath || undefined,
          signal: controller.signal,
        },
      );
      collected.push(...batchResults);
      setResults(batchResults);
      batchResults.forEach((result) => {
        inspectedIds.current.add(result.annotation_id);
        if (!result.temporal_exact_match) {
          autoPinned.add(result.annotation_id);
        }
      });
      setPinnedIds(new Set(autoPinned));
      setProgress({ current: batchResults.length, total: frames.length });

      const summary = computeSummary(collected);
      if (!summary) return;

      const elapsedMin = (performance.now() - batchStartTime) / 60000;
      const imagesPerMinute = elapsedMin > 0 ? Math.round(summary.totalFrames / elapsedMin) : null;

      const saveResult = await savePhysicalRuntimeEval({
        square_accuracy: summary.temporalSquareAccuracy,
        non_empty_accuracy: summary.temporalNonEmptyAccuracy,
        exact_match_rate: summary.temporalExactFrames / summary.totalFrames,
        sample_size: summary.totalFrames,
        elapsed_ms_avg: summary.avgElapsedMs,
        images_per_minute: imagesPerMinute,
        stateless_square_accuracy: summary.statelessSquareAccuracy,
        stateless_non_empty_accuracy: summary.statelessNonEmptyAccuracy,
        stateless_exact_match_rate: summary.statelessExactFrames / summary.totalFrames,
        notes: selectedModelLabel,
        model_path: modelPath || undefined,
      });
      await fetchHistory();

      const pinStateObj: Record<string, boolean> = {};
      autoPinned.forEach((annotationId) => {
        pinStateObj[annotationId] = true;
      });

      const { session_id } = await createPhysicalRuntimeSession({
        results: collected,
        square_accuracy: summary.temporalSquareAccuracy,
        non_empty_accuracy: summary.temporalNonEmptyAccuracy,
        exact_match_rate: summary.temporalExactFrames / summary.totalFrames,
        sample_size: summary.totalFrames,
        pin_state: pinStateObj,
        evaluation_id: saveResult.id,
      });
      setSessionId(session_id);
      router.replace(`/evaluate/physical/${session_id}`, { scroll: false });
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }

  const summary = useMemo(() => computeSummary(results), [results]);

  const { pinned, unpinned } = useMemo(() => {
    const nextPinned: PhysicalRuntimeEvalResult[] = [];
    const nextUnpinned: PhysicalRuntimeEvalResult[] = [];
    for (const result of results) {
      if (pinnedIds.has(resultKey(result))) nextPinned.push(result);
      else nextUnpinned.push(result);
    }
    return { pinned: nextPinned, unpinned: nextUnpinned };
  }, [results, pinnedIds]);

  const chartData = [...evalHistory].reverse().map((evaluation) => ({
    date: new Date(evaluation.evaluated_at).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
    accuracy: Math.round(evaluation.accuracy * 1000) / 10,
    exact_match_rate:
      evaluation.per_class?.exact_match_rate != null
        ? Math.round(evaluation.per_class.exact_match_rate * 1000) / 10
        : null,
    model_label:
      formatModelLabel(
        evaluation.per_class?.model_label ?? evaluation.per_class?.model_path ?? evaluation.notes,
      ) ?? "default",
    sample_size: evaluation.sample_size,
  }));

  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm text-muted-foreground">
          Sample from held-out physical validation frames:
          <InfoIcon tip="Samples rectified held-out physical board annotations and compares both stateless and deployed temporal runtime predictions against ground truth one frame at a time." />
        </span>
        <input
          type="number"
          value={sampleSize}
          onChange={(event) =>
            setSampleSize(parsePositiveInteger(event.target.value, 8))
          }
          min={1}
          max={200}
          className="w-16 px-2 py-1.5 border rounded text-sm"
        />
        <input
          list="physical-runtime-models"
          value={modelPath}
          onChange={(event) => setModelPath(event.target.value)}
          placeholder="weights/physical/best.pt"
          className="min-w-72 flex-1 max-w-xl px-2 py-1.5 border rounded text-sm font-mono"
        />
        <datalist id="physical-runtime-models">
          {modelOptions.map((option) => (
            <option key={option.path} value={option.path}>
              {option.label}
            </option>
          ))}
        </datalist>
        <button
          onClick={runBatch}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Inspecting..." : "Sample & Inspect"}
        </button>
        <span className="text-xs text-muted-foreground font-mono">
          model: {selectedModelLabel}
        </span>

        <div className="flex-1" />

        <div className="relative" ref={sessionListRef}>
          <button
            onClick={fetchRecentSessions}
            className="px-3 py-1.5 border rounded text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {sessionId ? <span className="font-mono">{sessionId}</span> : "Sessions"}
          </button>
          {showSessionList && (
            <div className="absolute right-0 top-full mt-1 z-50 w-80 bg-background border rounded-lg shadow-lg overflow-hidden">
              {recentSessions.length > 0 ? (
                <div className="max-h-64 overflow-y-auto">
                  {recentSessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => {
                        setShowSessionList(false);
                        router.push(`/evaluate/physical/${session.id}`);
                      }}
                      className={`w-full text-left px-3 py-2 text-xs hover:bg-muted/50 transition-colors border-b last:border-b-0 ${
                        session.id === sessionId ? "bg-muted/30" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-muted-foreground truncate">{session.id}</span>
                        {session.square_accuracy != null && (
                          <span className="font-medium shrink-0">
                            {(session.square_accuracy * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <div className="text-muted-foreground mt-0.5">
                        {new Date(session.created_at).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                        {" · "}n={session.sample_size}
                        {session.model_label
                          ? ` · ${formatModelLabel(session.model_label)}`
                          : ""}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="px-3 py-2 text-xs text-muted-foreground">
                  No saved sessions yet.
                </div>
              )}
            </div>
          )}
        </div>

        {sessionId && (
          <button
            onClick={() => navigator.clipboard.writeText(window.location.href)}
            className="px-3 py-1.5 border rounded text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Copy session URL"
          >
            Copy URL
          </button>
        )}
      </div>

      {emptyMessage && (
        <div className="border rounded-lg px-3 py-2 text-sm text-muted-foreground">
          {emptyMessage}
        </div>
      )}

      {loading && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Inspecting physical frames...</span>
            <div className="flex items-center gap-2">
              <span>
                {progress.current}/{progress.total}
              </span>
              <button
                onClick={() => abortRef.current?.abort()}
                className="px-2 py-0.5 rounded bg-destructive text-destructive-foreground text-xs hover:bg-destructive/90"
              >
                Stop
              </button>
            </div>
          </div>
          <div className="h-2 bg-muted rounded overflow-hidden">
            <div
              className="h-full bg-foreground rounded transition-all duration-300"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {chartData.length >= 2 && (
        <div className="border rounded-lg p-3">
          <h3 className="text-sm font-medium mb-2">Physical runtime history</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 10 }}
                tickLine={false}
                tickFormatter={(value) => `${value}%`}
                width={40}
              />
              <Tooltip
                formatter={(value, name) => [
                  `${value}%`,
                  name === "accuracy" ? "Square accuracy" : "Exact boards",
                ]}
                labelFormatter={(label, payload) => {
                  const datum = payload?.[0]?.payload;
                  return `${label} — ${datum?.model_label ?? "default"} (n=${datum?.sample_size ?? "?"})`;
                }}
              />
              <Line
                type="monotone"
                dataKey="accuracy"
                name="Square accuracy"
                stroke="currentColor"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="exact_match_rate"
                name="Exact boards"
                stroke="#22c55e"
                strokeWidth={1.5}
                dot={{ r: 2 }}
                strokeDasharray="4 4"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {summary && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <div className="border rounded-lg p-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Temporal square</p>
            <p className="mt-1 text-xl font-semibold">
              {formatPercent(summary.temporalSquareAccuracy)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {summary.temporalCorrectSquares}/{summary.totalSquares} correct squares
            </p>
          </div>
          <div className="border rounded-lg p-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Exact boards</p>
            <p className="mt-1 text-xl font-semibold">
              {summary.temporalExactFrames}/{summary.totalFrames}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {formatPercent(summary.temporalExactFrames / summary.totalFrames)} exact matches
            </p>
          </div>
          <div className="border rounded-lg p-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Temporal vs single</p>
            <p className="mt-1 text-xl font-semibold">
              {summary.temporalSquareAccuracy >= summary.statelessSquareAccuracy ? "+" : ""}
              {((summary.temporalSquareAccuracy - summary.statelessSquareAccuracy) * 100).toFixed(1)} pts
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              better on {summary.temporalBetterFrames} frames · worse on {summary.temporalWorseFrames}
            </p>
          </div>
          <div className="border rounded-lg p-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Avg inspect time</p>
            <p className="mt-1 text-xl font-semibold">{summary.avgElapsedMs.toFixed(1)}ms</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {summary.statelessErrorCount - summary.temporalErrorCount >= 0 ? "+" : ""}
              {summary.statelessErrorCount - summary.temporalErrorCount} squares recovered vs single
            </p>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-4">
          {pinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                Pinned ({pinned.length})
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {pinned.map((result) => (
                  <PhysicalRuntimeCard
                    key={result.annotation_id}
                    result={result}
                    pinned
                    sessionId={sessionId}
                    onPin={() => togglePin(result.annotation_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {unpinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                {pinned.length > 0 ? `Others (${unpinned.length})` : `${results.length} results`}
              </p>

              <div className="flex flex-wrap gap-1.5">
                {unpinned
                  .filter((result) => !expandedIds.has(resultKey(result)))
                  .map((result) => {
                    const key = resultKey(result);
                    const src = thumbnailSrc(result, sessionId);
                    const title = `${result.clip_filename} · frame ${result.frame_index}\n` +
                      `${result.temporal_exact_match ? "✓ exact" : `✗ ${result.temporal_error_count} wrong`}\n` +
                      `single ${result.stateless_error_count} wrong · temp ${result.temporal_error_count} wrong`;

                    return (
                      <button
                        key={key}
                        onClick={() =>
                          result.temporal_exact_match ? toggleExpand(key) : togglePin(key)
                        }
                        title={title}
                        className="relative w-16 h-16 rounded border overflow-hidden flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30"
                      >
                        {src ? (
                          <img
                            src={src}
                            alt={key}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full bg-muted flex items-center justify-center text-[9px] text-muted-foreground px-1 text-center">
                            f{result.frame_index}
                          </div>
                        )}
                        <div
                          className={`absolute inset-0 ${
                            result.temporal_exact_match ? "bg-green-500/25" : "bg-red-500/35"
                          }`}
                        />
                        <span
                          className={`absolute top-0 left-0 text-[9px] leading-none px-0.5 py-px ${
                            result.temporal_exact_match
                              ? "bg-green-500/80 text-white"
                              : "bg-red-500/80 text-white"
                          }`}
                        >
                          {result.temporal_exact_match ? "✓" : "✗"}
                        </span>
                        {!result.temporal_exact_match && (
                          <span className="absolute bottom-0 right-0 text-[9px] leading-none bg-black/60 text-white px-0.5 py-px">
                            {result.temporal_error_count}
                          </span>
                        )}
                      </button>
                    );
                  })}
              </div>

              {unpinned.filter((result) => expandedIds.has(resultKey(result))).length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                  {unpinned
                    .filter((result) => expandedIds.has(resultKey(result)))
                    .map((result) => {
                      const key = resultKey(result);
                      return (
                        <div key={key} className="relative">
                          <button
                            onClick={() => toggleExpand(key)}
                            className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded bg-muted/80 text-muted-foreground hover:text-foreground text-xs"
                            title="Collapse"
                          >
                            ✕
                          </button>
                          <PhysicalRuntimeCard
                            result={result}
                            pinned={false}
                            sessionId={sessionId}
                            onPin={() => togglePin(key)}
                          />
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
