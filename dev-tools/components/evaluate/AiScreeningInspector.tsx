"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import VideoCard, {
  type InspectResult,
  computeAgreement,
  classColor,
} from "@/components/evaluate/VideoCard";
import { youtubeThumb } from "@/components/video-shared";
import {
  createScreeningSession,
  listScreeningSessions,
  updateSessionPins,
  type ScreeningSession,
} from "@/lib/api";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: Record<string, unknown> | null;
}

interface AiScreeningInspectorProps {
  initialSession?: {
    id: string;
    results: InspectResult[];
    accuracy: number | null;
    per_class: Record<string, { correct: number; total: number }> | null;
    pin_state: Record<string, boolean>;
    model_version: string | null;
    created_at: string;
  };
}

export default function AiScreeningInspector({
  initialSession,
}: AiScreeningInspectorProps) {
  const router = useRouter();
  const [sampleSize, setSampleSize] = useState(20);
  const [results, setResults] = useState<InspectResult[]>(
    initialSession?.results ?? [],
  );
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [evalHistory, setEvalHistory] = useState<EvalPoint[]>([]);
  const [modelVersion, setModelVersion] = useState<string | null>(
    initialSession?.model_version ?? null,
  );
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(
    new Set(
      initialSession?.pin_state
        ? Object.entries(initialSession.pin_state)
            .filter(([, v]) => v)
            .map(([k]) => k)
        : [],
    ),
  );
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [sessionId, setSessionId] = useState<string | null>(
    initialSession?.id ?? null,
  );
  const [recentSessions, setRecentSessions] = useState<ScreeningSession[]>([]);
  const [showSessionList, setShowSessionList] = useState(false);
  const inspectedIds = useRef<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const sessionListRef = useRef<HTMLDivElement>(null);

  // Fetch eval history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  // Close session list on click outside
  useEffect(() => {
    if (!showSessionList) return;
    const handleClick = (e: MouseEvent) => {
      if (
        sessionListRef.current &&
        !sessionListRef.current.contains(e.target as Node)
      ) {
        setShowSessionList(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showSessionList]);

  async function fetchHistory() {
    try {
      const res = await fetch(
        "/api/models/evaluations?model_name=ai_screening",
      );
      if (res.ok) {
        const data = await res.json();
        setEvalHistory(data.evaluations);
      }
    } catch (e) {
      console.warn("Failed to fetch evaluation history:", e);
    }
  }

  async function fetchRecentSessions() {
    try {
      const { sessions } = await listScreeningSessions(20);
      setRecentSessions(sessions);
      setShowSessionList(true);
    } catch (e) {
      console.warn("Failed to fetch sessions:", e);
    }
  }

  const togglePin = useCallback(
    (videoId: string) => {
      setPinnedIds((prev) => {
        const next = new Set(prev);
        if (next.has(videoId)) next.delete(videoId);
        else next.add(videoId);

        // Persist to server if we have a session
        if (sessionId) {
          updateSessionPins(sessionId, { [videoId]: !prev.has(videoId) }).catch(
            () => {},
          );
        }
        return next;
      });
      // If unpinning, also collapse
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(videoId);
        return next;
      });
    },
    [sessionId],
  );

  const toggleExpand = useCallback((videoId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) next.delete(videoId);
      else next.add(videoId);
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
    setModelVersion(null);
    setPinnedIds(new Set());
    setExpandedIds(new Set());
    setSessionId(null);

    const collected: InspectResult[] = [];
    const autoPinned = new Set<string>();
    let detectedModelVersion: string | null = null;
    const batchStartTime = performance.now();

    try {
      // Step 1: get sample video IDs (excluding previously inspected)
      const excludeParam =
        inspectedIds.current.size > 0
          ? `&exclude=${Array.from(inspectedIds.current).join(",")}`
          : "";
      const sampleRes = await fetch(
        `/api/models/ai-screening/sample?limit=${sampleSize}${excludeParam}`,
        { signal: controller.signal },
      );
      if (!sampleRes.ok) throw new Error(await sampleRes.text());
      const { video_ids } = await sampleRes.json();

      setProgress({ current: 0, total: video_ids.length });

      // Step 2: inspect each video individually for progress
      for (let i = 0; i < video_ids.length; i++) {
        if (controller.signal.aborted) break;
        const res = await fetch("/api/models/ai-screening/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: video_ids[i] }),
          signal: controller.signal,
        });
        if (res.ok) {
          const result: InspectResult = await res.json();
          collected.push(result);
          setResults((prev) => [...prev, result]);
          inspectedIds.current.add(video_ids[i]);
          if (result.model_version) {
            detectedModelVersion = result.model_version;
            setModelVersion(result.model_version);
          }
          // Auto-pin disagreements
          const agrees = computeAgreement(result);
          if (agrees === false) {
            autoPinned.add(result.video_id);
            setPinnedIds((prev) => new Set([...prev, result.video_id]));
          }
        }
        setProgress({ current: i + 1, total: video_ids.length });
      }

      // Step 3: auto-save evaluation results
      const batchLabeled = collected.filter(
        (r) => r.human_label && r.prediction,
      );
      let evaluationId: number | null = null;

      if (batchLabeled.length > 0) {
        const batchAgrees = batchLabeled.filter(
          (r) => computeAgreement(r) === true,
        );
        const batchClassCounts: Record<
          string,
          { correct: number; total: number }
        > = {};
        for (const r of batchLabeled) {
          const cls = r.prediction!.class;
          if (!batchClassCounts[cls])
            batchClassCounts[cls] = { correct: 0, total: 0 };
          batchClassCounts[cls].total++;
          if (computeAgreement(r) === true) batchClassCounts[cls].correct++;
        }

        const accuracy = batchAgrees.length / batchLabeled.length;
        const elapsedMin = (performance.now() - batchStartTime) / 60000;
        const imagesPerMinute =
          elapsedMin > 0 ? Math.round(collected.length / elapsedMin) : null;

        const saveRes = await fetch("/api/models/ai-screening/save-eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            accuracy,
            sample_size: batchLabeled.length,
            per_class: {
              ...batchClassCounts,
              images_per_minute: imagesPerMinute,
            },
            model_version: detectedModelVersion,
          }),
        });
        if (saveRes.ok) {
          const evalResult = await saveRes.json();
          evaluationId = evalResult.id;
          await fetchHistory();
        }

        // Step 4: save screening session
        try {
          const pinStateObj: Record<string, boolean> = {};
          autoPinned.forEach((id) => (pinStateObj[id] = true));

          const { session_id } = await createScreeningSession({
            results: collected,
            accuracy,
            sample_size: batchLabeled.length,
            per_class: batchClassCounts,
            model_version: detectedModelVersion,
            pin_state: pinStateObj,
            evaluation_id: evaluationId,
          });
          setSessionId(session_id);
          router.replace(`/evaluate/screening/${session_id}`, {
            scroll: false,
          });
        } catch (e) {
          console.warn("Failed to save session:", e);
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      const msg = e instanceof Error ? e.message : "Unknown error";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  // Compute accuracy summary from results
  const labeled = results.filter((r) => r.human_label && r.prediction);
  const agrees = labeled.filter((r) => computeAgreement(r) === true);

  const classCounts: Record<string, { correct: number; total: number }> = {};
  for (const r of labeled) {
    const cls = r.prediction!.class;
    if (!classCounts[cls]) classCounts[cls] = { correct: 0, total: 0 };
    classCounts[cls].total++;
    if (computeAgreement(r) === true) classCounts[cls].correct++;
  }

  // Sort results into pinned and unpinned
  const { pinned, unpinned } = useMemo(() => {
    const p: InspectResult[] = [];
    const u: InspectResult[] = [];
    for (const r of results) {
      if (pinnedIds.has(r.video_id)) p.push(r);
      else u.push(r);
    }
    return { pinned: p, unpinned: u };
  }, [results, pinnedIds]);

  // Prepare chart data (chronological)
  const chartData = [...evalHistory].reverse().map((ev) => ({
    date: new Date(ev.evaluated_at).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
    accuracy: Math.round(ev.accuracy * 1000) / 10,
    images_per_minute:
      ((ev.per_class as Record<string, unknown>)?.images_per_minute as
        | number
        | null) ?? null,
    notes: ev.notes,
    sample_size: ev.sample_size,
  }));

  const hasPerformanceData = chartData.some((d) => d.images_per_minute != null);

  // Version annotations: evals where notes starts with "v"
  const versionLines = chartData
    .map((d, i) => ({ ...d, idx: i }))
    .filter((d) => d.notes && /^v\d/i.test(d.notes));

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm text-muted-foreground">
          Sample from labeled videos:
        </span>
        <input
          type="number"
          value={sampleSize}
          onChange={(e) => setSampleSize(Number(e.target.value))}
          min={1}
          max={50}
          className="w-16 px-2 py-1.5 border rounded text-sm"
        />
        <button
          onClick={runBatch}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Inspecting..." : "Sample & Inspect"}
        </button>
        {modelVersion && (
          <span className="text-xs text-muted-foreground font-mono">
            model: {modelVersion}
          </span>
        )}

        <div className="flex-1" />

        {/* Session info + recent sessions */}
        <div className="relative" ref={sessionListRef}>
          <button
            onClick={fetchRecentSessions}
            className="px-3 py-1.5 border rounded text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {sessionId ? (
              <span className="font-mono">{sessionId}</span>
            ) : (
              "Sessions"
            )}
          </button>
          {showSessionList && recentSessions.length > 0 && (
            <div className="absolute right-0 top-full mt-1 z-50 w-72 bg-background border rounded-lg shadow-lg overflow-hidden">
              <div className="max-h-64 overflow-y-auto">
                {recentSessions.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setShowSessionList(false);
                      router.push(`/evaluate/screening/${s.id}`);
                    }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-muted/50 transition-colors border-b last:border-b-0 ${
                      s.id === sessionId ? "bg-muted/30" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-muted-foreground">
                        {s.id}
                      </span>
                      {s.accuracy != null && (
                        <span className="font-medium">
                          {(s.accuracy * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="text-muted-foreground mt-0.5">
                      {new Date(s.created_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                      {" \u00b7 "}n={s.sample_size}
                      {s.model_version ? ` \u00b7 ${s.model_version}` : ""}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {sessionId && (
          <button
            onClick={() => {
              navigator.clipboard.writeText(window.location.href);
            }}
            className="px-3 py-1.5 border rounded text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Copy session URL"
          >
            Copy URL
          </button>
        )}
      </div>

      {/* Progress bar */}
      {loading && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Inspecting videos...</span>
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
              style={{
                width: `${(progress.current / progress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Charts: accuracy + performance side-by-side */}
      {chartData.length >= 2 && (
        <div
          className={`grid gap-4 ${hasPerformanceData ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}
        >
          <div className="border rounded-lg p-3">
            <h3 className="text-sm font-medium mb-2">Accuracy over time</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                  width={40}
                />
                <Tooltip
                  formatter={(value) => [`${value}%`, "Accuracy"]}
                  labelFormatter={(label, payload) => {
                    const d = payload?.[0]?.payload;
                    return `${label}${d?.notes ? ` \u2014 ${d.notes}` : ""} (n=${d?.sample_size ?? "?"})`;
                  }}
                />
                {versionLines.map((v) => (
                  <ReferenceLine
                    key={v.idx}
                    x={v.date}
                    stroke="currentColor"
                    strokeDasharray="4 4"
                    strokeOpacity={0.4}
                    label={{
                      value: v.notes!.split(":")[0].trim(),
                      position: "top",
                      fontSize: 10,
                      fontWeight: "bold",
                    }}
                  />
                ))}
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="currentColor"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {hasPerformanceData && (
            <div className="border rounded-lg p-3">
              <h3 className="text-sm font-medium mb-2">
                Performance over time
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart
                  data={chartData.filter((d) => d.images_per_minute != null)}
                >
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    tickFormatter={(v) => `${v}`}
                    width={40}
                  />
                  <Tooltip
                    formatter={(value) => [`${value} img/min`, "Throughput"]}
                    labelFormatter={(label, payload) => {
                      const d = payload?.[0]?.payload;
                      return `${label} (n=${d?.sample_size ?? "?"})`;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="images_per_minute"
                    stroke="currentColor"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Accuracy summary */}
      {labeled.length > 0 && (
        <div className="border rounded-lg p-3 space-y-2 bg-muted/20">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Accuracy: {agrees.length}/{labeled.length} (
              {((agrees.length / labeled.length) * 100).toFixed(1)}%)
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{
                  width: `${(agrees.length / labeled.length) * 100}%`,
                }}
              />
            </div>
          </div>
          <div className="flex gap-3 text-xs text-muted-foreground">
            {Object.entries(classCounts).map(([cls, { correct, total }]) => (
              <span key={cls}>
                <span className={classColor(cls)}>{cls}</span>: {correct}/
                {total}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          {/* Pinned section — full-size cards */}
          {pinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                Pinned ({pinned.length})
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {pinned.map((r) => (
                  <VideoCard
                    key={r.video_id}
                    result={r}
                    pinned
                    onPin={() => togglePin(r.video_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Thumbnail strip — small squares for unpinned */}
          {unpinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                {pinned.length > 0
                  ? `Others (${unpinned.length})`
                  : `${results.length} results`}
              </p>

              {/* Compact thumbnail grid */}
              <div className="flex flex-wrap gap-1.5">
                {unpinned
                  .filter((r) => !expandedIds.has(r.video_id))
                  .map((r) => {
                    const agreement = computeAgreement(r);
                    return (
                      <button
                        key={r.video_id}
                        onClick={() =>
                          agreement === false
                            ? togglePin(r.video_id)
                            : toggleExpand(r.video_id)
                        }
                        title={`${r.title}\n${r.prediction?.class ?? "?"} ${
                          agreement === true
                            ? "\u2713"
                            : agreement === false
                              ? "\u2717"
                              : "?"
                        }`}
                        className="relative w-16 h-12 rounded border overflow-hidden group flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30"
                      >
                        <img
                          src={youtubeThumb(r.video_id, 1)}
                          alt={r.title}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        <div
                          className={`absolute inset-0 ${
                            agreement === true
                              ? "bg-green-500/25"
                              : agreement === false
                                ? "bg-red-500/35"
                                : "bg-yellow-500/20"
                          }`}
                        />
                        <span className="absolute bottom-0 right-0 text-[9px] leading-none bg-black/60 text-white px-0.5 py-px">
                          {r.prediction?.class?.[0]?.toUpperCase() ?? "?"}
                        </span>
                        {agreement !== null && (
                          <span
                            className={`absolute top-0 left-0 text-[9px] leading-none px-0.5 py-px ${
                              agreement
                                ? "bg-green-500/80 text-white"
                                : "bg-red-500/80 text-white"
                            }`}
                          >
                            {agreement ? "\u2713" : "\u2717"}
                          </span>
                        )}
                      </button>
                    );
                  })}
              </div>

              {/* Expanded cards from thumbnail clicks */}
              {unpinned.filter((r) => expandedIds.has(r.video_id)).length >
                0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                  {unpinned
                    .filter((r) => expandedIds.has(r.video_id))
                    .map((r) => (
                      <div key={r.video_id} className="relative">
                        <button
                          onClick={() => toggleExpand(r.video_id)}
                          className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded bg-muted/80 text-muted-foreground hover:text-foreground text-xs"
                          title="Collapse"
                        >
                          {"\u2715"}
                        </button>
                        <VideoCard
                          result={r}
                          pinned={false}
                          onPin={() => togglePin(r.video_id)}
                        />
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
