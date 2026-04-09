"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
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
import OverlayEvalCard from "@/components/evaluate/OverlayEvalCard";
import {
  createOverlayEvalSession,
  listOverlayEvalSessions,
  updateOverlayEvalPins,
  type OverlayEvalResult,
  type OverlayEvalSession,
} from "@/lib/api";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: { fen_success_rate?: number; images_per_minute?: number } | null;
}

interface OverlayEvalInspectorProps {
  initialSession?: {
    id: string;
    results: OverlayEvalResult[];
    detection_rate: number | null;
    fen_success_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  };
}

/** Small inline info icon with a CSS tooltip. */
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

function resultKey(r: OverlayEvalResult): string {
  return r.frame_key;
}

export default function OverlayEvalInspector({
  initialSession,
}: OverlayEvalInspectorProps) {
  const router = useRouter();
  const [sampleSize, setSampleSize] = useState(20);
  const [results, setResults] = useState<OverlayEvalResult[]>(
    initialSession?.results ?? [],
  );
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [evalHistory, setEvalHistory] = useState<EvalPoint[]>([]);
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
  const [recentSessions, setRecentSessions] = useState<OverlayEvalSession[]>(
    [],
  );
  const [showSessionList, setShowSessionList] = useState(false);
  const [emptyMessage, setEmptyMessage] = useState<string | null>(null);

  // Save/reject/edit state
  const [editedFens, setEditedFens] = useState<Record<string, string>>({});
  const [rejected, setRejected] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState<Set<string>>(new Set());
  const [savingId, setSavingId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const sessionListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHistory();
  }, []);

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
        "/api/models/evaluations?model_name=overlay_detection",
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
      const { sessions } = await listOverlayEvalSessions(20);
      setRecentSessions(sessions);
      setShowSessionList(true);
    } catch (e) {
      console.warn("Failed to fetch sessions:", e);
    }
  }

  const togglePin = useCallback(
    (key: string) => {
      setPinnedIds((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key);
        else next.add(key);

        if (sessionId) {
          updateOverlayEvalPins(sessionId, {
            [key]: !prev.has(key),
          }).catch(() => {});
        }
        return next;
      });
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    },
    [sessionId],
  );

  const toggleExpand = useCallback((key: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  async function saveOne(r: OverlayEvalResult) {
    const key = resultKey(r);
    const fen = editedFens[key];
    if (!fen) return;

    setSavingId(key);
    try {
      const res = await fetch("/api/models/overlay-test/extract-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmations: [
            {
              video_id: r.video_id,
              frame_name: r.frame_name,
              fen,
              image_b64: r.image_b64,
            },
          ],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.errors?.length > 0) {
        alert(data.errors.join("\n"));
      } else {
        setSaved((prev) => new Set(prev).add(key));
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSavingId(null);
    }
  }

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
    setEditedFens({});
    setRejected(new Set());
    setSaved(new Set());
    setEmptyMessage(null);

    const collected: OverlayEvalResult[] = [];
    const autoPinned = new Set<string>();
    const fenPromises: Promise<void>[] = [];
    const batchStartTime = performance.now();

    try {
      // Step 1: get candidate video IDs
      const candRes = await fetch(
        `/api/models/overlay-test/extract-candidates?limit=${sampleSize}`,
        { signal: controller.signal },
      );
      if (!candRes.ok) throw new Error(await candRes.text());
      const { video_ids } = await candRes.json();

      if (video_ids.length === 0) {
        setEmptyMessage(
          "No overlay samples are available. Approve/download overlay videos or keep the committed fixture frames checked out.",
        );
        return;
      }

      setProgress({ current: 0, total: video_ids.length });

      // Step 2: process each video — fast detect, then async FEN
      for (let i = 0; i < video_ids.length; i++) {
        if (controller.signal.aborted) break;

        const detectStart = performance.now();
        const res = await fetch("/api/models/overlay-test/extract-detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: video_ids[i] }),
          signal: controller.signal,
        });

        if (res.ok) {
          const raw = await res.json();
          const detectMs = Math.round(performance.now() - detectStart);
          const result: OverlayEvalResult = {
            frame_key: raw.frame_key ?? raw.video_id,
            video_id: raw.video_id,
            frame_name: raw.frame_name,
            status: raw.status,
            warning: raw.warning,
            image_b64: raw.image_b64,
            predicted_fen: raw.predicted_fen,
            elapsed_ms: detectMs,
            overlay_detect_ms: raw.overlay_detect_ms,
            grid_detect_ms: raw.grid_detect_ms,
          };

          if (result.status === "detected" && result.frame_name) {
            result.fen_loading = true;
            collected.push(result);
            setResults((prev) => [...prev, result]);

            // Slow phase: classify FEN — track promise so we can await before saving
            const frameKey = resultKey(result);
            const frameName = result.frame_name;
            const videoId = result.video_id;
            const fenPromise = fetch("/api/models/overlay-test/extract-fen", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                video_id: videoId,
                frame_name: frameName,
                image_b64: result.image_b64,
              }),
            })
              .then((fenRes) => fenRes.json())
              .then((fenData) => {
                if (fenData.status === "error" || !fenData.predicted_fen) {
                  autoPinned.add(frameKey);
                  setPinnedIds((prev) => new Set([...prev, frameKey]));
                  setResults((prev) =>
                    prev.map((r) =>
                      resultKey(r) === frameKey
                        ? {
                            ...r,
                            status: "no_overlay" as const,
                            warning: fenData.warning,
                            fen_loading: false,
                            piece_classify_ms: fenData.piece_classify_ms,
                          }
                        : r,
                    ),
                  );
                  const idx = collected.findIndex((r) => resultKey(r) === frameKey);
                  if (idx >= 0) {
                    collected[idx] = {
                      ...collected[idx],
                      status: "no_overlay",
                      warning: fenData.warning,
                      fen_loading: false,
                      piece_classify_ms: fenData.piece_classify_ms,
                    };
                  }
                  return;
                }
                setResults((prev) =>
                  prev.map((r) =>
                    resultKey(r) === frameKey
                      ? {
                          ...r,
                          predicted_fen: fenData.predicted_fen,
                          status: fenData.status,
                          warning: fenData.warning,
                          fen_loading: false,
                          piece_classify_ms: fenData.piece_classify_ms,
                        }
                      : r,
                  ),
                );
                const idx = collected.findIndex((r) => resultKey(r) === frameKey);
                if (idx >= 0) {
                  collected[idx] = {
                    ...collected[idx],
                    predicted_fen: fenData.predicted_fen,
                    status: fenData.status,
                    warning: fenData.warning,
                    fen_loading: false,
                    piece_classify_ms: fenData.piece_classify_ms,
                  };
                }
                if (fenData.predicted_fen) {
                  setEditedFens((prev) => ({
                    ...prev,
                    [frameKey]: fenData.predicted_fen,
                  }));
                }
                if (fenData.status === "warning") {
                  autoPinned.add(frameKey);
                  setPinnedIds((prev) => new Set([...prev, frameKey]));
                }
              })
              .catch(() => {
                autoPinned.add(frameKey);
                setPinnedIds((prev) => new Set([...prev, frameKey]));
                setResults((prev) =>
                  prev.map((r) =>
                    resultKey(r) === frameKey
                      ? {
                          ...r,
                          status: "no_overlay" as const,
                          warning: "FEN classification failed",
                          fen_loading: false,
                        }
                      : r,
                  ),
                );
                const idx = collected.findIndex((r) => resultKey(r) === frameKey);
                if (idx >= 0) {
                  collected[idx] = {
                    ...collected[idx],
                    status: "no_overlay",
                    warning: "FEN classification failed",
                    fen_loading: false,
                  };
                }
              });
            fenPromises.push(fenPromise);
          } else {
            // no_overlay
            collected.push(result);
            setResults((prev) => [...prev, result]);
            const key = resultKey(result);
            autoPinned.add(key);
            setPinnedIds((prev) => new Set([...prev, key]));
          }
        }

        setProgress({ current: i + 1, total: video_ids.length });
      }

      // Step 3: await all FEN classification calls before computing metrics
      await Promise.allSettled(fenPromises);

      const total = collected.length;
      if (total > 0) {
        const detected = collected.filter(
          (r) => r.status !== "no_overlay",
        ).length;
        const fenSuccess = collected.filter(
          (r) => r.status === "ok",
        ).length;
        const detectionRate = detected / total;
        const fenSuccessRate = total > 0 ? fenSuccess / total : 0;
        const elapsedMin = (performance.now() - batchStartTime) / 60000;
        const imagesPerMinute =
          elapsedMin > 0 ? Math.round(total / elapsedMin) : null;

        const saveRes = await fetch("/api/models/overlay-eval/save-eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            detection_rate: detectionRate,
            fen_success_rate: fenSuccessRate,
            sample_size: total,
            images_per_minute: imagesPerMinute,
          }),
        });
        let evaluationId: number | null = null;
        if (saveRes.ok) {
          const evalResult = await saveRes.json();
          evaluationId = evalResult.id;
          await fetchHistory();
        }

        // Step 4: save session
        try {
          const pinStateObj: Record<string, boolean> = {};
          autoPinned.forEach((id) => (pinStateObj[id] = true));

          const { session_id } = await createOverlayEvalSession({
            results: collected,
            detection_rate: detectionRate,
            fen_success_rate: fenSuccessRate,
            sample_size: total,
            pin_state: pinStateObj,
            evaluation_id: evaluationId,
          });
          setSessionId(session_id);
          router.replace(`/evaluate/overlay/${session_id}`, { scroll: false });
        } catch (e) {
          console.warn("Failed to save session:", e);
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      const msg = e instanceof Error ? e.message : "Unknown error";
      alert(msg);
    } finally {
      setLoading(false);
    }
  }

  // Compute summary metrics
  const total = results.length;
  const detected = results.filter(
    (r) => r.status !== "no_overlay",
  ).length;
  const fenSuccess = results.filter((r) => r.status === "ok").length;

  // Sort into pinned / unpinned
  const { pinned, unpinned } = useMemo(() => {
    const p: OverlayEvalResult[] = [];
    const u: OverlayEvalResult[] = [];
    for (const r of results) {
      if (pinnedIds.has(resultKey(r))) p.push(r);
      else u.push(r);
    }
    return { pinned: p, unpinned: u };
  }, [results, pinnedIds]);

  // Chart data
  const chartData = [...evalHistory].reverse().map((ev) => ({
    date: new Date(ev.evaluated_at).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
    accuracy: Math.round(ev.accuracy * 1000) / 10,
    fen_success_rate: ev.per_class?.fen_success_rate
      ? Math.round(ev.per_class.fen_success_rate * 1000) / 10
      : null,
    images_per_minute: ev.per_class?.images_per_minute ?? null,
    notes: ev.notes,
    sample_size: ev.sample_size,
  }));

  const hasPerformanceData = chartData.some((d) => d.images_per_minute != null);

  const versionLines = chartData
    .map((d, i) => ({ ...d, idx: i }))
    .filter((d) => d.notes && /^v\d/i.test(d.notes));

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm text-muted-foreground">
          Sample from overlay videos:
          <InfoIcon tip="Samples approved overlay videos when local screening data exists, then falls back to committed fixture frames so the inspector still works in a fresh clone." />
        </span>
        <input
          type="number"
          value={sampleSize}
          onChange={(e) => setSampleSize(Number(e.target.value))}
          min={1}
          max={200}
          className="w-16 px-2 py-1.5 border rounded text-sm"
        />
        <button
          onClick={runBatch}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Inspecting..." : "Sample & Inspect"}
        </button>

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
                      router.push(`/evaluate/overlay/${s.id}`);
                    }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-muted/50 transition-colors border-b last:border-b-0 ${
                      s.id === sessionId ? "bg-muted/30" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-muted-foreground">
                        {s.id}
                      </span>
                      {s.detection_rate != null && (
                        <span className="font-medium">
                          {(s.detection_rate * 100).toFixed(1)}%
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
                      {s.fen_success_rate != null
                        ? ` \u00b7 ${(s.fen_success_rate * 100).toFixed(1)}% FEN`
                        : ""}
                    </div>
                  </button>
                ))}
              </div>
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

      {/* Progress bar */}
      {loading && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Inspecting overlays...</span>
            <span>
              {progress.current}/{progress.total}
            </span>
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
            <h3 className="text-sm font-medium mb-2">
              Detection rate over time
            </h3>
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
                  formatter={(value, name) => [
                    `${value}%`,
                    name === "accuracy" ? "Detection rate" : "FEN success rate",
                  ]}
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
      {total > 0 && (
        <div className="border rounded-lg p-3 space-y-2 bg-background">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Detection rate: {detected}/{total} (
              {((detected / total) * 100).toFixed(1)}%)
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{ width: `${(detected / total) * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            FEN success: {fenSuccess}/{total} (
            {((fenSuccess / total) * 100).toFixed(1)}%)
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          {/* Pinned section */}
          {pinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                Pinned ({pinned.length})
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {pinned.map((r) => {
                  const key = resultKey(r);
                  return (
                    <OverlayEvalCard
                      key={key}
                      result={r}
                      pinned
                      onPin={() => togglePin(key)}
                      editedFen={editedFens[key]}
                      onEditFen={(fen) =>
                        setEditedFens((prev) => ({ ...prev, [key]: fen }))
                      }
                      onSave={() => saveOne(r)}
                      onReject={() =>
                        setRejected((prev) => {
                          const next = new Set(prev);
                          if (next.has(key)) next.delete(key);
                          else next.add(key);
                          return next;
                        })
                      }
                      isSaved={saved.has(key)}
                      isRejected={rejected.has(key)}
                      isSaving={savingId === key}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Thumbnail strip */}
          {unpinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                {pinned.length > 0
                  ? `Others (${unpinned.length})`
                  : `${results.length} results`}
              </p>

              <div className="flex flex-wrap gap-1.5">
                {unpinned
                  .filter((r) => !expandedIds.has(resultKey(r)))
                  .map((r) => {
                    const key = resultKey(r);
                    const isOk = r.status === "ok";
                    const isWarning = r.status === "warning";
                    const isNoOverlay = r.status === "no_overlay";
                    const isFenLoading = r.fen_loading;

                    return (
                      <button
                        key={key}
                        onClick={() =>
                          isNoOverlay || isWarning
                            ? togglePin(key)
                            : toggleExpand(key)
                        }
                        title={`${r.frame_name ?? r.video_id}\n${isOk ? "\u2713 ok" : isWarning ? "\u26A0 warning" : isNoOverlay ? "\u2717 no overlay" : "\u2026 loading"}`}
                        className="relative w-16 h-16 rounded border overflow-hidden flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30"
                      >
                        {r.image_b64 ? (
                          <img
                            src={`data:image/jpeg;base64,${r.image_b64}`}
                            alt={key}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full bg-muted flex items-center justify-center text-[9px] text-muted-foreground">
                            {isNoOverlay ? "\u2717" : "?"}
                          </div>
                        )}
                        <div
                          className={`absolute inset-0 ${
                            isOk
                              ? "bg-green-500/25"
                              : isWarning
                                ? "bg-yellow-500/25"
                                : isNoOverlay
                                  ? "bg-red-500/35"
                                  : isFenLoading
                                    ? "bg-blue-500/15"
                                    : "bg-muted/25"
                          }`}
                        />
                        <span
                          className={`absolute top-0 left-0 text-[9px] leading-none px-0.5 py-px ${
                            isOk
                              ? "bg-green-500/80 text-white"
                              : isWarning
                                ? "bg-yellow-500/80 text-white"
                                : isNoOverlay
                                  ? "bg-red-500/80 text-white"
                                  : "bg-blue-500/80 text-white"
                          }`}
                        >
                          {isOk ? "\u2713" : isWarning ? "\u26A0" : isNoOverlay ? "\u2717" : "\u2026"}
                        </span>
                      </button>
                    );
                  })}
              </div>

              {/* Expanded cards */}
              {unpinned.filter((r) => expandedIds.has(resultKey(r))).length >
                0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                  {unpinned
                    .filter((r) => expandedIds.has(resultKey(r)))
                    .map((r) => {
                      const key = resultKey(r);
                      return (
                        <div key={key} className="relative">
                          <button
                            onClick={() => toggleExpand(key)}
                            className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded bg-muted/80 text-muted-foreground hover:text-foreground text-xs"
                            title="Collapse"
                          >
                            {"\u2715"}
                          </button>
                          <OverlayEvalCard
                            result={r}
                            pinned={false}
                            onPin={() => togglePin(key)}
                            editedFen={editedFens[key]}
                            onEditFen={(fen) =>
                              setEditedFens((prev) => ({ ...prev, [key]: fen }))
                            }
                            onSave={() => saveOne(r)}
                            onReject={() =>
                              setRejected((prev) => {
                                const next = new Set(prev);
                                if (next.has(key)) next.delete(key);
                                else next.add(key);
                                return next;
                              })
                            }
                            isSaved={saved.has(key)}
                            isRejected={rejected.has(key)}
                            isSaving={savingId === key}
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
