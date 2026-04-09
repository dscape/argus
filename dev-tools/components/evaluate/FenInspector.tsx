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
import OverlayTestCard from "@/components/evaluate/OverlayTestCard";
import {
  sampleOverlayBoards,
  inspectOverlayBoard,
  createOverlayTestSession,
  listOverlayTestSessions,
  updateOverlaySessionPins,
  overlayBoardImageUrl,
  type OverlayTestResult,
  type OverlayTestSession,
} from "@/lib/api";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: { piece_accuracy?: number; images_per_minute?: number } | null;
}

interface FenInspectorProps {
  initialSession?: {
    id: string;
    results: OverlayTestResult[];
    accuracy: number | null;
    piece_accuracy: number | null;
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

export default function FenInspector({
  initialSession,
}: FenInspectorProps) {
  const router = useRouter();
  const [sampleSize, setSampleSize] = useState(20);
  const [results, setResults] = useState<OverlayTestResult[]>(
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
  const [recentSessions, setRecentSessions] = useState<OverlayTestSession[]>(
    [],
  );
  const [showSessionList, setShowSessionList] = useState(false);
  const [emptyMessage, setEmptyMessage] = useState<string | null>(null);
  const inspectedFiles = useRef<Set<string>>(new Set());
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
        "/api/models/evaluations?model_name=piece_classifier",
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
      const { sessions } = await listOverlayTestSessions(20);
      setRecentSessions(sessions);
      setShowSessionList(true);
    } catch (e) {
      console.warn("Failed to fetch sessions:", e);
    }
  }

  const togglePin = useCallback(
    (filename: string) => {
      setPinnedIds((prev) => {
        const next = new Set(prev);
        if (next.has(filename)) next.delete(filename);
        else next.add(filename);

        if (sessionId) {
          updateOverlaySessionPins(sessionId, {
            [filename]: !prev.has(filename),
          }).catch(() => {});
        }
        return next;
      });
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(filename);
        return next;
      });
    },
    [sessionId],
  );

  const toggleExpand = useCallback((filename: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
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

    const collected: OverlayTestResult[] = [];
    const autoPinned = new Set<string>();
    const batchStartTime = performance.now();

    try {
      // Step 1: get sample filenames
      const excludeArr = Array.from(inspectedFiles.current);
      const { filenames } = await sampleOverlayBoards(sampleSize, excludeArr);

      if (filenames.length === 0) {
        setEmptyMessage(
          "No board samples are available. Populate the local overlay board dataset or keep the committed fixture boards checked out.",
        );
        return;
      }

      setProgress({ current: 0, total: filenames.length });

      // Step 2: inspect each board
      for (let i = 0; i < filenames.length; i++) {
        if (controller.signal.aborted) break;

        const result = await inspectOverlayBoard(filenames[i]);
        collected.push(result);
        setResults((prev) => [...prev, result]);
        inspectedFiles.current.add(filenames[i]);

        // Auto-pin mismatches
        if (result.match === false) {
          autoPinned.add(result.filename);
          setPinnedIds((prev) => new Set([...prev, result.filename]));
        }

        setProgress({ current: i + 1, total: filenames.length });
      }

      // Step 3: compute accuracy & save
      const total = collected.length;
      if (total > 0) {
        const matches = collected.filter((r) => r.match).length;
        const accuracy = matches / total;
        const avgPieceAccuracy =
          collected.reduce((s, r) => s + (r.piece_accuracy ?? 0), 0) / total;
        const elapsedMin = (performance.now() - batchStartTime) / 60000;
        const imagesPerMinute =
          elapsedMin > 0 ? Math.round(total / elapsedMin) : null;

        const saveRes = await fetch("/api/models/overlay-test/save-eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            accuracy,
            sample_size: total,
            piece_accuracy: avgPieceAccuracy,
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

          const { session_id } = await createOverlayTestSession({
            results: collected,
            accuracy,
            sample_size: total,
            piece_accuracy: avgPieceAccuracy,
            pin_state: pinStateObj,
            evaluation_id: evaluationId,
          });
          setSessionId(session_id);
          router.replace(`/evaluate/fen/${session_id}`, { scroll: false });
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

  // Accuracy summary
  const total = results.length;
  const matches = results.filter((r) => r.match).length;
  const avgPieceAccuracy =
    total > 0
      ? results.reduce((s, r) => s + (r.piece_accuracy ?? 0), 0) / total
      : 0;

  // Sort into pinned / unpinned
  const { pinned, unpinned } = useMemo(() => {
    const p: OverlayTestResult[] = [];
    const u: OverlayTestResult[] = [];
    for (const r of results) {
      if (pinnedIds.has(r.filename)) p.push(r);
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
    piece_accuracy: ev.per_class?.piece_accuracy
      ? Math.round(ev.per_class.piece_accuracy * 1000) / 10
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
          Samples from board fixtures:
          <InfoIcon tip="Samples committed board-crop fixtures so this view isolates piece-classifier accuracy from full-frame overlay extraction. Runtime real-frame coverage lives on the Overlay evaluator." />
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
                      router.push(`/evaluate/fen/${s.id}`);
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
                      {s.piece_accuracy != null
                        ? ` \u00b7 ${(s.piece_accuracy * 100).toFixed(1)}% sq`
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
            <span>Inspecting boards...</span>
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
              Board accuracy over time
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
                    name === "accuracy" ? "Board accuracy" : "Square accuracy",
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
              Board accuracy: {matches}/{total} (
              {((matches / total) * 100).toFixed(1)}%)
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{ width: `${(matches / total) * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            Square accuracy: {(avgPieceAccuracy * 100).toFixed(2)}% (
            {Math.round(avgPieceAccuracy * 64)}/64 avg)
          </div>
        </div>
      )}

      {/* Per-step timing summary */}
      {total > 0 && results.some((r) => r.overlay_detect_ms != null) && (
        <div className="border rounded-lg p-3 space-y-2 bg-background">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Per-step average timing
          </p>
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center p-2 rounded bg-muted/30">
              <div className="text-lg font-bold">
                {(
                  results.reduce((s, r) => s + (r.overlay_detect_ms ?? 0), 0) /
                  total
                ).toFixed(1)}
                ms
              </div>
              <div className="text-xs text-muted-foreground">
                Overlay Detect
              </div>
            </div>
            <div className="text-center p-2 rounded bg-muted/30">
              <div className="text-lg font-bold">
                {(
                  results.reduce((s, r) => s + (r.grid_detect_ms ?? 0), 0) /
                  total
                ).toFixed(1)}
                ms
              </div>
              <div className="text-xs text-muted-foreground">Grid Detect</div>
            </div>
            <div className="text-center p-2 rounded bg-muted/30">
              <div className="text-lg font-bold">
                {(
                  results.reduce((s, r) => s + (r.piece_classify_ms ?? 0), 0) /
                  total
                ).toFixed(1)}
                ms
              </div>
              <div className="text-xs text-muted-foreground">
                Piece Classify
              </div>
            </div>
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
                {pinned.map((r) => (
                  <OverlayTestCard
                    key={r.filename}
                    result={r}
                    pinned
                    onPin={() => togglePin(r.filename)}
                  />
                ))}
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
                  .filter((r) => !expandedIds.has(r.filename))
                  .map((r) => (
                    <button
                      key={r.filename}
                      onClick={() =>
                        r.match === false
                          ? togglePin(r.filename)
                          : toggleExpand(r.filename)
                      }
                      title={`${r.filename}\n${r.match ? "\u2713 match" : `\u2717 ${r.square_diffs.length} wrong`}`}
                      className="relative w-16 h-16 rounded border overflow-hidden flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30"
                    >
                      <img
                        src={
                          r.board_image_b64
                            ? `data:image/jpeg;base64,${r.board_image_b64}`
                            : overlayBoardImageUrl(r.filename)
                        }
                        alt={r.filename}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                      <div
                        className={`absolute inset-0 ${
                          r.match ? "bg-green-500/25" : "bg-red-500/35"
                        }`}
                      />
                      <span
                        className={`absolute top-0 left-0 text-[9px] leading-none px-0.5 py-px ${
                          r.match
                            ? "bg-green-500/80 text-white"
                            : "bg-red-500/80 text-white"
                        }`}
                      >
                        {r.match ? "\u2713" : "\u2717"}
                      </span>
                      {r.match === false && (
                        <span className="absolute bottom-0 right-0 text-[9px] leading-none bg-black/60 text-white px-0.5 py-px">
                          {r.square_diffs.length}
                        </span>
                      )}
                    </button>
                  ))}
              </div>

              {/* Expanded cards */}
              {unpinned.filter((r) => expandedIds.has(r.filename)).length >
                0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                  {unpinned
                    .filter((r) => expandedIds.has(r.filename))
                    .map((r) => (
                      <div key={r.filename} className="relative">
                        <button
                          onClick={() => toggleExpand(r.filename)}
                          className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded bg-muted/80 text-muted-foreground hover:text-foreground text-xs"
                          title="Collapse"
                        >
                          {"\u2715"}
                        </button>
                        <OverlayTestCard
                          result={r}
                          pinned={false}
                          onPin={() => togglePin(r.filename)}
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
