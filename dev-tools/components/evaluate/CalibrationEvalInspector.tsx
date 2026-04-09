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
import {
  sampleCalibrationClips,
  inspectCalibration,
  saveCalibrationEval,
  createCalibrationEvalSession,
  listCalibrationEvalSessions,
  updateCalibrationEvalPins,
  getModelVersions,
  calibrationSessionImageUrl,
  type CalibrationEvalSessionSummary,
} from "@/lib/api";

function imgSrc(val: string, sessionId: string | null): string {
  if (val.length > 200) return `data:image/jpeg;base64,${val}`;
  if (sessionId) return calibrationSessionImageUrl(sessionId, val);
  return `data:image/jpeg;base64,${val}`;
}

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: Record<string, unknown> | null;
}

interface CalibrationEvalInspectorProps {
  initialSession?: {
    id: string;
    results: any[];
    overlay_iou_avg: number | null;
    theme_accuracy: number | null;
    orientation_accuracy: number | null;
    grid_success_rate: number | null;
    fen_validity_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  };
}

function metricColor(value: number): string {
  if (value >= 0.8) return "bg-green-500/80 text-white";
  if (value >= 0.5) return "bg-yellow-500/80 text-white";
  return "bg-red-500/80 text-white";
}

function metricBarColor(value: number): string {
  if (value >= 0.8) return "bg-green-500";
  if (value >= 0.5) return "bg-yellow-500";
  return "bg-red-500";
}

export default function CalibrationEvalInspector({
  initialSession,
}: CalibrationEvalInspectorProps) {
  const router = useRouter();
  const [sampleSize, setSampleSize] = useState(5);
  const [results, setResults] = useState<any[]>(initialSession?.results ?? []);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [evalHistory, setEvalHistory] = useState<EvalPoint[]>([]);
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(
    new Set(
      initialSession?.pin_state
        ? Object.entries(initialSession.pin_state)
            .filter(([, v]) => v)
            .map(([k]) => k)
        : []
    )
  );
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [sessionId, setSessionId] = useState<string | null>(
    initialSession?.id ?? null
  );
  const [recentSessions, setRecentSessions] = useState<
    CalibrationEvalSessionSummary[]
  >([]);
  const [showSessionList, setShowSessionList] = useState(false);
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const inspectedClipIds = useRef<Set<number>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const sessionListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHistory();
    getModelVersions().then((v) => {
      const parts = [v.overlay_yolo, v.overlay, v.otb_yolo].filter(Boolean);
      setModelVersion(parts.length > 0 ? parts.join(" / ") : null);
    });
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
        "/api/models/evaluations?model_name=auto_calibration"
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
      const { sessions } = await listCalibrationEvalSessions(20);
      setRecentSessions(sessions);
      setShowSessionList(true);
    } catch (e) {
      console.warn("Failed to fetch sessions:", e);
    }
  }

  const togglePin = useCallback(
    (clipId: string) => {
      setPinnedIds((prev) => {
        const next = new Set(prev);
        if (next.has(clipId)) next.delete(clipId);
        else next.add(clipId);

        if (sessionId) {
          updateCalibrationEvalPins(sessionId, {
            [clipId]: !prev.has(clipId),
          }).catch(() => {});
        }
        return next;
      });
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(clipId);
        return next;
      });
    },
    [sessionId]
  );

  const toggleExpand = useCallback((clipId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(clipId)) next.delete(clipId);
      else next.add(clipId);
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

    const collected: any[] = [];
    const autoPinned = new Set<string>();

    try {
      // Step 1: get sample clips
      const excludeArr = Array.from(inspectedClipIds.current);
      const { clips } = await sampleCalibrationClips(sampleSize, excludeArr);

      setProgress({ current: 0, total: clips.length });

      // Step 2: inspect each clip
      for (let i = 0; i < clips.length; i++) {
        if (controller.signal.aborted) break;

        const clip = clips[i];
        const clipId = clip.clip_id ?? clip.id ?? clip;
        const result = await inspectCalibration(clipId);
        collected.push(result);
        setResults((prev) => [...prev, result]);
        inspectedClipIds.current.add(clipId);

        // Auto-pin problematic clips
        const key = String(result.clip_id ?? clipId);
        if (
          (result.overlay_iou != null && result.overlay_iou < 0.7) ||
          (result.theme_accuracy != null && result.theme_accuracy < 1.0) ||
          (result.grid_success_rate != null && result.grid_success_rate < 0.5)
        ) {
          autoPinned.add(key);
          setPinnedIds((prev) => new Set([...prev, key]));
        }

        setProgress({ current: i + 1, total: clips.length });
      }

      // Step 3: compute aggregates & save eval
      const total = collected.length;
      if (total > 0) {
        const avg = (arr: number[]) =>
          arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        const overlayIous = collected
          .filter((r) => r.overlay_iou != null)
          .map((r) => r.overlay_iou);
        const gridSuccesses = collected
          .filter((r) => r.grid_success_rate != null)
          .map((r) => r.grid_success_rate);
        const fenValidities = collected
          .filter((r) => r.fen_validity_rate != null)
          .map((r) => r.fen_validity_rate);
        const themeAccuracies = collected
          .filter((r) => r.theme_accuracy != null)
          .map((r) => r.theme_accuracy);
        const orientationAccuracies = collected
          .filter((r) => r.orientation_accuracy != null)
          .map((r) => r.orientation_accuracy);
        const cameraIous = collected
          .filter((r) => r.camera_iou != null)
          .map((r) => r.camera_iou);

        const overlayIouAvg = avg(overlayIous);
        const gridSuccessRate = avg(gridSuccesses);
        const fenValidityRate = avg(fenValidities);
        const themeAccuracy = avg(themeAccuracies);
        const orientationAccuracy = avg(orientationAccuracies);
        const cameraIouAvg = avg(cameraIous);

        const saveRes = await saveCalibrationEval({
          overlay_iou: overlayIouAvg,
          grid_success_rate: gridSuccessRate,
          fen_validity_rate: fenValidityRate,
          theme_accuracy: themeAccuracy,
          orientation_accuracy: orientationAccuracy,
          camera_iou: cameraIouAvg,
          sample_size: total,
        });

        let evaluationId: number | null = null;
        if (saveRes?.id) {
          evaluationId = saveRes.id;
          await fetchHistory();
        }

        // Step 4: save session
        try {
          const pinStateObj: Record<string, boolean> = {};
          autoPinned.forEach((id) => (pinStateObj[id] = true));

          const { session_id } = await createCalibrationEvalSession({
            results: collected,
            overlay_iou_avg: overlayIouAvg,
            theme_accuracy: themeAccuracy,
            orientation_accuracy: orientationAccuracy,
            grid_success_rate: gridSuccessRate,
            fen_validity_rate: fenValidityRate,
            sample_size: total,
            pin_state: pinStateObj,
            evaluation_id: evaluationId,
          });
          setSessionId(session_id);
          router.replace(`/evaluate/calibration/${session_id}`, {
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

  // Compute aggregate metrics from current results
  const total = results.length;
  const avgMetric = (key: string) => {
    const vals = results.filter((r) => r[key] != null).map((r) => r[key]);
    return vals.length > 0 ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
  };
  const overlayIouAvg = avgMetric("overlay_iou");
  const gridSuccessRate = avgMetric("grid_success_rate");
  const fenValidityRate = avgMetric("fen_validity_rate");
  const themeAccuracy = avgMetric("theme_accuracy");
  const orientationAccuracy = avgMetric("orientation_accuracy");
  const cameraIouAvg = avgMetric("camera_iou");

  // Sort into pinned / unpinned
  const { pinned, unpinned } = useMemo(() => {
    const p: any[] = [];
    const u: any[] = [];
    for (const r of results) {
      const key = String(r.clip_id);
      if (pinnedIds.has(key)) p.push(r);
      else u.push(r);
    }
    return { pinned: p, unpinned: u };
  }, [results, pinnedIds]);

  // Chart data
  const chartData = [...evalHistory]
    .reverse()
    .map((ev) => ({
      date: new Date(ev.evaluated_at).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      }),
      accuracy: Math.round(ev.accuracy * 1000) / 10,
      notes: ev.notes,
      sample_size: ev.sample_size,
    }));

  const versionLines = chartData
    .map((d, i) => ({ ...d, idx: i }))
    .filter((d) => d.notes && /^v\d/i.test(d.notes));

  const metrics = [
    { label: "Overlay IoU", value: overlayIouAvg, color: metricBarColor(overlayIouAvg) },
    { label: "Grid success", value: gridSuccessRate, color: metricBarColor(gridSuccessRate) },
    { label: "FEN validity", value: fenValidityRate, color: metricBarColor(fenValidityRate) },
    { label: "Theme accuracy", value: themeAccuracy, color: metricBarColor(themeAccuracy) },
    { label: "Orientation accuracy", value: orientationAccuracy, color: metricBarColor(orientationAccuracy) },
    { label: "Camera IoU", value: cameraIouAvg, color: metricBarColor(cameraIouAvg) },
  ];

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm text-muted-foreground">
          Sample calibrated clips:
        </span>
        <input
          type="number"
          value={sampleSize}
          onChange={(e) => setSampleSize(Number(e.target.value))}
          min={1}
          max={100}
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
            <div className="absolute right-0 top-full mt-1 z-50 w-80 bg-background border rounded-lg shadow-lg overflow-hidden">
              <div className="max-h-64 overflow-y-auto">
                {recentSessions.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setShowSessionList(false);
                      router.push(`/evaluate/calibration/${s.id}`);
                    }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-muted/50 transition-colors border-b last:border-b-0 ${
                      s.id === sessionId ? "bg-muted/30" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-muted-foreground">
                        {s.id}
                      </span>
                      {s.overlay_iou_avg != null && (
                        <span className="font-medium">
                          IoU {(s.overlay_iou_avg * 100).toFixed(1)}%
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
                      {s.theme_accuracy != null
                        ? ` \u00b7 theme ${(s.theme_accuracy * 100).toFixed(0)}%`
                        : ""}
                      {s.grid_success_rate != null
                        ? ` \u00b7 grid ${(s.grid_success_rate * 100).toFixed(0)}%`
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

      {/* Progress bar */}
      {loading && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Inspecting clips...</span>
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
      {chartData.length >= 1 && (
        <div className="grid gap-4 grid-cols-1 lg:grid-cols-2">
          <div className="border rounded-lg p-3">
            <h3 className="text-sm font-medium mb-2">
              Accuracy over time
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
          <div className="border rounded-lg p-3">
            <h3 className="text-sm font-medium mb-2">
              Performance over time
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
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                  width={40}
                />
                <Tooltip
                  formatter={(value) => [`${value} clips/batch`, "Throughput"]}
                  labelFormatter={(label, payload) => {
                    const d = payload?.[0]?.payload;
                    return `${label} (n=${d?.sample_size ?? "?"})`;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="sample_size"
                  name="Clips per batch"
                  stroke="currentColor"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Metrics summary */}
      {total > 0 && (
        <div className="border rounded-lg p-3 space-y-2 bg-muted/20">
          {metrics.map((m) => (
            <div key={m.label} className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground w-40 shrink-0">
                {m.label}:
              </span>
              <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
                <div
                  className={`h-full rounded ${m.color}`}
                  style={{ width: `${m.value * 100}%` }}
                />
              </div>
              <span className="text-xs font-medium w-12 text-right">
                {(m.value * 100).toFixed(1)}%
              </span>
            </div>
          ))}
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
                  <CalibrationResultCard
                    key={r.clip_id}
                    result={r}
                    pinned
                    sessionId={sessionId}
                    onPin={() => togglePin(String(r.clip_id))}
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
                  .filter((r) => !expandedIds.has(String(r.clip_id)))
                  .map((r) => {
                    const key = String(r.clip_id);
                    const iou = r.overlay_iou ?? 0;
                    const needsPin =
                      iou < 0.7 ||
                      (r.theme_accuracy != null && r.theme_accuracy < 1.0) ||
                      (r.grid_success_rate != null &&
                        r.grid_success_rate < 0.5);
                    return (
                      <button
                        key={key}
                        onClick={() =>
                          needsPin ? togglePin(key) : toggleExpand(key)
                        }
                        title={`clip ${r.clip_id}\nIoU: ${(iou * 100).toFixed(1)}%`}
                        className="relative w-16 h-16 rounded border overflow-hidden flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30"
                      >
                        {r.frames?.[0]?.frame_b64 ? (
                          <img
                            src={imgSrc(r.frames[0].frame_b64, sessionId)}
                            alt={`clip ${r.clip_id}`}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full bg-muted flex items-center justify-center text-[10px] text-muted-foreground">
                            {r.clip_id}
                          </div>
                        )}
                        <div
                          className={`absolute inset-0 ${
                            !needsPin ? "bg-green-500/25" : "bg-red-500/35"
                          }`}
                        />
                        <span
                          className={`absolute top-0 left-0 text-[9px] leading-none px-0.5 py-px ${
                            !needsPin
                              ? "bg-green-500/80 text-white"
                              : "bg-red-500/80 text-white"
                          }`}
                        >
                          {!needsPin ? "\u2713" : "\u2717"}
                        </span>
                      </button>
                    );
                  })}
              </div>

              {/* Expanded cards */}
              {unpinned.filter((r) => expandedIds.has(String(r.clip_id)))
                .length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                  {unpinned
                    .filter((r) => expandedIds.has(String(r.clip_id)))
                    .map((r) => (
                      <div key={r.clip_id} className="relative">
                        <button
                          onClick={() => toggleExpand(String(r.clip_id))}
                          className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded bg-muted/80 text-muted-foreground hover:text-foreground text-xs"
                          title="Collapse"
                        >
                          {"\u2715"}
                        </button>
                        <CalibrationResultCard
                          result={r}
                          pinned={false}
                          sessionId={sessionId}
                          onPin={() => togglePin(String(r.clip_id))}
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

// ── Result Card ──────────────────────────────────────────────

function CalibrationResultCard({
  result,
  pinned,
  sessionId,
  onPin,
}: {
  result: any;
  pinned: boolean;
  sessionId: string | null;
  onPin: () => void;
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  const r = result;
  const middleFrame =
    r.frames && r.frames.length > 0
      ? r.frames[Math.floor(r.frames.length / 2)]
      : null;

  const metricBadges = [
    { label: "Overlay IoU", value: r.overlay_iou },
    { label: "Grid", value: r.grid_success_rate },
    { label: "FEN", value: r.fen_validity_rate },
    { label: "Theme", value: r.theme_accuracy },
    { label: "Orientation", value: r.orientation_accuracy },
  ];

  return (
    <div className="border rounded-lg overflow-hidden bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b">
        <div className="flex items-center gap-2 text-xs min-w-0">
          <button
            onClick={onPin}
            title={pinned ? "Unpin from top" : "Pin to top"}
            className={`flex-shrink-0 w-5 h-5 flex items-center justify-center rounded transition-colors ${
              pinned
                ? "text-foreground"
                : "text-muted-foreground/40 hover:text-foreground"
            }`}
          >
            <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
              <path d="M9.828.722a.5.5 0 0 1 .354.146l4.95 4.95a.5.5 0 0 1-.707.707l-.71-.71-3.18 3.18a5.5 5.5 0 0 1-1.32 4.988.5.5 0 0 1-.707 0L5.464 10.94l-3.89 3.89a.5.5 0 0 1-.707-.708l3.89-3.889L1.714 7.19a.5.5 0 0 1 0-.707 5.5 5.5 0 0 1 4.988-1.32L9.88 1.985l-.71-.71a.5.5 0 0 1 .5-.853z" />
            </svg>
          </button>
          <span className="font-mono truncate">
            clip {r.clip_id}
          </span>
          {r.video_id && (
            <span className="text-muted-foreground truncate">
              {r.video_id}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
          {r.start_sec != null && r.end_sec != null && (
            <span>
              {r.start_sec.toFixed(1)}s &ndash; {r.end_sec.toFixed(1)}s
            </span>
          )}
          {r.stored_theme && (
            <span className="px-1.5 py-0.5 rounded bg-muted text-[10px]">
              {r.stored_theme}
            </span>
          )}
          {r.stored_orientation && (
            <span className="px-1.5 py-0.5 rounded bg-muted text-[10px]">
              {r.stored_orientation}
            </span>
          )}
        </div>
      </div>

      {/* Middle frame preview */}
      {middleFrame?.frame_b64 && (
        <div className="relative">
          <img
            src={imgSrc(middleFrame.frame_b64, sessionId)}
            alt={`clip ${r.clip_id} annotated frame`}
            className="w-full"
          />
          <div className="absolute bottom-1 left-1 flex gap-1 text-[9px]">
            <span className="px-1 py-0.5 rounded bg-green-500/80 text-white">
              stored overlay
            </span>
            <span className="px-1 py-0.5 rounded bg-blue-500/80 text-white">
              stored camera
            </span>
            <span className="px-1 py-0.5 rounded bg-yellow-500/80 text-white">
              detected overlay
            </span>
          </div>
        </div>
      )}

      {/* Overlay crop */}
      {middleFrame?.crop_b64 && (
        <div className="px-3 py-2 border-b">
          <p className="text-[10px] text-muted-foreground mb-1">Overlay crop</p>
          <img
            src={imgSrc(middleFrame.crop_b64, sessionId)}
            alt="overlay crop"
            className="max-h-32 rounded border"
          />
        </div>
      )}

      {/* Metric badges */}
      <div className="px-3 py-2 flex flex-wrap gap-1.5">
        {metricBadges.map(
          (m) =>
            m.value != null && (
              <span
                key={m.label}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${metricColor(m.value)}`}
              >
                {m.label}: {(m.value * 100).toFixed(1)}%
              </span>
            )
        )}
        {r.camera_iou != null && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-muted text-muted-foreground">
            Camera IoU: {(r.camera_iou * 100).toFixed(1)}%
          </span>
        )}
      </div>

      {/* FEN readout */}
      {middleFrame?.fen && (
        <div className="px-3 pb-2">
          <p className="text-[10px] text-muted-foreground mb-0.5">FEN</p>
          <code className="text-[11px] font-mono break-all bg-muted px-1.5 py-0.5 rounded block">
            {middleFrame.fen}
          </code>
        </div>
      )}

      {/* Collapsible details */}
      {r.frames && r.frames.length > 0 && (
        <div className="border-t">
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className="w-full px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
          >
            <span>{detailsOpen ? "\u25BC" : "\u25B6"}</span>
            <span>
              Per-frame details ({r.frames.length} frame
              {r.frames.length !== 1 ? "s" : ""})
            </span>
          </button>
          {detailsOpen && (
            <div className="px-3 pb-3 overflow-x-auto">
              <table className="text-[10px] w-full border-collapse">
                <thead>
                  <tr className="text-muted-foreground border-b">
                    <th className="text-left py-1 pr-2">Frame</th>
                    <th className="text-left py-1 pr-2">Overlay IoU</th>
                    <th className="text-left py-1 pr-2">Grid</th>
                    <th className="text-left py-1 pr-2">FEN valid</th>
                    <th className="text-left py-1 pr-2">Theme</th>
                    <th className="text-left py-1 pr-2">Orientation</th>
                    <th className="text-left py-1">FEN</th>
                  </tr>
                </thead>
                <tbody>
                  {r.frames.map((f: any, i: number) => (
                    <tr key={i} className="border-b last:border-b-0">
                      <td className="py-1 pr-2 font-mono">{i}</td>
                      <td className="py-1 pr-2">
                        {f.overlay_iou != null
                          ? (f.overlay_iou * 100).toFixed(1) + "%"
                          : "-"}
                      </td>
                      <td className="py-1 pr-2">
                        {f.grid_success != null
                          ? f.grid_success
                            ? "Y"
                            : "N"
                          : "-"}
                      </td>
                      <td className="py-1 pr-2">
                        {f.fen_valid != null
                          ? f.fen_valid
                            ? "Y"
                            : "N"
                          : "-"}
                      </td>
                      <td className="py-1 pr-2">
                        {f.theme_match != null
                          ? f.theme_match
                            ? "Y"
                            : "N"
                          : "-"}
                      </td>
                      <td className="py-1 pr-2">
                        {f.orientation_match != null
                          ? f.orientation_match
                            ? "Y"
                            : "N"
                          : "-"}
                      </td>
                      <td className="py-1 font-mono truncate max-w-[120px]">
                        {f.fen ?? "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
