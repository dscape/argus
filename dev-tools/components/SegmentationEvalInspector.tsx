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
import {
  sampleSegmentationVideos,
  inspectSegmentation,
  saveSegmentationEval,
  createSegmentationEvalSession,
  listSegmentationEvalSessions,
  updateSegmentationEvalPins,
  type SegmentEvalSessionSummary,
} from "@/lib/api";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
  per_class: {
    segment_consistency?: number;
    gap_consistency?: number;
    piece_readability?: number;
    false_negative_rate?: number;
    coverage_ratio?: number;
  } | null;
}

interface SegmentResult {
  video_id: string;
  duration: number;
  segments: {
    start: number;
    end: number;
    overlay_bbox: number[] | null;
    validation: {
      frames_sampled: number;
      overlay_detected: number;
      grid_found: number;
      pieces_readable: number;
    };
    segment_consistency: number;
    piece_readability: number;
  }[];
  gaps: {
    start: number;
    end: number;
    has_overlay: boolean;
  }[];
  segment_consistency: number;
  gap_consistency: number;
  piece_readability: number;
  false_negative_count: number;
  coverage_ratio: number;
}

interface SegmentationEvalInspectorProps {
  initialSession?: {
    id: string;
    results: any[];
    segment_consistency: number | null;
    gap_consistency: number | null;
    piece_readability: number | null;
    false_negative_rate: number | null;
    coverage_ratio: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  };
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SegmentationEvalInspector({
  initialSession,
}: SegmentationEvalInspectorProps) {
  const router = useRouter();
  const [sampleSize, setSampleSize] = useState(5);
  const [results, setResults] = useState<SegmentResult[]>(
    initialSession?.results ?? []
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
        : []
    )
  );
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [expandedSegDetails, setExpandedSegDetails] = useState<Set<string>>(
    new Set()
  );
  const [selectedFrames, setSelectedFrames] = useState<{
    videoId: string;
    frames: any[];
    label: string;
  } | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(
    initialSession?.id ?? null
  );
  const [recentSessions, setRecentSessions] = useState<
    SegmentEvalSessionSummary[]
  >([]);
  const [showSessionList, setShowSessionList] = useState(false);
  const inspectedVideos = useRef<Set<string>>(new Set());
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
        "/api/models/evaluations?model_name=segmenter"
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
      const { sessions } = await listSegmentationEvalSessions(20);
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

        if (sessionId) {
          updateSegmentationEvalPins(sessionId, {
            [videoId]: !prev.has(videoId),
          }).catch(() => {});
        }
        return next;
      });
      setExpandedIds((prev) => {
        const next = new Set(prev);
        next.delete(videoId);
        return next;
      });
    },
    [sessionId]
  );

  const toggleExpand = useCallback((videoId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) next.delete(videoId);
      else next.add(videoId);
      return next;
    });
  }, []);

  const toggleSegDetails = useCallback((videoId: string) => {
    setExpandedSegDetails((prev) => {
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
    setPinnedIds(new Set());
    setExpandedIds(new Set());
    setExpandedSegDetails(new Set());
    setSessionId(null);

    const collected: SegmentResult[] = [];
    const autoPinned = new Set<string>();

    try {
      // Step 1: get sample video_ids
      const excludeArr = Array.from(inspectedVideos.current);
      const { video_ids } = await sampleSegmentationVideos(
        sampleSize,
        excludeArr
      );

      setProgress({ current: 0, total: video_ids.length });

      // Step 2: inspect each video
      for (let i = 0; i < video_ids.length; i++) {
        if (controller.signal.aborted) break;

        const result = await inspectSegmentation(video_ids[i]);
        collected.push(result);
        setResults((prev) => [...prev, result]);
        inspectedVideos.current.add(video_ids[i]);

        // Auto-pin: false negatives or low segment consistency
        const hasFalseNeg = result.gaps?.some(
          (g: any) => g.has_overlay === true
        );
        const lowConsistency = result.segment_consistency < 0.8;
        if (hasFalseNeg || lowConsistency) {
          autoPinned.add(result.video_id);
          setPinnedIds((prev) => new Set([...prev, result.video_id]));
        }

        setProgress({ current: i + 1, total: video_ids.length });
      }

      // Step 3: compute aggregate metrics & save
      const total = collected.length;
      if (total > 0) {
        const avgSegConsistency =
          collected.reduce((s, r) => s + r.segment_consistency, 0) / total;
        const avgGapConsistency =
          collected.reduce((s, r) => s + r.gap_consistency, 0) / total;
        const avgPieceReadability =
          collected.reduce((s, r) => s + r.piece_readability, 0) / total;
        const totalFalseNegs = collected.reduce(
          (s, r) => s + r.false_negative_count,
          0
        );
        const totalGaps = collected.reduce(
          (s, r) => s + (r.gaps?.length ?? 0),
          0
        );
        const falseNegRate =
          totalGaps > 0 ? totalFalseNegs / totalGaps : 0;
        const avgCoverage =
          collected.reduce((s, r) => s + r.coverage_ratio, 0) / total;

        const saveRes = await saveSegmentationEval({
          segment_consistency: avgSegConsistency,
          gap_consistency: avgGapConsistency,
          piece_readability: avgPieceReadability,
          false_negative_rate: falseNegRate,
          coverage_ratio: avgCoverage,
          sample_size: total,
        });
        const evaluationId = saveRes?.id ?? null;
        await fetchHistory();

        // Step 4: save session
        try {
          const pinStateObj: Record<string, boolean> = {};
          autoPinned.forEach((id) => (pinStateObj[id] = true));

          const { session_id } = await createSegmentationEvalSession({
            results: collected,
            segment_consistency: avgSegConsistency,
            gap_consistency: avgGapConsistency,
            piece_readability: avgPieceReadability,
            false_negative_rate: falseNegRate,
            coverage_ratio: avgCoverage,
            sample_size: total,
            pin_state: pinStateObj,
            evaluation_id: evaluationId,
          });
          setSessionId(session_id);
          router.replace(`/models/segmentation/${session_id}`, {
            scroll: false,
          });
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

  // Aggregate metrics
  const total = results.length;
  const avgSegConsistency =
    total > 0
      ? results.reduce((s, r) => s + r.segment_consistency, 0) / total
      : 0;
  const avgGapConsistency =
    total > 0
      ? results.reduce((s, r) => s + r.gap_consistency, 0) / total
      : 0;
  const avgPieceReadability =
    total > 0
      ? results.reduce((s, r) => s + r.piece_readability, 0) / total
      : 0;
  const totalFalseNegs = results.reduce(
    (s, r) => s + r.false_negative_count,
    0
  );

  // Sort into pinned / unpinned
  const { pinned, unpinned } = useMemo(() => {
    const p: SegmentResult[] = [];
    const u: SegmentResult[] = [];
    for (const r of results) {
      if (pinnedIds.has(r.video_id)) p.push(r);
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

  // --- Render helpers ---

  function renderTimelineBar(result: SegmentResult) {
    const { duration, segments, gaps } = result;
    if (duration <= 0) return null;

    const activeKey = selectedFrames?.videoId === result.video_id ? selectedFrames.label : null;

    return (
      <div className="space-y-1">
        <div className="relative h-6 bg-muted rounded overflow-hidden w-full">
          {segments.map((seg, i) => {
            const key = `seg-${i}`;
            const isActive = activeKey === key;
            return (
              <div
                key={key}
                className={`absolute top-0 bottom-0 cursor-pointer transition-opacity ${
                  isActive ? "bg-green-500 ring-1 ring-green-300" : "bg-green-500/70 hover:bg-green-500/90"
                } border-r border-green-600/30`}
                style={{
                  left: `${(seg.start / duration) * 100}%`,
                  width: `${((seg.end - seg.start) / duration) * 100}%`,
                }}
                title={`Segment ${i + 1}: ${formatTime(seg.start)} - ${formatTime(seg.end)} — click to inspect frames`}
                onClick={() =>
                  setSelectedFrames(
                    isActive ? null :
                    { videoId: result.video_id, frames: seg.frames ?? [], label: key }
                  )
                }
              />
            );
          })}
          {gaps.map((gap, i) => {
            const key = `gap-${i}`;
            const isActive = activeKey === key;
            return (
              <div
                key={key}
                className={`absolute top-0 bottom-0 cursor-pointer transition-opacity ${
                  gap.has_overlay
                    ? isActive ? "bg-red-500 ring-1 ring-red-300" : "bg-red-500/50 hover:bg-red-500/70 border border-red-500"
                    : isActive ? "bg-gray-400/60" : "bg-gray-400/30 hover:bg-gray-400/50"
                }`}
                style={{
                  left: `${(gap.start / duration) * 100}%`,
                  width: `${((gap.end - gap.start) / duration) * 100}%`,
                }}
                title={`Gap ${i + 1}: ${formatTime(gap.start)} - ${formatTime(gap.end)}${
                  gap.has_overlay ? " (FALSE NEGATIVE)" : ""
                } — click to inspect frames`}
                onClick={() =>
                  setSelectedFrames(
                    isActive ? null :
                    { videoId: result.video_id, frames: gap.frames ?? [], label: key }
                  )
                }
              />
            );
          })}
        </div>

        {/* Frame strip for selected region */}
        {selectedFrames?.videoId === result.video_id && selectedFrames.frames.length > 0 && (
          <div className="flex gap-1 overflow-x-auto pb-1">
            {selectedFrames.frames.map((f: any, i: number) => (
              <div key={i} className="flex-shrink-0 text-center">
                {f.thumbnail_b64 ? (
                  <img
                    src={`data:image/jpeg;base64,${f.thumbnail_b64}`}
                    alt={`t=${f.time}s`}
                    className="h-20 w-auto rounded border object-cover"
                  />
                ) : (
                  <div className="h-20 w-28 rounded border bg-muted flex items-center justify-center text-xs text-muted-foreground">
                    no frame
                  </div>
                )}
                <div className="text-[10px] text-muted-foreground mt-0.5">{formatTime(f.time)}</div>
                <div className="flex gap-0.5 justify-center mt-0.5">
                  {f.overlay_detected !== undefined && (
                    <span className={`text-[9px] px-1 rounded ${f.overlay_detected ? "bg-green-500/20 text-green-700" : "bg-red-500/20 text-red-700"}`}>
                      {f.overlay_detected ? "ov✓" : "ov✗"}
                    </span>
                  )}
                  {f.grid_found !== undefined && (
                    <span className={`text-[9px] px-1 rounded ${f.grid_found ? "bg-green-500/20 text-green-700" : "bg-gray-500/20 text-gray-600"}`}>
                      {f.grid_found ? "grid✓" : "grid✗"}
                    </span>
                  )}
                  {f.pieces_readable !== undefined && (
                    <span className={`text-[9px] px-1 rounded ${f.pieces_readable ? "bg-green-500/20 text-green-700" : "bg-gray-500/20 text-gray-600"}`}>
                      {f.pieces_readable ? "pc✓" : "pc✗"}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  function renderVideoCard(result: SegmentResult, isPinned: boolean) {
    const hasFalseNegs = result.gaps?.some((g) => g.has_overlay === true);
    const showDetails = expandedSegDetails.has(result.video_id);

    return (
      <div
        key={result.video_id}
        className={`border rounded-lg p-3 space-y-3 ${
          isPinned ? "ring-2 ring-yellow-500/50" : ""
        }`}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-medium truncate">
                {result.video_id}
              </span>
              <button
                onClick={() => togglePin(result.video_id)}
                className="text-xs px-1.5 py-0.5 rounded border hover:bg-muted transition-colors flex-shrink-0"
                title={isPinned ? "Unpin" : "Pin"}
              >
                {isPinned ? "\u2605" : "\u2606"}
              </button>
            </div>
            <div className="text-xs text-muted-foreground mt-0.5">
              {formatTime(result.duration)} &middot;{" "}
              {result.segments.length} segment
              {result.segments.length !== 1 ? "s" : ""} &middot;{" "}
              {result.gaps.length} gap{result.gaps.length !== 1 ? "s" : ""}
            </div>
          </div>

          {/* Metrics badges */}
          <div className="flex gap-1.5 flex-shrink-0">
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                result.segment_consistency >= 0.9
                  ? "bg-green-500/20 text-green-700"
                  : result.segment_consistency >= 0.7
                    ? "bg-yellow-500/20 text-yellow-700"
                    : "bg-red-500/20 text-red-700"
              }`}
            >
              seg {(result.segment_consistency * 100).toFixed(0)}%
            </span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                result.piece_readability >= 0.9
                  ? "bg-green-500/20 text-green-700"
                  : result.piece_readability >= 0.7
                    ? "bg-yellow-500/20 text-yellow-700"
                    : "bg-red-500/20 text-red-700"
              }`}
            >
              read {(result.piece_readability * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Timeline bar */}
        {renderTimelineBar(result)}

        {/* False negative warnings */}
        {hasFalseNegs && (
          <div className="text-xs text-red-600 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded px-2 py-1.5">
            False negative: overlay detected in gap
            {result.gaps.filter((g) => g.has_overlay).length > 1 ? "s" : ""}{" "}
            {result.gaps
              .filter((g) => g.has_overlay)
              .map((g) => `${formatTime(g.start)}-${formatTime(g.end)}`)
              .join(", ")}
          </div>
        )}

        {/* Segment details (collapsed) */}
        <div>
          <button
            onClick={() => toggleSegDetails(result.video_id)}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showDetails ? "\u25BC" : "\u25B6"} Segment details
          </button>
          {showDetails && (
            <div className="mt-2 space-y-2">
              {result.segments.map((seg, i) => (
                <div
                  key={i}
                  className="text-xs border rounded p-2 bg-muted/10 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">
                      Segment {i + 1}: {formatTime(seg.start)} -{" "}
                      {formatTime(seg.end)}
                    </span>
                    <span
                      className={`px-1.5 py-0.5 rounded ${
                        seg.segment_consistency >= 0.9
                          ? "bg-green-500/20 text-green-700"
                          : seg.segment_consistency >= 0.7
                            ? "bg-yellow-500/20 text-yellow-700"
                            : "bg-red-500/20 text-red-700"
                      }`}
                    >
                      {(seg.segment_consistency * 100).toFixed(0)}%
                    </span>
                  </div>
                  {seg.overlay_bbox && (
                    <div className="text-muted-foreground">
                      bbox: [{seg.overlay_bbox.map((v) => v.toFixed(0)).join(", ")}]
                    </div>
                  )}
                  <div className="text-muted-foreground">
                    Validation ({seg.validation.frames_sampled} frames):
                    overlay {seg.validation.overlay_detected}/
                    {seg.validation.frames_sampled}, grid{" "}
                    {seg.validation.grid_found}/
                    {seg.validation.frames_sampled}, pieces{" "}
                    {seg.validation.pieces_readable}/
                    {seg.validation.frames_sampled}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm text-muted-foreground">
          Sample segmented videos:
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
                      router.push(`/models/segmentation/${s.id}`);
                    }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-muted/50 transition-colors border-b last:border-b-0 ${
                      s.id === sessionId ? "bg-muted/30" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-muted-foreground">
                        {s.id}
                      </span>
                      {s.segment_consistency != null && (
                        <span className="font-medium">
                          {(s.segment_consistency * 100).toFixed(1)}%
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
                      {s.piece_readability != null
                        ? ` \u00b7 ${(s.piece_readability * 100).toFixed(1)}% read`
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
            <span>Inspecting segmentations...</span>
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

      {/* Chart: segment consistency over time */}
      {chartData.length >= 2 && (
        <div className="grid gap-4 grid-cols-1">
          <div className="border rounded-lg p-3">
            <h3 className="text-sm font-medium mb-2">
              Segment consistency over time
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
                    name === "accuracy"
                      ? "Segment consistency"
                      : String(name),
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
        </div>
      )}

      {/* Summary bar */}
      {total > 0 && (
        <div className="border rounded-lg p-3 space-y-2 bg-muted/20">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Segment consistency:{" "}
              {(avgSegConsistency * 100).toFixed(1)}%
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{ width: `${avgSegConsistency * 100}%` }}
              />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Gap consistency:{" "}
              {(avgGapConsistency * 100).toFixed(1)}%
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{ width: `${avgGapConsistency * 100}%` }}
              />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Piece readability:{" "}
              {(avgPieceReadability * 100).toFixed(1)}%
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{ width: `${avgPieceReadability * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            False negatives:{" "}
            <span className={totalFalseNegs > 0 ? "text-red-600 font-medium" : ""}>
              {totalFalseNegs}
            </span>
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
                {pinned.map((r) => renderVideoCard(r, true))}
              </div>
            </div>
          )}

          {/* Unpinned */}
          {unpinned.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                {pinned.length > 0
                  ? `Others (${unpinned.length})`
                  : `${results.length} results`}
              </p>

              {/* Thumbnail strip for collapsed */}
              <div className="flex flex-wrap gap-1.5">
                {unpinned
                  .filter((r) => !expandedIds.has(r.video_id))
                  .map((r) => {
                    const hasFN = r.gaps?.some((g) => g.has_overlay);
                    const lowCons = r.segment_consistency < 0.8;
                    const isOk = !hasFN && !lowCons;

                    return (
                      <button
                        key={r.video_id}
                        onClick={() =>
                          !isOk
                            ? togglePin(r.video_id)
                            : toggleExpand(r.video_id)
                        }
                        title={`${r.video_id}\n${r.segments.length} segments, ${r.gaps.length} gaps\nconsistency: ${(r.segment_consistency * 100).toFixed(0)}%`}
                        className={`relative px-2 py-1.5 rounded border text-xs font-mono overflow-hidden flex-shrink-0 transition-all hover:ring-2 hover:ring-foreground/30 ${
                          isOk
                            ? "bg-green-500/10 border-green-500/30"
                            : "bg-red-500/10 border-red-500/30"
                        }`}
                      >
                        <span className="truncate max-w-[80px] inline-block">
                          {r.video_id.slice(0, 8)}
                        </span>
                        <span
                          className={`ml-1 ${
                            isOk ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {(r.segment_consistency * 100).toFixed(0)}%
                        </span>
                      </button>
                    );
                  })}
              </div>

              {/* Expanded cards */}
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
                        {renderVideoCard(r, false)}
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
