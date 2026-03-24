"use client";

import { useState, useEffect, useRef } from "react";
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
import VideoCard, { type InspectResult, classColor } from "@/components/VideoCard";

interface EvalPoint {
  id: number;
  evaluated_at: string;
  accuracy: number;
  sample_size: number;
  notes: string | null;
}

export default function AiScreeningInspector() {
  const [sampleSize, setSampleSize] = useState(20);
  const [results, setResults] = useState<InspectResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [evalHistory, setEvalHistory] = useState<EvalPoint[]>([]);
  const [saving, setSaving] = useState(false);
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const inspectedIds = useRef<Set<string>>(new Set());

  // Fetch eval history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    try {
      const res = await fetch("/api/models/evaluations?model_name=ai_screening");
      if (res.ok) {
        const data = await res.json();
        setEvalHistory(data.evaluations);
      }
    } catch {}
  }

  async function runBatch() {
    setLoading(true);
    setResults([]);
    setProgress({ current: 0, total: 0 });
    setModelVersion(null);

    try {
      // Step 1: get sample video IDs (excluding previously inspected)
      const excludeParam = inspectedIds.current.size > 0
        ? `&exclude=${Array.from(inspectedIds.current).join(",")}`
        : "";
      const sampleRes = await fetch(`/api/models/ai-screening/sample?limit=${sampleSize}${excludeParam}`);
      if (!sampleRes.ok) throw new Error(await sampleRes.text());
      const { video_ids } = await sampleRes.json();

      setProgress({ current: 0, total: video_ids.length });

      // Step 2: inspect each video individually for progress
      for (let i = 0; i < video_ids.length; i++) {
        const res = await fetch("/api/models/ai-screening/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: video_ids[i] }),
        });
        if (res.ok) {
          const result: InspectResult = await res.json();
          setResults((prev) => [...prev, result]);
          inspectedIds.current.add(video_ids[i]);
          if (result.model_version) setModelVersion(result.model_version);
        }
        setProgress({ current: i + 1, total: video_ids.length });
      }
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function saveEval() {
    if (labeled.length === 0) return;
    setSaving(true);
    try {
      const res = await fetch("/api/models/ai-screening/save-eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          accuracy: agrees.length / labeled.length,
          sample_size: labeled.length,
          per_class: classCounts,
          model_version: modelVersion,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      await fetchHistory();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setSaving(false);
    }
  }

  // Compute accuracy summary from results
  const labeled = results.filter((r) => r.human_label && r.prediction);
  const agrees = labeled.filter((r) => {
    const pred = r.prediction!;
    return (
      (r.human_label === "approved" && pred.class !== "reject") ||
      (r.human_label === "rejected" && pred.class === "reject")
    );
  });

  const classCounts: Record<string, { correct: number; total: number }> = {};
  for (const r of labeled) {
    const cls = r.prediction!.class;
    if (!classCounts[cls]) classCounts[cls] = { correct: 0, total: 0 };
    classCounts[cls].total++;
    const isCorrect =
      (r.human_label === "approved" && cls !== "reject") ||
      (r.human_label === "rejected" && cls === "reject");
    if (isCorrect) classCounts[cls].correct++;
  }

  // Prepare chart data (chronological)
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

  // Version annotations: evals where notes starts with "v"
  const versionLines = chartData
    .map((d, i) => ({ ...d, idx: i }))
    .filter((d) => d.notes && /^v\d/i.test(d.notes));

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-center">
        <span className="text-sm text-muted-foreground">Sample from labeled videos:</span>
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
      </div>

      {/* Progress bar */}
      {loading && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Inspecting videos...</span>
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

      {/* Accuracy chart */}
      {chartData.length >= 2 && (
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
                formatter={(value: number) => [`${value}%`, "Accuracy"]}
                labelFormatter={(label, payload) => {
                  const d = payload?.[0]?.payload;
                  return `${label}${d?.notes ? ` — ${d.notes}` : ""} (n=${d?.sample_size ?? "?"})`;
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
      )}

      {/* Accuracy summary + save */}
      {labeled.length > 0 && (
        <div className="border rounded-lg p-3 space-y-2 bg-muted/20">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">
              Accuracy: {agrees.length}/{labeled.length}{" "}
              ({((agrees.length / labeled.length) * 100).toFixed(1)}%)
            </span>
            <div className="flex-1 h-2 bg-muted rounded overflow-hidden max-w-xs">
              <div
                className="h-full bg-green-500 rounded"
                style={{
                  width: `${(agrees.length / labeled.length) * 100}%`,
                }}
              />
            </div>
            <button
              onClick={saveEval}
              disabled={saving}
              className="px-3 py-1 border rounded text-xs disabled:opacity-50 hover:bg-muted"
            >
              {saving ? "Committing..." : "Commit"}
            </button>
          </div>
          <div className="flex gap-3 text-xs text-muted-foreground">
            {Object.entries(classCounts).map(([cls, { correct, total }]) => (
              <span key={cls}>
                <span className={classColor(cls)}>{cls}</span>: {correct}/{total}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            {results.length} result{results.length !== 1 ? "s" : ""}
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {results.map((r) => (
              <VideoCard key={r.video_id} result={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
