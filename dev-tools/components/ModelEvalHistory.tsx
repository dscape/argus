"use client";

import { useState, useEffect } from "react";

interface EvalRun {
  id: number;
  model_name: string;
  evaluated_at: string;
  sample_size: number;
  accuracy: number;
  precision_avg: number;
  recall_avg: number;
  f1_avg: number;
  per_class: Record<string, { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }>;
  threshold: number | null;
  auto_rate: number | null;
  notes: string | null;
}

// ── Accuracy Chart ──────────────────────────────────────────

function AccuracyChart({ evaluations }: { evaluations: EvalRun[] }) {
  const [hovered, setHovered] = useState<number | null>(null);

  // Chronological order (oldest first)
  const sorted = [...evaluations].reverse();
  if (sorted.length < 2) return null;

  const W = 600, H = 200;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Y scale: accuracy 0-1
  const yMin = 0, yMax = 1;
  const toY = (v: number) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
  const toX = (i: number) => pad.left + (i / (sorted.length - 1)) * plotW;

  const points = sorted.map((ev, i) => ({
    x: toX(i),
    y: toY(ev.accuracy ?? 0),
    ev,
    idx: i,
  }));

  const polyline = points.map((p) => `${p.x},${p.y}`).join(" ");

  // Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];

  // Version markers: evaluations with notes starting with "v"
  const versionMarkers = points.filter(
    (p) => p.ev.notes && /^v\d/i.test(p.ev.notes)
  );

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[600px]">
        {/* Grid lines */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={pad.left}
              y1={toY(t)}
              x2={W - pad.right}
              y2={toY(t)}
              stroke="currentColor"
              strokeOpacity={0.1}
            />
            <text
              x={pad.left - 6}
              y={toY(t) + 4}
              textAnchor="end"
              className="fill-muted-foreground"
              fontSize={10}
            >
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* Version markers */}
        {versionMarkers.map((p) => (
          <g key={`v-${p.idx}`}>
            <line
              x1={p.x}
              y1={pad.top}
              x2={p.x}
              y2={H - pad.bottom}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeDasharray="4,4"
            />
            <text
              x={p.x}
              y={pad.top - 6}
              textAnchor="middle"
              className="fill-foreground"
              fontSize={10}
              fontWeight="bold"
            >
              {p.ev.notes!.split(":")[0].trim()}
            </text>
          </g>
        ))}

        {/* Line */}
        <polyline
          points={polyline}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinejoin="round"
        />

        {/* Points */}
        {points.map((p) => (
          <circle
            key={p.idx}
            cx={p.x}
            cy={p.y}
            r={hovered === p.idx ? 5 : 3}
            className={hovered === p.idx ? "fill-foreground" : "fill-foreground/60"}
            onMouseEnter={() => setHovered(p.idx)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "pointer" }}
          />
        ))}

        {/* X-axis labels (every few points) */}
        {points
          .filter((_, i) => i === 0 || i === points.length - 1 || i % Math.ceil(points.length / 6) === 0)
          .map((p) => (
            <text
              key={`x-${p.idx}`}
              x={p.x}
              y={H - pad.bottom + 16}
              textAnchor="middle"
              className="fill-muted-foreground"
              fontSize={9}
            >
              {new Date(p.ev.evaluated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
            </text>
          ))}
      </svg>

      {/* Tooltip */}
      {hovered !== null && points[hovered] && (
        <div
          className="absolute bg-background border rounded px-2 py-1 shadow-md text-xs pointer-events-none"
          style={{
            left: `${(points[hovered].x / W) * 100}%`,
            top: `${(points[hovered].y / H) * 100}%`,
            transform: "translate(-50%, -120%)",
          }}
        >
          <div className="font-medium">
            {new Date(points[hovered].ev.evaluated_at).toLocaleDateString()}{" "}
            {new Date(points[hovered].ev.evaluated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </div>
          <div>Accuracy: <b>{((points[hovered].ev.accuracy ?? 0) * 100).toFixed(1)}%</b></div>
          <div>F1: {((points[hovered].ev.f1_avg ?? 0) * 100).toFixed(1)}%</div>
          <div>N: {points[hovered].ev.sample_size}</div>
          {points[hovered].ev.notes && (
            <div className="text-muted-foreground">{points[hovered].ev.notes}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────

export default function ModelEvalHistory() {
  const [evaluations, setEvaluations] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [runningEval, setRunningEval] = useState(false);
  const [sampleSize, setSampleSize] = useState(500);
  const [notes, setNotes] = useState("");

  async function fetchHistory() {
    try {
      const res = await fetch("/api/models/evaluations");
      if (res.ok) {
        const data = await res.json();
        setEvaluations(data.evaluations);
      }
    } catch (e) {
      console.warn("Failed to fetch evaluation history:", e);
    }
  }

  useEffect(() => {
    fetchHistory();
  }, []);

  async function runEvaluation() {
    setRunningEval(true);
    try {
      const res = await fetch("/api/models/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: "ai_screening",
          sample_size: sampleSize,
          notes: notes.trim() || null,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setNotes("");
      await fetchHistory();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setRunningEval(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Run new evaluation */}
      <div className="flex gap-2 items-center flex-wrap">
        <span className="text-sm">Run evaluation:</span>
        <select className="border rounded px-2 py-1.5 text-sm" disabled>
          <option>ai_screening</option>
        </select>
        <label className="text-xs text-muted-foreground">
          Sample:
          <input
            type="number"
            value={sampleSize}
            onChange={(e) => setSampleSize(Number(e.target.value))}
            min={50}
            max={5000}
            step={50}
            className="w-20 ml-1 px-2 py-1.5 border rounded text-sm"
          />
        </label>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="e.g., v3: added new features"
          className="px-2 py-1.5 border rounded text-sm w-52"
        />
        <button
          onClick={runEvaluation}
          disabled={runningEval}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {runningEval ? "Evaluating..." : "Run"}
        </button>
      </div>

      {/* Accuracy chart */}
      {evaluations.length >= 2 && <AccuracyChart evaluations={evaluations} />}

      {/* History table */}
      {evaluations.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="p-2">Date</th>
                <th className="p-2">Model</th>
                <th className="p-2">N</th>
                <th className="p-2">Accuracy</th>
                <th className="p-2">Precision</th>
                <th className="p-2">Recall</th>
                <th className="p-2">F1</th>
                <th className="p-2">Per-Class</th>
                <th className="p-2">Notes</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.map((ev) => (
                <tr key={ev.id} className="border-b hover:bg-muted/20">
                  <td className="p-2 text-xs text-muted-foreground">
                    {new Date(ev.evaluated_at).toLocaleDateString()}{" "}
                    {new Date(ev.evaluated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="p-2 font-mono text-xs">{ev.model_name}</td>
                  <td className="p-2">{ev.sample_size}</td>
                  <td className="p-2 font-mono font-bold">
                    {ev.accuracy !== null ? (ev.accuracy * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="p-2 font-mono">
                    {ev.precision_avg !== null ? (ev.precision_avg * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="p-2 font-mono">
                    {ev.recall_avg !== null ? (ev.recall_avg * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="p-2 font-mono">
                    {ev.f1_avg !== null ? (ev.f1_avg * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="p-2">
                    {ev.per_class && (
                      <div className="flex gap-2 text-[10px]">
                        {Object.entries(ev.per_class).map(([cls, m]) => (
                          <span key={cls} className="bg-muted px-1 rounded">
                            {cls}: F1={(m.f1 * 100).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td className="p-2 text-xs text-muted-foreground max-w-32 truncate">
                    {ev.notes || ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          No evaluations yet. Run one to establish a baseline.
        </p>
      )}
    </div>
  );
}
