"use client";

export interface FrameResult {
  label: string;
  width: number;
  height: number;
  image_base64: string;
  overlay_score: number;
  otb_score: number;
}

export interface Prediction {
  class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface InspectResult {
  video_id: string;
  title: string;
  title_score?: number | null;
  vertical?: boolean;
  frames: FrameResult[];
  prediction: Prediction | null;
  human_label: string | null;
  human_layout_type: string | null;
  model_version?: string | null;
  error?: string;
}

export function classColor(cls: string) {
  if (cls === "overlay") return "text-green-600";
  if (cls === "otb_only") return "text-blue-600";
  return "text-red-600";
}

export function ProbBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-right text-muted-foreground">{label}</span>
      <div className="flex-1 h-3 bg-muted rounded overflow-hidden">
        <div
          className={`h-full rounded ${
            label === "overlay"
              ? "bg-green-500"
              : label === "otb_only"
              ? "bg-blue-500"
              : "bg-red-500"
          }`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
      <span className="w-12 text-muted-foreground">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

export default function VideoCard({ result }: { result: InspectResult }) {
  const pred = result.prediction;
  const agrees =
    result.human_label && pred
      ? (result.human_label === "approved" && pred.class !== "reject") ||
        (result.human_label === "rejected" && pred.class === "reject")
      : null;

  return (
    <div className="border rounded-lg p-3 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{result.title}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <a href={`/videos/${result.video_id}`} target="_blank" rel="noopener noreferrer" className="hover:underline hover:text-foreground">{result.video_id}</a>
            {result.title_score != null && (
              <span
                className={`px-1 rounded ${
                  result.title_score >= 0.5
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
                title="Title score — below 0.5 would auto-reject"
              >
                title: {result.title_score}
              </span>
            )}
            {result.vertical && (
              <span className="px-1 rounded bg-orange-100 text-orange-700">
                vertical
              </span>
            )}
          </div>
        </div>
        {pred && (
          <div className="text-right shrink-0">
            <span className={`text-sm font-bold ${classColor(pred.class)}`}>
              {pred.class}
            </span>
            <p className="text-xs text-muted-foreground">
              {(pred.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>

      {/* Model vs Human labels */}
      <div className="flex items-center gap-2 flex-wrap">
        {pred && (
          <span className={`text-xs px-2 py-0.5 rounded ${
            pred.class === "reject" ? "bg-red-100 text-red-700" :
            pred.class === "otb_only" ? "bg-blue-100 text-blue-700" :
            "bg-green-100 text-green-700"
          }`}>
            Model: {pred.class}
          </span>
        )}
        {!pred && (
          <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
            Model: no prediction
          </span>
        )}
        {result.human_label && (
          <span className={`text-xs px-2 py-0.5 rounded ${
            result.human_label === "rejected" ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
          }`}>
            Human: {result.human_label}{result.human_label !== "rejected" && result.human_layout_type ? ` (${result.human_layout_type})` : ""}
          </span>
        )}
        {agrees !== null && (
          <span className={`text-xs font-bold ${agrees ? "text-green-600" : "text-red-600"}`}>
            {agrees ? "✓" : "✗"}
          </span>
        )}
      </div>

      {/* All 4 frames */}
      <div className="grid grid-cols-4 gap-2">
        {result.frames.map((frame) => (
          <div key={frame.label} className="space-y-1">
            <img
              src={`data:image/jpeg;base64,${frame.image_base64}`}
              alt={frame.label}
              className="w-full rounded border"
            />
            <div className="text-[10px] text-center text-muted-foreground">
              {frame.label} ({frame.width}x{frame.height})
            </div>
            <div className="flex justify-center gap-2 text-[10px]">
              <span title="Overlay scanner score — auxiliary signal (8 of 3080 model features)">
                OVL: <b>{frame.overlay_score.toFixed(2)}</b>
              </span>
              <span title="OTB detector score — auxiliary signal (8 of 3080 model features)">
                OTB: <b>{frame.otb_score.toFixed(2)}</b>
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Probability distribution */}
      {pred && (
        <div className="space-y-1">
          {Object.entries(pred.probabilities).map(([cls, prob]) => (
            <ProbBar key={cls} label={cls} value={prob} />
          ))}
        </div>
      )}
    </div>
  );
}
