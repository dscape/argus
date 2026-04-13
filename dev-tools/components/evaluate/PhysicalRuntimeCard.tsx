"use client";

import {
  physicalRuntimeSessionImageUrl,
  type PhysicalRuntimeEvalResult,
} from "@/lib/api";

interface PhysicalRuntimeCardProps {
  result: PhysicalRuntimeEvalResult;
  pinned?: boolean;
  sessionId?: string | null;
  onPin?: () => void;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatDelta(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${value}`;
}

function detailImageSrc(
  result: PhysicalRuntimeEvalResult,
  sessionId: string | null | undefined,
): string | null {
  if (result.image_b64) return `data:image/png;base64,${result.image_b64}`;
  if (sessionId && result.image_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.image_filename);
  }
  return null;
}

export default function PhysicalRuntimeCard({
  result,
  pinned,
  sessionId,
  onPin,
}: PhysicalRuntimeCardProps) {
  const imageSrc = detailImageSrc(result, sessionId);
  const clipLabel = result.clip_filename || result.annotation_id;
  const errorDelta = result.temporal_error_count - result.stateless_error_count;

  return (
    <div className="border rounded-lg p-3 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            {onPin && (
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
            )}
            <p className="text-sm font-medium font-mono truncate">
              {clipLabel} · f{result.frame_index.toString().padStart(4, "0")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mt-0.5">
            <span>
              gtΔ {formatDelta(result.gt_change_count)} · singleΔ {formatDelta(result.stateless_change_count)}
              {" "}· tempΔ {formatDelta(result.temporal_change_count)}
            </span>
            <span>{result.elapsed_ms.toFixed(1)}ms</span>
            <span>single conf {result.stateless_mean_confidence.toFixed(2)}</span>
            <span>temp conf {result.temporal_mean_confidence.toFixed(2)}</span>
          </div>
        </div>
        <div className="text-right shrink-0">
          <span
            className={`text-sm font-bold ${
              result.temporal_exact_match ? "text-green-600" : "text-red-600"
            }`}
          >
            {result.temporal_exact_match
              ? "✓ Exact"
              : `✗ ${result.temporal_error_count} wrong`}
          </span>
          <div className="text-xs text-muted-foreground mt-0.5">
            temp {errorDelta > 0 ? "+" : ""}
            {errorDelta} vs single
          </div>
        </div>
      </div>

      {imageSrc ? (
        <img
          src={imageSrc}
          alt={`${clipLabel} frame ${result.frame_index}`}
          className="w-full rounded border"
          loading="lazy"
        />
      ) : (
        <div className="w-full rounded border bg-muted/30 px-3 py-8 text-center text-sm text-muted-foreground">
          Image unavailable
        </div>
      )}

      <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2 xl:grid-cols-4">
        <div>Temporal square: {formatPercent(result.temporal_square_accuracy)}</div>
        <div>Temporal non-empty: {formatPercent(result.temporal_non_empty_accuracy)}</div>
        <div>Stateless square: {formatPercent(result.stateless_square_accuracy)}</div>
        <div>Stateless non-empty: {formatPercent(result.stateless_non_empty_accuracy)}</div>
        <div>Temporal exact: {result.temporal_exact_match ? "yes" : "no"}</div>
        <div>Stateless exact: {result.stateless_exact_match ? "yes" : "no"}</div>
        <div>
          Annotation: <code>{result.annotation_id}</code>
        </div>
        <div>
          Board: <code>{result.board_path.split("/").at(-1)}</code>
        </div>
      </div>
    </div>
  );
}
