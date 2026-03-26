"use client";

import type { OverlayTestResult } from "@/lib/api";

interface OverlayTestCardProps {
  result: OverlayTestResult;
  pinned?: boolean;
  onPin?: () => void;
}

export default function OverlayTestCard({ result, pinned, onPin }: OverlayTestCardProps) {
  const wrongSquares = result.square_diffs.length;

  return (
    <div className="border rounded-lg p-3 space-y-3">
      {/* Header */}
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
              {result.filename}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
            <span>{result.elapsed_ms}ms</span>
            <span>
              {Math.round(result.piece_accuracy * 100)}% squares correct
            </span>
          </div>
        </div>
        <div className="text-right shrink-0">
          <span
            className={`text-sm font-bold ${
              result.match ? "text-green-600" : "text-red-600"
            }`}
          >
            {result.match ? "\u2713 Match" : `\u2717 ${wrongSquares} wrong`}
          </span>
        </div>
      </div>

      {/* FEN comparison */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={`text-xs px-2 py-0.5 rounded ${
            result.match
              ? "bg-green-100 text-green-700"
              : "bg-red-100 text-red-700"
          }`}
        >
          Predicted: {result.predicted_fen ?? "error"}
        </span>
        <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
          Expected: {result.expected_fen}
        </span>
      </div>

      {/* Board image */}
      {result.board_image_b64 && (
        <img
          src={`data:image/jpeg;base64,${result.board_image_b64}`}
          alt={result.filename}
          className="w-full max-w-[400px] rounded border"
          loading="lazy"
        />
      )}

      {/* Square diffs */}
      {result.square_diffs.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            Mismatches ({result.square_diffs.length}):
          </p>
          <div className="flex flex-wrap gap-1">
            {result.square_diffs.map((diff) => (
              <span
                key={diff}
                className="text-xs px-1.5 py-0.5 rounded bg-red-50 text-red-700 font-mono"
              >
                {diff}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
