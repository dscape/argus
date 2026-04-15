"use client";

import { ChessBoard } from "@/components/ChessBoard";
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

const LIGHT_SQUARE: [number, number, number] = [240, 217, 181];
const DARK_SQUARE: [number, number, number] = [181, 136, 99];
const CORRECT_COLOR: [number, number, number] = [51, 171, 88];
const ERROR_COLOR: [number, number, number] = [204, 65, 65];
const GT_CHANGED_BORDER = "#F4D03F";
const PRED_CHANGED_BORDER = "#49A0FF";

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function detailImageSrc(
  result: PhysicalRuntimeEvalResult,
  sessionId: string | null | undefined,
): string | null {
  if (result.thumbnail_b64) return `data:image/png;base64,${result.thumbnail_b64}`;
  if (sessionId && result.thumbnail_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.thumbnail_filename);
  }
  if (result.image_b64) return `data:image/png;base64,${result.image_b64}`;
  if (sessionId && result.image_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.image_filename);
  }
  return null;
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function blend(
  base: [number, number, number],
  tint: [number, number, number],
  weight: number,
): string {
  const clamped = clamp(weight);
  const blended = base.map((channel, index) =>
    Math.round((1 - clamped) * channel + clamped * tint[index]!),
  );
  return `rgb(${blended[0]}, ${blended[1]}, ${blended[2]})`;
}

function squareName(squareIndex: number): string {
  const file = String.fromCharCode("a".charCodeAt(0) + (squareIndex % 8));
  const rank = 8 - Math.floor(squareIndex / 8);
  return `${file}${rank}`;
}

function squareBaseColor(squareIndex: number): [number, number, number] {
  const row = Math.floor(squareIndex / 8);
  const col = squareIndex % 8;
  return (row + col) % 2 === 0 ? LIGHT_SQUARE : DARK_SQUARE;
}

function groundTruthSquareStyles(changedSquares?: string[]) {
  const styles: Record<string, { stroke?: string; strokeWidth?: number }> = {};
  for (const square of changedSquares ?? []) {
    styles[square] = { stroke: GT_CHANGED_BORDER, strokeWidth: 3 };
  }
  return styles;
}

function predictionSquareStyles(
  errorSquares?: string[],
  changedSquares?: string[],
  confidences?: number[],
) {
  const styles: Record<
    string,
    { fill?: string; stroke?: string; strokeWidth?: number }
  > = {};
  const errorSet = new Set(errorSquares ?? []);
  const changedSet = new Set(changedSquares ?? []);

  for (let index = 0; index < 64; index += 1) {
    const square = squareName(index);
    const confidence = clamp(confidences?.[index] ?? 1);
    const tint = errorSet.has(square) ? ERROR_COLOR : CORRECT_COLOR;
    const weight = 0.2 + 0.55 * confidence;
    styles[square] = {
      fill: blend(squareBaseColor(index), tint, weight),
    };
    if (changedSet.has(square)) {
      styles[square] = {
        ...styles[square],
        stroke: PRED_CHANGED_BORDER,
        strokeWidth: 3,
      };
    }
  }

  return styles;
}

function deltaLabel(recoveredSquares: number): string {
  if (recoveredSquares > 0) return `+${recoveredSquares} recovered vs single`;
  if (recoveredSquares < 0) return `${Math.abs(recoveredSquares)} worse vs single`;
  return "no change vs single";
}

function BoardPanel({
  title,
  subtitle,
  fen,
  squareStyles,
}: {
  title: string;
  subtitle: string;
  fen?: string;
  squareStyles?: Record<string, { fill?: string; stroke?: string; strokeWidth?: number }>;
}) {
  return (
    <div className="space-y-2">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </p>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </div>
      {fen ? (
        <ChessBoard fen={fen} size={192} squareStyles={squareStyles} />
      ) : (
        <div className="flex aspect-square items-center justify-center rounded border bg-muted/30 text-sm text-muted-foreground">
          Board unavailable
        </div>
      )}
    </div>
  );
}

export default function PhysicalRuntimeCard({
  result,
  pinned,
  sessionId,
  onPin,
}: PhysicalRuntimeCardProps) {
  const imageSrc = detailImageSrc(result, sessionId);
  const clipLabel = result.clip_filename || result.annotation_id;
  const recoveredSquares = result.stateless_error_count - result.temporal_error_count;

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
            <span>{result.elapsed_ms.toFixed(1)}ms</span>
            <span>{deltaLabel(recoveredSquares)}</span>
            <span>single {result.stateless_error_count} wrong</span>
            <span>temp {result.temporal_error_count} wrong</span>
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
            temp {formatPercent(result.temporal_square_accuracy)} · single {formatPercent(result.stateless_square_accuracy)}
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-4">
        <div className="space-y-2">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Board image
            </p>
            <p className="text-xs text-muted-foreground">Rectified frame crop</p>
          </div>
          {imageSrc ? (
            <img
              src={imageSrc}
              alt={`${clipLabel} frame ${result.frame_index}`}
              className="aspect-square w-full rounded border bg-muted/20 object-contain"
              loading="lazy"
            />
          ) : (
            <div className="flex aspect-square items-center justify-center rounded border bg-muted/30 text-sm text-muted-foreground">
              Image unavailable
            </div>
          )}
        </div>

        <BoardPanel
          title="Ground truth"
          subtitle="Yellow border = GT changed square"
          fen={result.gt_fen}
          squareStyles={groundTruthSquareStyles(result.gt_changed_squares)}
        />
        <BoardPanel
          title="Stateless"
          subtitle={`${result.stateless_error_count} wrong · ${formatPercent(result.stateless_square_accuracy)}`}
          fen={result.stateless_fen}
          squareStyles={predictionSquareStyles(
            result.stateless_error_squares,
            result.stateless_changed_squares,
            result.stateless_square_confidences,
          )}
        />
        <BoardPanel
          title="Temporal"
          subtitle={`${result.temporal_error_count} wrong · ${formatPercent(result.temporal_square_accuracy)}`}
          fen={result.temporal_fen}
          squareStyles={predictionSquareStyles(
            result.temporal_error_squares,
            result.temporal_changed_squares,
            result.temporal_square_confidences,
          )}
        />
      </div>

      <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2 xl:grid-cols-4">
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
