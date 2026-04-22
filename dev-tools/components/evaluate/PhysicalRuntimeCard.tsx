"use client";

import { type CSSProperties, useEffect, useMemo, useState } from "react";

import { ChessBoard } from "@/components/ChessBoard";
import {
  groundTruthSquareStyles,
  predictionSquareStyles,
  type PhysicalBoardSquareStyle,
} from "@/components/evaluate/physicalBoardStyles";
import {
  physicalRuntimeSessionImageUrl,
  type PhysicalRuntimeEvalResult,
  type PhysicalRuntimeGeometryBbox,
  type PhysicalRuntimeGeometryQuad,
} from "@/lib/api";

interface PhysicalRuntimeCardProps {
  result: PhysicalRuntimeEvalResult;
  pinned?: boolean;
  sessionId?: string | null;
  onPin?: () => void;
}

interface HoveredSquareState {
  boardTitle: string;
  square: string;
  fen?: string;
}

interface SquareEvidenceEntry {
  square: string;
  bbox: PhysicalRuntimeGeometryBbox | null;
  gtClass: string;
  statelessClass: string;
  temporalClass: string;
  top1Class: string | null;
  top1Probability: number | null;
  probabilities: Array<{ className: string; probability: number }>;
}

const SQUARE_CLASS_NAMES = [
  "empty",
  "P",
  "N",
  "B",
  "R",
  "Q",
  "K",
  "p",
  "n",
  "b",
  "r",
  "q",
  "k",
] as const;

const SQUARE_CLASS_LABELS: Record<string, string> = {
  empty: "empty",
  P: "white pawn",
  N: "white knight",
  B: "white bishop",
  R: "white rook",
  Q: "white queen",
  K: "white king",
  p: "black pawn",
  n: "black knight",
  b: "black bishop",
  r: "black rook",
  q: "black queen",
  k: "black king",
};

const BOARD_SQUARES = Array.from({ length: 64 }, (_value, index) => {
  const file = String.fromCharCode("a".charCodeAt(0) + (index % 8));
  const rank = 8 - Math.floor(index / 8);
  return `${file}${rank}`;
});

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function detailImageSrc(
  result: PhysicalRuntimeEvalResult,
  sessionId: string | null | undefined,
): string | null {
  if (result.image_b64) return `data:image/png;base64,${result.image_b64}`;
  if (sessionId && result.image_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.image_filename);
  }
  if (result.thumbnail_b64) return `data:image/png;base64,${result.thumbnail_b64}`;
  if (sessionId && result.thumbnail_filename) {
    return physicalRuntimeSessionImageUrl(sessionId, result.thumbnail_filename);
  }
  return null;
}

function deltaLabel(recoveredSquares: number): string {
  if (recoveredSquares > 0) return `+${recoveredSquares} recovered vs single`;
  if (recoveredSquares < 0) return `${Math.abs(recoveredSquares)} worse vs single`;
  return "no change vs single";
}

function formatTransition(
  kind: string | null | undefined,
  label: string | null | undefined,
): string {
  if (kind === "stay") return "stay";
  if (kind === "legal_move") return label ?? "legal move";
  if (kind === "illegal_single_piece_transfer") {
    return label ? `illegal ${label}` : "illegal single-piece transfer";
  }
  if (kind === "unexplained_disturbance") return "unexplained disturbance";
  if (kind === "smoothed_logits") return "EMA-smoothed logits";
  if (kind === "initial") return "initial";
  return label ?? "-";
}

function quadPoints(quad: PhysicalRuntimeGeometryQuad): string {
  return quad.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ");
}

function bboxRect(bbox: PhysicalRuntimeGeometryBbox): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const [xmin, ymin, xmax, ymax] = bbox;
  return {
    x: xmin * 1000,
    y: ymin * 1000,
    width: Math.max((xmax - xmin) * 1000, 1),
    height: Math.max((ymax - ymin) * 1000, 1),
  };
}

function geometryBounds(
  geometrySquareQuads: Record<string, PhysicalRuntimeGeometryQuad>,
  geometryPieceBboxes: Record<string, PhysicalRuntimeGeometryBbox>,
): PhysicalRuntimeGeometryBbox | null {
  let xmin = Number.POSITIVE_INFINITY;
  let ymin = Number.POSITIVE_INFINITY;
  let xmax = Number.NEGATIVE_INFINITY;
  let ymax = Number.NEGATIVE_INFINITY;

  for (const quad of Object.values(geometrySquareQuads)) {
    for (const [x, y] of quad) {
      xmin = Math.min(xmin, x);
      ymin = Math.min(ymin, y);
      xmax = Math.max(xmax, x);
      ymax = Math.max(ymax, y);
    }
  }

  for (const [x0, y0, x1, y1] of Object.values(geometryPieceBboxes)) {
    xmin = Math.min(xmin, x0);
    ymin = Math.min(ymin, y0);
    xmax = Math.max(xmax, x1);
    ymax = Math.max(ymax, y1);
  }

  if (![xmin, ymin, xmax, ymax].every(Number.isFinite)) return null;
  if (xmax <= xmin || ymax <= ymin) return null;
  return [xmin, ymin, xmax, ymax];
}

function expandedBounds(
  bbox: PhysicalRuntimeGeometryBbox,
  paddingRatio = 0.06,
  minPadding = 0.015,
): PhysicalRuntimeGeometryBbox {
  const [xmin, ymin, xmax, ymax] = bbox;
  const xPad = Math.max((xmax - xmin) * paddingRatio, minPadding);
  const yPad = Math.max((ymax - ymin) * paddingRatio, minPadding);
  return [
    Math.max(0, xmin - xPad),
    Math.max(0, ymin - yPad),
    Math.min(1, xmax + xPad),
    Math.min(1, ymax + yPad),
  ];
}

function focusedViewportStyle(
  bbox: PhysicalRuntimeGeometryBbox,
): CSSProperties | null {
  const [xmin, ymin, xmax, ymax] = bbox;
  const width = Math.max(xmax - xmin, 1e-3);
  const height = Math.max(ymax - ymin, 1e-3);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;

  return {
    width: `${(1 / width) * 100}%`,
    height: `${(1 / height) * 100}%`,
    left: `${(-xmin / width) * 100}%`,
    top: `${(-ymin / height) * 100}%`,
  };
}

function projectedCropStyle(
  bbox: PhysicalRuntimeGeometryBbox,
): CSSProperties | null {
  const [xmin, ymin, xmax, ymax] = bbox;
  const width = Math.max(xmax - xmin, 1e-3);
  const height = Math.max(ymax - ymin, 1e-3);
  const scale = 1 / Math.max(width, height);
  if (!Number.isFinite(scale)) return null;

  return {
    width: `${scale * 100}%`,
    height: `${scale * 100}%`,
    left: `${-xmin * scale * 100}%`,
    top: `${(1 - height * scale - ymin * scale) * 100}%`,
  };
}

function squareCoords(square: string): { row: number; col: number } | null {
  if (!/^[a-h][1-8]$/i.test(square)) return null;
  const col = square[0]!.toLowerCase().charCodeAt(0) - "a".charCodeAt(0);
  const row = 8 - Number.parseInt(square[1]!, 10);
  return { row, col };
}

function squareIndex(square: string): number | null {
  const coords = squareCoords(square);
  if (!coords) return null;
  return coords.row * 8 + coords.col;
}

function pieceSymbolAtSquare(fen: string | undefined, square: string): string | null {
  if (!fen) return null;
  const coords = squareCoords(square);
  if (!coords) return null;

  const rows = (fen.split(" ")[0] ?? fen).split("/");
  let col = 0;
  for (const symbol of rows[coords.row] ?? "") {
    if (symbol >= "1" && symbol <= "8") {
      col += Number.parseInt(symbol, 10);
      continue;
    }
    if (col === coords.col) return symbol;
    col += 1;
  }
  return null;
}

function pieceLabel(symbol: string | null): string {
  if (!symbol) return "empty";
  return SQUARE_CLASS_LABELS[symbol] ?? symbol;
}

function boardClassAtSquare(fen: string | undefined, square: string): string {
  return pieceSymbolAtSquare(fen, square) ?? "empty";
}

function compactClassLabel(className: string | null): string {
  if (className == null || className === "empty") return "∅";
  return className;
}

function probabilityRows(probabilities: number[] | undefined): Array<{
  className: string;
  probability: number;
}> {
  if (!probabilities || probabilities.length === 0) return [];
  return probabilities
    .map((probability, classIndex) => ({
      className: SQUARE_CLASS_NAMES[classIndex] ?? `cls${classIndex}`,
      probability,
    }))
    .sort((left, right) => right.probability - left.probability);
}

function defaultEvidenceSquare(result: PhysicalRuntimeEvalResult): string {
  return (
    result.stateless_error_squares?.[0] ??
    result.temporal_error_squares?.[0] ??
    result.gt_changed_squares?.[0] ??
    BOARD_SQUARES[0]!
  );
}

function GeometryOverlay({
  geometrySquareQuads,
  geometryPieceBboxes,
  occupiedSquares,
  hoveredSquare,
}: {
  geometrySquareQuads: Record<string, PhysicalRuntimeGeometryQuad>;
  geometryPieceBboxes: Record<string, PhysicalRuntimeGeometryBbox>;
  occupiedSquares: Set<string>;
  hoveredSquare?: string | null;
}) {
  const baseEntries = Object.entries(geometrySquareQuads);
  const bboxEntries = Object.entries(geometryPieceBboxes);
  if (baseEntries.length === 0 && bboxEntries.length === 0) return null;

  return (
    <svg
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {baseEntries.map(([square, quad]) => {
        const hovered = square === hoveredSquare;
        return (
          <polygon
            key={`base-${square}`}
            points={quadPoints(quad)}
            fill={hovered ? "rgba(217, 70, 239, 0.12)" : "none"}
            stroke={hovered ? "rgba(217, 70, 239, 0.95)" : "rgba(250, 204, 21, 0.12)"}
            strokeWidth={hovered ? 5 : 1.25}
          />
        );
      })}
      {bboxEntries.map(([square, bbox]) => {
        const rect = bboxRect(bbox);
        const hovered = square === hoveredSquare;
        const occupied = occupiedSquares.has(square);
        return (
          <rect
            key={`bbox-${square}`}
            x={rect.x}
            y={rect.y}
            width={rect.width}
            height={rect.height}
            fill={hovered ? "rgba(217, 70, 239, 0.16)" : "none"}
            stroke={
              hovered
                ? "rgba(217, 70, 239, 0.98)"
                : occupied
                  ? "rgba(250, 204, 21, 0.24)"
                  : "rgba(250, 204, 21, 0.08)"
            }
            strokeWidth={hovered ? 6 : occupied ? 1.75 : 1}
          />
        );
      })}
    </svg>
  );
}

function HoveredSquarePreview({
  imageSrc,
  hoveredSquare,
  hoveredBbox,
}: {
  imageSrc: string | null;
  hoveredSquare: HoveredSquareState | null;
  hoveredBbox: PhysicalRuntimeGeometryBbox | null;
}) {
  const cropStyle = useMemo(
    () => (hoveredBbox ? projectedCropStyle(hoveredBbox) : null),
    [hoveredBbox],
  );
  const hasPreview = Boolean(imageSrc && hoveredSquare && hoveredBbox && cropStyle);

  return (
    <div className="grid min-h-[108px] grid-cols-[88px,1fr] gap-2 rounded border p-2">
      <div className="relative aspect-square overflow-hidden rounded-sm bg-muted/20 ring-1 ring-inset ring-black/10 dark:ring-white/10">
        {hasPreview ? (
          <img
            src={imageSrc!}
            alt={`${hoveredSquare!.square} projected piece crop`}
            className="absolute max-w-none select-none"
            style={cropStyle!}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-[10px] uppercase tracking-wide text-muted-foreground/60">
            empty
          </div>
        )}
      </div>
      <div className="space-y-1">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Hovered square
        </div>
        <div className="min-h-5 text-sm font-medium">
          {hoveredSquare ? `${hoveredSquare.boardTitle} · ${hoveredSquare.square}` : "\u00a0"}
        </div>
        <div className="min-h-4 text-xs text-muted-foreground">
          {hoveredSquare
            ? pieceLabel(pieceSymbolAtSquare(hoveredSquare.fen, hoveredSquare.square))
            : "\u00a0"}
        </div>
        <div className="text-xs text-muted-foreground">
          {imageSrc
            ? hasPreview
              ? "Projected piece-box crop used for board-probe token pooling."
              : "Hover any GT/single/temporal square to inspect the projected piece-box crop."
            : "Runtime geometry preview unavailable."}
        </div>
      </div>
    </div>
  );
}

function BoardPanel({
  title,
  subtitle,
  fen,
  squareStyles,
  onSquareHover,
}: {
  title: string;
  subtitle: string;
  fen?: string;
  squareStyles?: Record<string, PhysicalBoardSquareStyle>;
  onSquareHover?: (square: string | null) => void;
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
        <ChessBoard
          fen={fen}
          size={192}
          squareStyles={squareStyles}
          onSquareHover={onSquareHover}
        />
      ) : (
        <div className="flex aspect-square items-center justify-center rounded border bg-muted/30 text-sm text-muted-foreground">
          Board unavailable
        </div>
      )}
    </div>
  );
}

function ProbabilityDistribution({
  entry,
}: {
  entry: SquareEvidenceEntry | null;
}) {
  if (!entry) return null;

  if (entry.probabilities.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        Raw stateless softmax probabilities are unavailable for this frame.
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {entry.probabilities.map(({ className, probability }) => {
        const tags: string[] = [];
        if (className === entry.gtClass) tags.push("GT");
        if (className === entry.statelessClass) tags.push("stateless");
        if (className === entry.temporalClass) tags.push("temporal");
        return (
          <div
            key={`${entry.square}-${className}`}
            className="grid grid-cols-[96px_minmax(0,1fr)_52px_auto] items-center gap-2"
          >
            <div className="truncate text-xs text-muted-foreground">
              {SQUARE_CLASS_LABELS[className] ?? className}
            </div>
            <div className="h-2 overflow-hidden rounded bg-muted">
              <div
                className="h-full rounded bg-foreground/80 transition-[width] duration-150"
                style={{ width: `${Math.max(probability, 0) * 100}%` }}
              />
            </div>
            <div className="text-right font-mono text-xs tabular-nums">
              {(probability * 100).toFixed(1)}%
            </div>
            <div className="flex min-h-5 flex-wrap justify-end gap-1">
              {tags.map((tag) => (
                <span
                  key={`${entry.square}-${className}-${tag}`}
                  className="rounded border px-1 py-0 text-[10px] text-muted-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function PhysicalRuntimeCard({
  result,
  pinned,
  sessionId,
  onPin,
}: PhysicalRuntimeCardProps) {
  const [hoveredSquare, setHoveredSquare] = useState<HoveredSquareState | null>(null);
  const [selectedEvidenceSquare, setSelectedEvidenceSquare] = useState<string>(() =>
    defaultEvidenceSquare(result),
  );
  const [showSquareEvidence, setShowSquareEvidence] = useState(Boolean(pinned));

  useEffect(() => {
    setSelectedEvidenceSquare(defaultEvidenceSquare(result));
  }, [result.annotation_id]);

  useEffect(() => {
    if (pinned) setShowSquareEvidence(true);
  }, [pinned]);

  const imageSrc = detailImageSrc(result, sessionId);
  const clipLabel = result.clip_filename || result.annotation_id;
  const recoveredSquares = result.stateless_error_count - result.temporal_error_count;
  const geometrySquareQuads = result.geometry_square_quads ?? {};
  const geometryPieceBboxes = result.geometry_piece_bboxes ?? {};
  const hoveredBbox =
    hoveredSquare != null ? geometryPieceBboxes[hoveredSquare.square] ?? null : null;
  const occupiedSquares = useMemo(() => {
    const occupied = new Set<string>();
    for (const square of Object.keys(geometryPieceBboxes)) {
      if (pieceSymbolAtSquare(result.gt_fen, square)) occupied.add(square);
    }
    return occupied;
  }, [geometryPieceBboxes, result.gt_fen]);
  const boardViewportBounds = useMemo(() => {
    const bounds = geometryBounds(geometrySquareQuads, geometryPieceBboxes);
    return bounds ? expandedBounds(bounds) : null;
  }, [geometryPieceBboxes, geometrySquareQuads]);
  const boardViewportStyle = useMemo(
    () => (boardViewportBounds ? focusedViewportStyle(boardViewportBounds) : null),
    [boardViewportBounds],
  );
  const imageAspectRatio = useMemo(() => {
    if (
      boardViewportBounds &&
      result.image_width != null &&
      result.image_height != null &&
      result.image_width > 0 &&
      result.image_height > 0
    ) {
      const [xmin, ymin, xmax, ymax] = boardViewportBounds;
      const cropWidth = Math.max((xmax - xmin) * result.image_width, 1);
      const cropHeight = Math.max((ymax - ymin) * result.image_height, 1);
      return `${cropWidth} / ${cropHeight}`;
    }
    if (result.image_width && result.image_height) {
      return `${result.image_width} / ${result.image_height}`;
    }
    return "16 / 9";
  }, [boardViewportBounds, result.image_height, result.image_width]);
  const squareEvidenceEntries = useMemo<SquareEvidenceEntry[]>(() => {
    return BOARD_SQUARES.map((square) => {
      const index = squareIndex(square);
      const probabilities =
        index == null
          ? []
          : probabilityRows(result.stateless_square_probabilities?.[index]);
      const top1 = probabilities[0] ?? null;
      return {
        square,
        bbox: geometryPieceBboxes[square] ?? null,
        gtClass: boardClassAtSquare(result.gt_fen, square),
        statelessClass: boardClassAtSquare(result.stateless_fen, square),
        temporalClass: boardClassAtSquare(result.temporal_fen, square),
        top1Class: top1?.className ?? null,
        top1Probability: top1?.probability ?? null,
        probabilities,
      };
    });
  }, [
    geometryPieceBboxes,
    result.gt_fen,
    result.stateless_fen,
    result.stateless_square_probabilities,
    result.temporal_fen,
  ]);
  const activeEvidenceSquare = hoveredSquare?.square ?? selectedEvidenceSquare;
  const activeEvidenceEntry = useMemo(
    () =>
      squareEvidenceEntries.find((entry) => entry.square === activeEvidenceSquare) ??
      squareEvidenceEntries[0] ??
      null,
    [activeEvidenceSquare, squareEvidenceEntries],
  );

  function handleBoardSquareHover(boardTitle: string, fen?: string) {
    return (square: string | null) => {
      setHoveredSquare(square ? { boardTitle, square, fen } : null);
    };
  }

  function handleEvidenceSquareHover(square: string | null) {
    setHoveredSquare(
      square ? { boardTitle: "Square evidence", square, fen: result.stateless_fen } : null,
    );
  }

  return (
    <div className="border rounded-lg p-3 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            {onPin && (
              <button
                onClick={onPin}
                title={pinned ? "Unpin from top" : "Pin to top"}
                className={`relative flex-shrink-0 w-5 h-5 flex items-center justify-center rounded transition-[color,transform] active:scale-[0.96] before:absolute before:inset-y-[-10px] before:-left-[4px] before:-right-[6px] before:content-[''] ${
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
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mt-0.5 tabular-nums">
            <span>{result.elapsed_ms.toFixed(1)}ms</span>
            <span>{deltaLabel(recoveredSquares)}</span>
            <span>single {result.stateless_error_count} wrong</span>
            <span>temp {result.temporal_error_count} wrong</span>
          </div>
        </div>
        <div className="text-right shrink-0">
          <span
            className={`text-sm font-bold tabular-nums ${
              result.temporal_exact_match ? "text-green-600" : "text-red-600"
            }`}
          >
            {result.temporal_exact_match
              ? "✓ Exact"
              : `✗ ${result.temporal_error_count} wrong`}
          </span>
          <div className="text-xs text-muted-foreground mt-0.5 tabular-nums">
            temp {formatPercent(result.temporal_square_accuracy)} · single {formatPercent(result.stateless_square_accuracy)}
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-5">
        <div className="space-y-2 2xl:col-span-2">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Runtime input
            </p>
            <p className="text-xs text-muted-foreground">
              Board-focused view with padding; yellow projections stay visible and hovered square is fuchsia
            </p>
          </div>
          {imageSrc ? (
            <div className="space-y-2">
              <div
                className="relative w-full overflow-hidden rounded border bg-muted/20"
                style={{ aspectRatio: imageAspectRatio }}
              >
                {boardViewportStyle ? (
                  <div className="absolute inset-0 overflow-hidden">
                    <div className="absolute" style={boardViewportStyle}>
                      <img
                        src={imageSrc}
                        alt={`${clipLabel} board-focused source frame ${result.frame_index}`}
                        className="absolute inset-0 h-full w-full object-fill"
                        loading="lazy"
                      />
                      <GeometryOverlay
                        geometrySquareQuads={geometrySquareQuads}
                        geometryPieceBboxes={geometryPieceBboxes}
                        occupiedSquares={occupiedSquares}
                        hoveredSquare={hoveredSquare?.square ?? null}
                      />
                    </div>
                  </div>
                ) : (
                  <>
                    <img
                      src={imageSrc}
                      alt={`${clipLabel} source frame ${result.frame_index}`}
                      className="absolute inset-0 h-full w-full object-fill"
                      loading="lazy"
                    />
                    <GeometryOverlay
                      geometrySquareQuads={geometrySquareQuads}
                      geometryPieceBboxes={geometryPieceBboxes}
                      occupiedSquares={occupiedSquares}
                      hoveredSquare={hoveredSquare?.square ?? null}
                    />
                  </>
                )}
              </div>
              <HoveredSquarePreview
                imageSrc={imageSrc}
                hoveredSquare={hoveredSquare}
                hoveredBbox={hoveredBbox}
              />
            </div>
          ) : (
            <div className="flex aspect-video items-center justify-center rounded border bg-muted/30 text-sm text-muted-foreground">
              Image unavailable
            </div>
          )}
        </div>

        <BoardPanel
          title="Ground truth"
          subtitle="Yellow border = GT changed · hover squares to inspect geometry"
          fen={result.gt_fen}
          squareStyles={groundTruthSquareStyles(result.gt_changed_squares)}
          onSquareHover={handleBoardSquareHover("Ground truth", result.gt_fen)}
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
          onSquareHover={handleBoardSquareHover("Stateless", result.stateless_fen)}
        />
        <BoardPanel
          title="Decoded (legal)"
          subtitle={`${result.temporal_error_count} wrong · ${formatPercent(result.temporal_square_accuracy)}`}
          fen={result.temporal_fen}
          squareStyles={predictionSquareStyles(
            result.temporal_error_squares,
            result.temporal_changed_squares,
            result.temporal_square_confidences,
          )}
          onSquareHover={handleBoardSquareHover("Temporal", result.temporal_fen)}
        />
      </div>

      <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2 xl:grid-cols-4">
        <div>Decoded exact: {result.temporal_exact_match ? "yes" : "no"}</div>
        <div>Stateless exact: {result.stateless_exact_match ? "yes" : "no"}</div>
        <div>
          Decoded transition: {formatTransition(result.temporal_transition_kind, result.temporal_transition_label)}
        </div>
        <div>
          Raw single-frame transition: {formatTransition(result.stateless_transition_kind, result.stateless_transition_label)}
        </div>
        <div>
          Annotation: <code>{result.annotation_id}</code>
        </div>
        <div>Geometry: client overlay on raw source frame + piece-box pooling</div>
      </div>

      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setShowSquareEvidence((current) => !current)}
          className="rounded border px-3 py-1.5 text-xs text-muted-foreground transition-[color,transform] hover:text-foreground active:scale-[0.96]"
        >
          {showSquareEvidence ? "Hide" : "Show"} raw square evidence
        </button>
      </div>

      {showSquareEvidence ? (
        <div className="rounded-md border p-3 space-y-3">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Raw square evidence
            </p>
            <p className="text-xs text-muted-foreground">
              These are projected piece-box crops from the raw source frame. The board probe pools
              patch tokens inside each box; it does not classify 64 independent square PNGs.
              Probabilities below are softmaxes from the raw per-frame stateless logits before legal
              decoding.
            </p>
          </div>

          <div className="grid grid-cols-8 gap-1.5">
            {squareEvidenceEntries.map((entry) => {
              const cropStyle = entry.bbox ? projectedCropStyle(entry.bbox) : null;
              const active = entry.square === activeEvidenceSquare;
              const wrong = entry.statelessClass !== entry.gtClass;
              const topLabel = compactClassLabel(entry.top1Class);
              const topProb =
                entry.top1Probability == null
                  ? "-"
                  : `${Math.round(entry.top1Probability * 100)}%`;

              return (
                <button
                  key={entry.square}
                  type="button"
                  onClick={() => setSelectedEvidenceSquare(entry.square)}
                  onMouseEnter={() => handleEvidenceSquareHover(entry.square)}
                  onMouseLeave={() => handleEvidenceSquareHover(null)}
                  className={`rounded-md border p-1 text-left transition-[color,background-color,border-color,transform] active:scale-[0.96] ${
                    active
                      ? "border-foreground bg-muted/20"
                      : wrong
                        ? "border-red-300/70 bg-red-500/5 hover:bg-red-500/10"
                        : "border-border hover:bg-muted/20"
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between font-mono text-[10px] tabular-nums">
                    <span>{entry.square}</span>
                    <span>{topProb}</span>
                  </div>
                  <div className="relative aspect-square overflow-hidden rounded-sm bg-muted/20 ring-1 ring-inset ring-black/10 dark:ring-white/10">
                    {imageSrc && entry.bbox && cropStyle ? (
                      <img
                        src={imageSrc}
                        alt={`${entry.square} projected piece crop`}
                        className="absolute max-w-none select-none"
                        style={cropStyle}
                        loading="lazy"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-[10px] uppercase tracking-wide text-muted-foreground/60">
                        empty
                      </div>
                    )}
                  </div>
                  <div className="mt-1 truncate text-center font-mono text-[10px] text-muted-foreground">
                    {topLabel}
                  </div>
                </button>
              );
            })}
          </div>

          {activeEvidenceEntry ? (
            <div className="space-y-3 rounded border p-3">
              <div className="grid gap-3 sm:grid-cols-[96px_minmax(0,1fr)]">
                <div className="relative aspect-square overflow-hidden rounded-sm bg-muted/20 ring-1 ring-inset ring-black/10 dark:ring-white/10">
                  {imageSrc && activeEvidenceEntry.bbox ? (
                    <img
                      src={imageSrc}
                      alt={`${activeEvidenceEntry.square} projected piece crop`}
                      className="absolute max-w-none select-none"
                      style={projectedCropStyle(activeEvidenceEntry.bbox) ?? undefined}
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-[10px] uppercase tracking-wide text-muted-foreground/60">
                      empty
                    </div>
                  )}
                </div>
                <div className="space-y-1">
                  <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Selected square
                  </div>
                  <div className="text-sm font-medium font-mono">
                    {activeEvidenceEntry.square}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    GT {pieceLabel(pieceSymbolAtSquare(result.gt_fen, activeEvidenceEntry.square))}
                    {" · "}
                    stateless {pieceLabel(pieceSymbolAtSquare(result.stateless_fen, activeEvidenceEntry.square))}
                    {" · "}
                    temporal {pieceLabel(pieceSymbolAtSquare(result.temporal_fen, activeEvidenceEntry.square))}
                  </div>
                  <div className="text-xs text-muted-foreground tabular-nums">
                    top-1 {pieceLabel(activeEvidenceEntry.top1Class)}
                    {activeEvidenceEntry.top1Probability != null
                      ? ` · ${(activeEvidenceEntry.top1Probability * 100).toFixed(1)}%`
                      : ""}
                  </div>
                </div>
              </div>

              <ProbabilityDistribution entry={activeEvidenceEntry} />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
