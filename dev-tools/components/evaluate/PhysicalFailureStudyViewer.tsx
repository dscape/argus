"use client";

import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { ChessBoard } from "@/components/ChessBoard";
import {
  GT_CHANGED_BORDER,
  PRED_CHANGED_BORDER,
  groundTruthSquareStyles,
  predictionSquareStyles,
  type PhysicalBoardSquareStyle,
} from "@/components/evaluate/physicalBoardStyles";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  exportPhysicalFailureStudyCsvUrl,
  getPhysicalFailureStudy,
  listPhysicalFailureStudies,
  physicalFailureStudyImageUrl,
  updatePhysicalFailureStudyEntry,
  type PhysicalFailureStudyDetail,
  type PhysicalFailureStudyEntry,
  type PhysicalFailureStudyFrame,
  type PhysicalFailureStudyListItem,
  type PhysicalRuntimeGeometryBbox,
  type PhysicalRuntimeGeometryQuad,
} from "@/lib/api";

const UNTAGGED_LABEL = "(untagged)";
const RECTIFICATION_BUCKET = "rectification / localization";
const CLASSIFIER_BUCKET = "piece classifier / square evidence";
const TEMPORAL_BUCKET = "temporal in-between / move execution ambiguity";
const DECODER_BUCKET = "decoder / wrong legal hypothesis / error propagation";
const EVAL_BUCKET = "eval or label issue";
const OTHER_BUCKET = "other / unclear";

const BUCKET_DESCRIPTIONS: Record<string, string> = {
  [RECTIFICATION_BUCKET]:
    "Crop or grid is visibly wrong: warped, shifted, clipped, or localizer drift.",
  [CLASSIFIER_BUCKET]:
    "Board geometry looks usable, but the square evidence itself is wrong.",
  [TEMPORAL_BUCKET]:
    "The frame is mid-move or one frame off the move boundary; the decoder was early or late.",
  [DECODER_BUCKET]:
    "Evidence was usable, but the decoder chose the wrong legal hypothesis or propagated a prior miss.",
  [EVAL_BUCKET]: "Ground truth looks wrong, missing, or fundamentally ambiguous.",
  [OTHER_BUCKET]: "Does not fit the buckets above; explain the reason in notes.",
};

const VISUAL_PANEL_SIZE = 216;
const SQUARE_CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];
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

interface HoveredSquareState {
  boardTitle: string;
  square: string;
  fen?: string | null;
}

interface SquareEvidenceEntry {
  square: string;
  bbox: PhysicalRuntimeGeometryBbox | null;
  gtClass: string;
  statelessClass: string;
  decodedClass: string;
  top1Class: string | null;
  top1Probability: number | null;
  probabilities: Array<{ className: string; probability: number }>;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatFrameIndex(value: number): string {
  return `f${value.toString().padStart(4, "0")}`;
}

function boardAccuracyLabel(errorCount: number | null | undefined): string {
  if (errorCount == null || !Number.isFinite(errorCount)) return "-";
  return formatPercent((64 - errorCount) / 64);
}

function statelessErrorSquares(frame: PhysicalFailureStudyFrame): string[] {
  if ((frame.stateless_error_squares ?? []).length > 0) {
    return frame.stateless_error_squares ?? [];
  }
  return (frame.square_diagnostics ?? [])
    .filter((square) => square.gt_class !== square.stateless_class)
    .map((square) => square.square);
}

function decodedErrorSquares(frame: PhysicalFailureStudyFrame): string[] {
  if ((frame.decoded_error_squares ?? []).length > 0) {
    return frame.decoded_error_squares ?? [];
  }
  return (frame.square_diagnostics ?? [])
    .filter((square) => square.gt_class !== square.decoded_class)
    .map((square) => square.square);
}

function processedCropImagePath(frame: PhysicalFailureStudyFrame): string | null {
  const path = frame.processed_image_path?.trim();
  return path ? path : null;
}

function rawSnapshotImagePath(frame: PhysicalFailureStudyFrame): string | null {
  const path = frame.raw_image_path?.trim();
  return path ? path : null;
}

function thumbnailImagePath(frame: PhysicalFailureStudyFrame): string {
  return rawSnapshotImagePath(frame) ?? processedCropImagePath(frame) ?? frame.image_path;
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

function pieceSymbolAtSquare(fen: string | undefined | null, square: string): string | null {
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

function boardClassAtSquare(fen: string | undefined | null, square: string): string {
  return pieceSymbolAtSquare(fen, square) ?? "empty";
}

function compactClassLabel(className: string | null): string {
  if (className == null || className === "empty") return "∅";
  return className;
}

function probabilityRowsForSquare(probabilities: number[] | undefined): Array<{
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

function defaultEvidenceSquare(frame: PhysicalFailureStudyFrame): string {
  return (
    frame.stateless_error_squares?.[0] ??
    frame.decoded_error_squares?.[0] ??
    frame.gt_changed_squares?.[0] ??
    BOARD_SQUARES[0]!
  );
}

function trackerInputDescription(temporalMode: string | undefined): string {
  if (temporalMode === "metadata") return "a metadata-smoothed tracker input";
  if (temporalMode === "fixed") return "an EMA-smoothed tracker input";
  return "the raw per-frame logits";
}

function bucketLabel(entry: PhysicalFailureStudyEntry): string {
  return entry.final_bucket?.trim() || UNTAGGED_LABEL;
}

function detailsText(entry: PhysicalFailureStudyEntry): string {
  if (entry.notes?.trim()) return entry.notes.trim();
  if (entry.suggested_bucket?.trim()) {
    return `suggested: ${entry.suggested_bucket.trim()}`;
  }
  return "No notes";
}

function badgeVariantForBucket(
  bucket: string,
): "default" | "secondary" | "destructive" | "outline" {
  if (bucket === UNTAGGED_LABEL) return "outline";
  if (bucket.includes("eval") || bucket.includes("other")) return "secondary";
  if (bucket.includes("decoder")) return "destructive";
  return "default";
}

function matchesEntry(
  entry: PhysicalFailureStudyEntry,
  query: string,
  bucketFilter: string,
): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  const currentBucket = bucketLabel(entry);
  if (bucketFilter !== "all" && currentBucket !== bucketFilter) {
    return false;
  }
  if (!normalizedQuery) return true;

  const haystack = [
    entry.selected_index,
    entry.episode_id,
    entry.clip_filename,
    entry.source_video_id,
    entry.first_frame_index,
    entry.length,
    entry.suggested_bucket,
    entry.final_bucket,
    entry.notes,
    entry.failing_frame?.decoded_move_uci,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return haystack.includes(normalizedQuery);
}

function classificationHint(
  entry: PhysicalFailureStudyEntry | null,
): { bucket: string; title: string; body: string } | null {
  if (!entry) return null;

  const frame = entry.failing_frame;
  const legal = frame.legal_from_previous_decoded;
  const tailNote =
    entry.length > 1
      ? `Tag the first FAIL frame, not the ${entry.length}-frame desync tail that follows it.`
      : "This is a single-frame failure episode.";

  if (frame.decoded_matches_previous_gt) {
    return {
      bucket: TEMPORAL_BUCKET,
      title: "Likely transition-frame miss",
      body:
        "Decoded still matches the previous GT on the first failing frame. " +
        `That usually means the model stayed one frame too early during move execution. ${tailNote}`,
    };
  }

  if (frame.decoded_matches_next_gt) {
    return {
      bucket: TEMPORAL_BUCKET,
      title: "Likely late commit on an in-between frame",
      body:
        "Decoded already matches the next GT on the first failing frame. " +
        `That usually means the model committed one frame too late. ${tailNote}`,
    };
  }

  if (legal?.best_legal_matches_gt) {
    return {
      bucket: DECODER_BUCKET,
      title: "Likely decoder hypothesis mistake",
      body:
        `From the previous decoded board, the best legal continuation ` +
        `(${legal.best_legal_move_uci ?? "a legal move"}) already matches GT. ` +
        `Evidence existed, but the decoder chose the wrong legal path. ${tailNote}`,
    };
  }

  if ((frame.stateless_error_count ?? 0) > 0) {
    return {
      bucket: CLASSIFIER_BUCKET,
      title: "Likely square-evidence issue",
      body:
        `The stateless board is already wrong on the first failing frame ` +
        `(${frame.stateless_error_count} squares), so the problem appears before ` +
        `temporal decoding. ${tailNote}`,
    };
  }

  return {
    bucket: entry.suggested_bucket || OTHER_BUCKET,
    title: "Heuristic fallback",
    body:
      `Suggested bucket is ${entry.suggested_bucket || OTHER_BUCKET}. ` +
      `${tailNote}`,
  };
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="mt-2 text-2xl font-semibold">{value}</div>
        {hint ? <div className="mt-1 text-xs text-muted-foreground">{hint}</div> : null}
      </CardContent>
    </Card>
  );
}

function ImagePanel({
  title,
  subtitle,
  studyPath,
  imagePath,
  alt,
  emptyLabel,
}: {
  title: string;
  subtitle: string;
  studyPath: string;
  imagePath?: string | null;
  alt: string;
  emptyLabel: string;
}) {
  return (
    <div className="space-y-2 rounded-lg border bg-muted/10 p-3">
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </div>
        <div className="text-xs text-muted-foreground">{subtitle}</div>
      </div>
      {imagePath ? (
        <div className="flex justify-center rounded border bg-muted/20 p-1">
          <img
            src={physicalFailureStudyImageUrl(studyPath, imagePath)}
            alt={alt}
            className="object-contain"
            style={{ width: VISUAL_PANEL_SIZE, height: VISUAL_PANEL_SIZE }}
          />
        </div>
      ) : (
        <div
          className="mx-auto flex items-center justify-center rounded border bg-muted/30 px-4 text-center text-sm text-muted-foreground"
          style={{ width: VISUAL_PANEL_SIZE, height: VISUAL_PANEL_SIZE }}
        >
          {emptyLabel}
        </div>
      )}
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
  fen?: string | null;
  squareStyles?: Record<string, PhysicalBoardSquareStyle>;
  onSquareHover?: (square: string | null) => void;
}) {
  return (
    <div className="space-y-2 rounded-lg border bg-muted/10 p-3">
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </div>
        <div className="text-xs text-muted-foreground">{subtitle}</div>
      </div>
      {fen ? (
        <div className="flex justify-center overflow-x-auto">
          <ChessBoard
            fen={fen}
            size={VISUAL_PANEL_SIZE}
            squareStyles={squareStyles}
            onSquareHover={onSquareHover}
          />
        </div>
      ) : (
        <div
          className="mx-auto flex items-center justify-center rounded border bg-muted/30 text-sm text-muted-foreground"
          style={{ width: VISUAL_PANEL_SIZE, height: VISUAL_PANEL_SIZE }}
        >
          Board unavailable
        </div>
      )}
    </div>
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
  const squareEntries = Object.entries(geometrySquareQuads);
  const bboxEntries = Object.entries(geometryPieceBboxes);
  if (squareEntries.length === 0 && bboxEntries.length === 0) return null;

  return (
    <svg
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {squareEntries.map(([square, quad]) => {
        const hovered = square === hoveredSquare;
        return (
          <polygon
            key={`quad-${square}`}
            points={quadPoints(quad)}
            fill={hovered ? "rgba(217, 70, 239, 0.12)" : "none"}
            stroke={hovered ? "rgba(217, 70, 239, 0.95)" : "rgba(250, 204, 21, 0.12)"}
            strokeWidth={hovered ? 5 : 1.25}
          />
        );
      })}
      {bboxEntries.map(([square, bbox]) => {
        const hovered = square === hoveredSquare;
        const occupied = occupiedSquares.has(square);
        const rect = bboxRect(bbox);
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

function ProbabilityDistribution({
  frame,
  hoveredSquare,
}: {
  frame: PhysicalFailureStudyFrame;
  hoveredSquare: HoveredSquareState | null;
}) {
  const squareProbabilityRows = useMemo(() => {
    if (!hoveredSquare) return [];
    const index = squareIndex(hoveredSquare.square);
    if (index == null) return [];
    const squareProbabilities = frame.stateless_square_probabilities?.[index] ?? [];
    return squareProbabilities
      .map((probability, classIndex) => ({
        className: SQUARE_CLASS_NAMES[classIndex] ?? `cls${classIndex}`,
        probability,
      }))
      .sort((left, right) => right.probability - left.probability);
  }, [frame.stateless_square_probabilities, hoveredSquare]);

  const gtClass = hoveredSquare ? boardClassAtSquare(frame.gt_fen, hoveredSquare.square) : null;
  const statelessClass = hoveredSquare
    ? boardClassAtSquare(frame.stateless_fen, hoveredSquare.square)
    : null;
  const decodedClass = hoveredSquare ? boardClassAtSquare(frame.decoded_fen, hoveredSquare.square) : null;

  return (
    <div className="space-y-2 rounded-md border p-3">
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Stateless square distribution
        </div>
        <div className="text-xs text-muted-foreground">
          Softmax probabilities from the raw per-frame stateless logits before legal decoding.
        </div>
      </div>
      {!hoveredSquare ? (
        <div className="text-sm text-muted-foreground">
          Hover any GT/stateless/decoded square to inspect the raw square distribution.
        </div>
      ) : squareProbabilityRows.length === 0 ? (
        <div className="text-sm text-muted-foreground">
          This bundle does not include per-class stateless probabilities yet. Regenerate it to inspect logits.
        </div>
      ) : (
        <div className="space-y-1.5">
          {squareProbabilityRows.map(({ className, probability }) => {
            const badges: string[] = [];
            if (className === gtClass) badges.push("GT");
            if (className === statelessClass) badges.push("stateless");
            if (className === decodedClass) badges.push("decoded");
            return (
              <div key={className} className="grid grid-cols-[120px_minmax(0,1fr)_56px_auto] items-center gap-2">
                <div className="truncate text-xs text-muted-foreground">
                  {SQUARE_CLASS_LABELS[className] ?? className}
                </div>
                <div className="h-2 overflow-hidden rounded bg-muted">
                  <div
                    className="h-full rounded bg-foreground/80 transition-[width] duration-150"
                    style={{ width: `${Math.max(probability, 0) * 100}%` }}
                  />
                </div>
                <div className="text-right font-mono text-xs tabular-nums">{(probability * 100).toFixed(1)}%</div>
                <div className="flex min-h-5 flex-wrap justify-end gap-1">
                  {badges.map((badge) => (
                    <Badge key={`${className}-${badge}`} variant="outline" className="px-1 py-0 text-[10px]">
                      {badge}
                    </Badge>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function HoverInspector({
  studyPath,
  frame,
  hoveredSquare,
}: {
  studyPath: string;
  frame: PhysicalFailureStudyFrame;
  hoveredSquare: HoveredSquareState | null;
}) {
  const geometrySquareQuads = frame.geometry_square_quads ?? {};
  const geometryPieceBboxes = frame.geometry_piece_bboxes ?? {};
  const hoveredBbox = hoveredSquare ? geometryPieceBboxes[hoveredSquare.square] ?? null : null;
  const imagePath = rawSnapshotImagePath(frame);
  const imageSrc = imagePath ? physicalFailureStudyImageUrl(studyPath, imagePath) : null;
  const occupiedSquares = useMemo(() => {
    const fen = hoveredSquare?.fen ?? frame.gt_fen;
    return new Set(
      Object.keys(geometryPieceBboxes).filter((square) => pieceSymbolAtSquare(fen, square)),
    );
  }, [frame.gt_fen, geometryPieceBboxes, hoveredSquare?.fen]);
  const cropStyle = useMemo(
    () => (hoveredBbox ? projectedCropStyle(hoveredBbox) : null),
    [hoveredBbox],
  );
  const hasPreview = Boolean(imageSrc && hoveredSquare && hoveredBbox && cropStyle);

  return (
    <div className="rounded-lg border bg-muted/10 p-3 xl:col-span-2 2xl:col-span-5">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
        <div className="space-y-2">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Hover / geometry inspector
            </div>
            <div className="text-xs text-muted-foreground">
              Hover any GT/stateless/decoded square to inspect the projected piece box on the raw source frame.
            </div>
          </div>
          {imageSrc ? (
            <div className="relative mx-auto max-w-[520px] overflow-hidden rounded border bg-muted/20">
              <img src={imageSrc} alt="Raw source frame with hover geometry" className="block h-auto w-full" />
              <GeometryOverlay
                geometrySquareQuads={geometrySquareQuads}
                geometryPieceBboxes={geometryPieceBboxes}
                occupiedSquares={occupiedSquares}
                hoveredSquare={hoveredSquare?.square ?? null}
              />
            </div>
          ) : (
            <div className="flex min-h-[220px] items-center justify-center rounded border bg-muted/30 px-4 text-center text-sm text-muted-foreground">
              Raw source frame unavailable for this annotation.
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="grid min-h-[108px] grid-cols-[88px,1fr] gap-2 rounded-md border p-3">
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
                {hoveredSquare ? pieceLabel(pieceSymbolAtSquare(hoveredSquare.fen, hoveredSquare.square)) : "\u00a0"}
              </div>
              <div className="text-xs text-muted-foreground">
                {imageSrc
                  ? hasPreview
                    ? "Projected piece-box crop used for board-probe token pooling."
                    : "Hover any GT/stateless/decoded square to inspect the projected piece-box crop."
                  : "Source-frame preview unavailable."}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

function SquareEvidencePanel({
  studyPath,
  frame,
  hoveredSquare,
  onHoverSquare,
}: {
  studyPath: string;
  frame: PhysicalFailureStudyFrame;
  hoveredSquare: HoveredSquareState | null;
  onHoverSquare: (hoveredSquare: HoveredSquareState | null) => void;
}) {
  const [selectedSquare, setSelectedSquare] = useState<string>(() => defaultEvidenceSquare(frame));

  useEffect(() => {
    setSelectedSquare(defaultEvidenceSquare(frame));
  }, [frame.annotation_id, frame.frame_index]);

  const imagePath = rawSnapshotImagePath(frame);
  const imageSrc = imagePath ? physicalFailureStudyImageUrl(studyPath, imagePath) : null;
  const geometryPieceBboxes = frame.geometry_piece_bboxes ?? {};
  const squareEvidenceEntries = useMemo<SquareEvidenceEntry[]>(() => {
    return BOARD_SQUARES.map((square) => {
      const index = squareIndex(square);
      const probabilities = probabilityRowsForSquare(
        index == null ? undefined : frame.stateless_square_probabilities?.[index],
      );
      const top1 = probabilities[0] ?? null;
      return {
        square,
        bbox: geometryPieceBboxes[square] ?? null,
        gtClass: boardClassAtSquare(frame.gt_fen, square),
        statelessClass: boardClassAtSquare(frame.stateless_fen, square),
        decodedClass: boardClassAtSquare(frame.decoded_fen, square),
        top1Class: top1?.className ?? null,
        top1Probability: top1?.probability ?? null,
        probabilities,
      };
    });
  }, [
    frame.decoded_fen,
    frame.geometry_piece_bboxes,
    frame.gt_fen,
    frame.stateless_fen,
    frame.stateless_square_probabilities,
    geometryPieceBboxes,
  ]);
  const activeSquareState = useMemo<HoveredSquareState | null>(() => {
    const square = hoveredSquare?.square ?? selectedSquare;
    if (!square) return null;
    return {
      boardTitle: hoveredSquare?.boardTitle ?? "Square evidence",
      square,
      fen: hoveredSquare?.fen ?? frame.stateless_fen,
    };
  }, [frame.stateless_fen, hoveredSquare, selectedSquare]);
  const activeEntry = useMemo(
    () =>
      squareEvidenceEntries.find((entry) => entry.square === activeSquareState?.square) ??
      squareEvidenceEntries[0] ??
      null,
    [activeSquareState?.square, squareEvidenceEntries],
  );
  const activeCropStyle = useMemo(
    () => (activeEntry?.bbox ? projectedCropStyle(activeEntry.bbox) : null),
    [activeEntry?.bbox],
  );

  return (
    <div className="rounded-lg border bg-muted/10 p-3 xl:col-span-2 2xl:col-span-5">
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Raw square evidence
        </div>
        <div className="text-xs text-muted-foreground">
          These are projected piece-box crops from the raw source frame. The board probe pools patch
          tokens inside each box; it does not classify 64 independent square PNGs. Probabilities below
          are softmaxes from the raw per-frame stateless logits before legal decoding.
        </div>
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
        <div className="grid grid-cols-8 gap-1.5">
          {squareEvidenceEntries.map((entry) => {
            const cropStyle = entry.bbox ? projectedCropStyle(entry.bbox) : null;
            const active = entry.square === activeSquareState?.square;
            const wrong = entry.statelessClass !== entry.gtClass;
            const topLabel = compactClassLabel(entry.top1Class);
            const topProb =
              entry.top1Probability == null ? "-" : `${Math.round(entry.top1Probability * 100)}%`;

            return (
              <button
                key={entry.square}
                type="button"
                onClick={() => setSelectedSquare(entry.square)}
                onMouseEnter={() =>
                  onHoverSquare({
                    boardTitle: "Square evidence",
                    square: entry.square,
                    fen: frame.stateless_fen,
                  })
                }
                onMouseLeave={() => onHoverSquare(null)}
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

        <div className="space-y-3 rounded-md border p-3">
          {activeEntry ? (
            <>
              <div className="grid gap-3 sm:grid-cols-[96px_minmax(0,1fr)]">
                <div className="relative aspect-square overflow-hidden rounded-sm bg-muted/20 ring-1 ring-inset ring-black/10 dark:ring-white/10">
                  {imageSrc && activeEntry.bbox && activeCropStyle ? (
                    <img
                      src={imageSrc}
                      alt={`${activeEntry.square} projected piece crop`}
                      className="absolute max-w-none select-none"
                      style={activeCropStyle}
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
                  <div className="text-sm font-medium font-mono">{activeEntry.square}</div>
                  <div className="text-xs text-muted-foreground">
                    GT {pieceLabel(pieceSymbolAtSquare(frame.gt_fen, activeEntry.square))}
                    {" · "}
                    stateless {pieceLabel(pieceSymbolAtSquare(frame.stateless_fen, activeEntry.square))}
                    {" · "}
                    decoded {pieceLabel(pieceSymbolAtSquare(frame.decoded_fen, activeEntry.square))}
                  </div>
                  <div className="text-xs text-muted-foreground tabular-nums">
                    top-1 {pieceLabel(activeEntry.top1Class)}
                    {activeEntry.top1Probability != null
                      ? ` · ${(activeEntry.top1Probability * 100).toFixed(1)}%`
                      : ""}
                  </div>
                </div>
              </div>

              <ProbabilityDistribution frame={frame} hoveredSquare={activeSquareState} />
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function LegendItem({
  fill,
  border,
  label,
}: {
  fill?: string;
  border?: string;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-md border px-2 py-1 text-xs">
      <span
        className="inline-block h-4 w-4 rounded-sm border"
        style={{
          backgroundColor: fill ?? "transparent",
          borderColor: border ?? "rgba(0, 0, 0, 0.25)",
          borderWidth: border ? 2 : 1,
        }}
      />
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}

export default function PhysicalFailureStudyViewer() {
  const [studies, setStudies] = useState<PhysicalFailureStudyListItem[]>([]);
  const [selectedStudyPath, setSelectedStudyPath] = useState<string>("");
  const [study, setStudy] = useState<PhysicalFailureStudyDetail | null>(null);
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<string | null>(null);
  const [frameIndexInWindow, setFrameIndexInWindow] = useState(0);
  const [query, setQuery] = useState("");
  const [bucketFilter, setBucketFilter] = useState("all");
  const [draftBucket, setDraftBucket] = useState("");
  const [draftNotes, setDraftNotes] = useState("");
  const [loadingStudies, setLoadingStudies] = useState(true);
  const [loadingStudy, setLoadingStudy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hoveredSquare, setHoveredSquare] = useState<HoveredSquareState | null>(null);
  const notesRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadStudies() {
      try {
        setLoadingStudies(true);
        const response = await listPhysicalFailureStudies();
        if (cancelled) return;
        setStudies(response.studies);
        setSelectedStudyPath((current) => {
          if (current && response.studies.some((item) => item.path === current)) {
            return current;
          }
          return response.studies[0]?.path || "";
        });
      } catch (error) {
        if (cancelled) return;
        toast.error(
          error instanceof Error ? error.message : "Failed to load failure studies",
        );
      } finally {
        if (!cancelled) setLoadingStudies(false);
      }
    }
    void loadStudies();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedStudyPath) {
      setStudy(null);
      setSelectedEpisodeId(null);
      return;
    }
    let cancelled = false;
    async function loadStudy() {
      try {
        setLoadingStudy(true);
        const response = await getPhysicalFailureStudy(selectedStudyPath);
        if (cancelled) return;
        setStudy(response);
        setSelectedEpisodeId((current) => {
          if (current && response.entries.some((entry) => entry.episode_id === current)) {
            return current;
          }
          return response.entries[0]?.episode_id ?? null;
        });
      } catch (error) {
        if (cancelled) return;
        toast.error(error instanceof Error ? error.message : "Failed to load study");
        setStudy(null);
        setSelectedEpisodeId(null);
      } finally {
        if (!cancelled) setLoadingStudy(false);
      }
    }
    void loadStudy();
    return () => {
      cancelled = true;
    };
  }, [selectedStudyPath]);

  const bucketOptions = useMemo(() => {
    const base = new Set<string>(study?.bucket_options ?? []);
    for (const entry of study?.entries ?? []) {
      const value = entry.final_bucket?.trim();
      if (value) base.add(value);
    }
    return ["all", UNTAGGED_LABEL, ...Array.from(base)];
  }, [study]);

  const filteredEntries = useMemo(() => {
    return (study?.entries ?? []).filter((entry) => matchesEntry(entry, query, bucketFilter));
  }, [bucketFilter, query, study]);

  useEffect(() => {
    if (!filteredEntries.length) return;
    if (selectedEpisodeId == null) {
      setSelectedEpisodeId(filteredEntries[0]!.episode_id);
      return;
    }
    if (!filteredEntries.some((entry) => entry.episode_id === selectedEpisodeId)) {
      setSelectedEpisodeId(filteredEntries[0]!.episode_id);
    }
  }, [filteredEntries, selectedEpisodeId]);

  const selectedEntry = useMemo(() => {
    return study?.entries.find((entry) => entry.episode_id === selectedEpisodeId) ?? null;
  }, [selectedEpisodeId, study]);

  const frameWindow = useMemo<PhysicalFailureStudyFrame[]>(() => {
    if (!selectedEntry) return [];
    return [...selectedEntry.preceding_frames, selectedEntry.failing_frame];
  }, [selectedEntry]);

  useEffect(() => {
    if (!frameWindow.length) {
      setFrameIndexInWindow(0);
      return;
    }
    setFrameIndexInWindow(frameWindow.length - 1);
  }, [selectedEpisodeId, frameWindow.length]);

  useEffect(() => {
    setDraftBucket(selectedEntry?.final_bucket ?? "");
    setDraftNotes(selectedEntry?.notes ?? "");
  }, [selectedEntry]);

  const currentFrame =
    frameWindow.length > 0
      ? frameWindow[Math.min(frameIndexInWindow, frameWindow.length - 1)]
      : null;

  useEffect(() => {
    setHoveredSquare(null);
  }, [currentFrame?.annotation_id, currentFrame?.frame_index]);

  const likelyHint = useMemo(() => classificationHint(selectedEntry), [selectedEntry]);
  const decodedInputLabel = useMemo(
    () => trackerInputDescription(study?.summary.config?.temporal_mode),
    [study?.summary.config?.temporal_mode],
  );

  const handleBoardSquareHover = useCallback(
    (boardTitle: string, fen?: string | null) => {
      return (square: string | null) => {
        setHoveredSquare(square ? { boardTitle, square, fen } : null);
      };
    },
    [],
  );

  const handleSave = useCallback(
    async (bucket: string, notes: string) => {
      if (!study || !selectedEntry) return;
      try {
        setSaving(true);
        const response = await updatePhysicalFailureStudyEntry({
          study_path: study.path,
          episode_id: selectedEntry.episode_id,
          final_bucket: bucket,
          notes,
        });
        setStudy((current) => {
          if (!current) return current;
          const nextEntries = current.entries.map((entry) =>
            entry.episode_id === response.episode_id
              ? {
                  ...entry,
                  final_bucket: response.final_bucket,
                  notes: response.notes,
                  bucket_updated_at: response.updated_at ?? undefined,
                }
              : entry,
          );
          const counts = nextEntries.reduce<Record<string, number>>((acc, entry) => {
            const key = entry.final_bucket?.trim() || UNTAGGED_LABEL;
            acc[key] = (acc[key] ?? 0) + 1;
            return acc;
          }, {});
          return { ...current, entries: nextEntries, bucket_counts: counts };
        });
        toast.success(`Saved bucket for ${response.episode_id}`);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to save bucket");
      } finally {
        setSaving(false);
      }
    },
    [selectedEntry, study],
  );

  const selectBucket = useCallback(
    (bucket: string) => {
      setDraftBucket(bucket);
      void handleSave(bucket, draftNotes);
    },
    [draftNotes, handleSave],
  );

  const stepFrame = useCallback(
    (delta: number) => {
      setFrameIndexInWindow((current) => {
        if (!frameWindow.length) return 0;
        const next = current + delta;
        if (next < 0) return 0;
        if (next > frameWindow.length - 1) return frameWindow.length - 1;
        return next;
      });
    },
    [frameWindow.length],
  );

  const stepEpisode = useCallback(
    (delta: number) => {
      if (!filteredEntries.length) return;
      const index = filteredEntries.findIndex(
        (entry) => entry.episode_id === selectedEpisodeId,
      );
      if (index < 0) {
        setSelectedEpisodeId(filteredEntries[0]!.episode_id);
        return;
      }
      const next = Math.max(0, Math.min(filteredEntries.length - 1, index + delta));
      const nextEntry = filteredEntries[next];
      if (nextEntry) setSelectedEpisodeId(nextEntry.episode_id);
    },
    [filteredEntries, selectedEpisodeId],
  );

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typingIntoField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement;
      if (typingIntoField) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepFrame(-1);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        stepFrame(1);
        return;
      }
      if (event.key === "j") {
        event.preventDefault();
        stepEpisode(1);
        return;
      }
      if (event.key === "k") {
        event.preventDefault();
        stepEpisode(-1);
        return;
      }
      if (event.key === "n") {
        event.preventDefault();
        notesRef.current?.focus();
        return;
      }
      const numeric = Number(event.key);
      if (
        Number.isInteger(numeric) &&
        numeric >= 1 &&
        numeric <= 6 &&
        study?.bucket_options
      ) {
        const candidate = study.bucket_options[numeric - 1];
        if (candidate) {
          event.preventDefault();
          selectBucket(candidate);
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [selectBucket, stepEpisode, stepFrame, study]);

  const exportUrl = study ? exportPhysicalFailureStudyCsvUrl(study.path) : null;
  const failingFrame = selectedEntry?.failing_frame ?? null;
  const topLegalCandidates =
    failingFrame?.legal_from_previous_decoded?.top_legal_candidates ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Physical Failure Study Viewer</CardTitle>
          <CardDescription>
            Pick a failure-study bundle, scrub the leading frames before each FAIL,
            and tag the episode based on the first failing frame. Bucket tags persist in
            <code> manual_buckets.csv</code>. Shortcuts: <code>←/→</code> = frame,
            <code> j/k</code> = episode, <code>1–6</code> = bucket,
            <code> n</code> = focus notes.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,2fr)_auto_auto]">
            <label className="space-y-1 text-sm">
              <span className="font-medium">Study bundle</span>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                disabled={loadingStudies || studies.length === 0}
                value={selectedStudyPath}
                onChange={(event) => setSelectedStudyPath(event.target.value)}
              >
                {studies.length === 0 ? (
                  <option value="">No failure studies found</option>
                ) : null}
                {studies.map((item) => (
                  <option key={item.path} value={item.path}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="space-y-1 text-sm">
              <span className="font-medium">Selected bundle</span>
              <div className="rounded-md border bg-muted/20 px-3 py-2 text-sm text-muted-foreground">
                {selectedStudyPath || "-"}
              </div>
            </div>

            <div className="flex items-end">
              {exportUrl ? (
                <Button asChild variant="outline">
                  <a href={exportUrl} download>
                    Download CSV
                  </a>
                </Button>
              ) : (
                <Button variant="outline" disabled>
                  Download CSV
                </Button>
              )}
            </div>
          </div>

          {studies.length === 0 && !loadingStudies ? (
            <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
              Run <code>python -m pipeline.cli physical-board-failure-study ...</code>
              and reload.
            </div>
          ) : null}
        </CardContent>
      </Card>

      {study ? (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Baseline board exact"
            value={formatPercent(study.report_metrics?.board_exact_match)}
            hint={study.summary.report_path ?? undefined}
          />
          <MetricCard
            label="Move recall"
            value={formatPercent(study.report_metrics?.move_detection_recall)}
            hint={`False-change ${formatPercent(
              study.report_metrics?.static_frame_false_change_rate,
            )}`}
          />
          <MetricCard
            label="Selected / total episodes"
            value={`${study.summary.selected_episodes ?? 0} / ${
              study.summary.total_episodes ?? 0
            }`}
            hint={`max_per_video=${study.summary.config?.max_per_video ?? "-"}`}
          />
          <MetricCard
            label="Config"
            value={`${study.summary.config?.observation_input ?? "-"} · ${
              study.summary.config?.tracker_mode ?? "-"
            }`}
            hint={`lookahead w=${study.summary.config?.lookahead_window ?? "-"} m=${
              study.summary.config?.lookahead_margin ?? "-"
            }`}
          />
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
        <Card className="min-h-[640px]">
          <CardHeader>
            <CardTitle className="text-lg">Episodes</CardTitle>
            <CardDescription>
              Each row is one failure episode. Filter by clip, bucket, or notes.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search clip, bucket, note, move, frame…"
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={bucketFilter}
                onChange={(event) => setBucketFilter(event.target.value)}
              >
                {bucketOptions.map((option) => (
                  <option key={option} value={option}>
                    {option === "all" ? "All buckets" : option}
                  </option>
                ))}
              </select>
            </div>

            {study ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(study.bucket_counts).map(([bucket, count]) => (
                  <Badge key={bucket} variant={badgeVariantForBucket(bucket)}>
                    {bucket} · {count}
                  </Badge>
                ))}
              </div>
            ) : null}

            <div className="max-h-[760px] space-y-2 overflow-y-auto pr-1">
              {loadingStudy ? (
                <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Loading study…
                </div>
              ) : filteredEntries.length === 0 ? (
                <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  No episodes match the current filters.
                </div>
              ) : (
                filteredEntries.map((entry) => {
                  const selected = entry.episode_id === selectedEpisodeId;
                  const bucket = bucketLabel(entry);
                  return (
                    <button
                      key={entry.episode_id}
                      type="button"
                      onClick={() => setSelectedEpisodeId(entry.episode_id)}
                      className={`w-full rounded-lg border p-3 text-left transition-colors ${
                        selected
                          ? "border-foreground bg-muted/30"
                          : "border-border hover:bg-muted/20"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs text-muted-foreground">
                              {entry.episode_id}
                            </span>
                            <span className="truncate text-sm font-medium">
                              {entry.clip_filename}
                            </span>
                          </div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            video {entry.source_video_id} ·
                            {" "}
                            {formatFrameIndex(entry.first_frame_index)} · len {entry.length}
                          </div>
                        </div>
                        <Badge
                          variant={badgeVariantForBucket(bucket)}
                          className="max-w-[180px] truncate"
                        >
                          {bucket}
                        </Badge>
                      </div>
                      <div className="mt-2 max-h-8 overflow-hidden text-xs text-muted-foreground">
                        {detailsText(entry)}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Context scrubber</CardTitle>
              <CardDescription>
                {selectedEntry
                  ? `${selectedEntry.episode_id} · ${selectedEntry.clip_filename} · ${formatFrameIndex(
                      selectedEntry.first_frame_index,
                    )}`
                  : "Select an episode from the list."}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!selectedEntry || !currentFrame || !study ? (
                <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                  Pick an episode to inspect its leading frames.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">
                        {formatFrameIndex(currentFrame.frame_index)}
                        {" · "}
                        {currentFrame.is_failing_frame
                          ? "FAIL"
                          : `pre ${
                              currentFrame.offset_from_failure > 0 ? "+" : ""
                            }${currentFrame.offset_from_failure}`}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        decoded {currentFrame.decoded_error_count} wrong · stateless
                        {" "}
                        {currentFrame.stateless_error_count} wrong · move
                        {" "}
                        {currentFrame.decoded_move_uci ?? "stay"}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={frameIndexInWindow <= 0}
                        onClick={() => stepFrame(-1)}
                      >
                        Prev
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={frameIndexInWindow >= frameWindow.length - 1}
                        onClick={() => stepFrame(1)}
                      >
                        Next
                      </Button>
                    </div>
                  </div>

                  <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-5">
                    <ImagePanel
                      title="Raw snapshot"
                      subtitle="Human-readable frame before rectification or board rendering."
                      studyPath={study.path}
                      imagePath={rawSnapshotImagePath(currentFrame)}
                      alt={`${currentFrame.clip_filename} ${formatFrameIndex(
                        currentFrame.frame_index,
                      )} raw snapshot`}
                      emptyLabel="Snapshot unavailable"
                    />
                    <ImagePanel
                      title="Model crop"
                      subtitle="Preprocessed board crop when this study/input exposes one."
                      studyPath={study.path}
                      imagePath={processedCropImagePath(currentFrame)}
                      alt={`${currentFrame.clip_filename} ${formatFrameIndex(
                        currentFrame.frame_index,
                      )} model crop`}
                      emptyLabel="No explicit crop for this study"
                    />
                    <BoardPanel
                      title="Ground truth"
                      subtitle="Yellow border = GT changed vs previous frame"
                      fen={currentFrame.gt_fen}
                      squareStyles={groundTruthSquareStyles(
                        currentFrame.gt_changed_squares,
                      )}
                      onSquareHover={handleBoardSquareHover("Ground truth", currentFrame.gt_fen)}
                    />
                    <BoardPanel
                      title="Stateless"
                      subtitle={`${currentFrame.stateless_error_count} wrong · ${boardAccuracyLabel(
                        currentFrame.stateless_error_count,
                      )}`}
                      fen={currentFrame.stateless_fen}
                      squareStyles={predictionSquareStyles(
                        statelessErrorSquares(currentFrame),
                        currentFrame.stateless_changed_squares,
                        currentFrame.stateless_square_confidences,
                      )}
                      onSquareHover={handleBoardSquareHover("Stateless", currentFrame.stateless_fen)}
                    />
                    <BoardPanel
                      title="Decoded"
                      subtitle={`${currentFrame.decoded_error_count} wrong · ${boardAccuracyLabel(
                        currentFrame.decoded_error_count,
                      )}`}
                      fen={currentFrame.decoded_fen}
                      squareStyles={predictionSquareStyles(
                        decodedErrorSquares(currentFrame),
                        currentFrame.decoded_changed_squares,
                        currentFrame.decoded_square_confidences,
                      )}
                      onSquareHover={handleBoardSquareHover("Decoded", currentFrame.decoded_fen)}
                    />

                    <HoverInspector
                      studyPath={study.path}
                      frame={currentFrame}
                      hoveredSquare={hoveredSquare}
                    />

                    <SquareEvidencePanel
                      studyPath={study.path}
                      frame={currentFrame}
                      hoveredSquare={hoveredSquare}
                      onHoverSquare={setHoveredSquare}
                    />

                    <div className="rounded-lg border bg-muted/10 p-3 xl:col-span-2 2xl:col-span-5">
                      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Legend
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <LegendItem
                          fill="#d7ead8"
                          label="light green / green = square matches GT"
                        />
                        <LegendItem
                          fill="#e9c1c1"
                          label="pink / red = square differs from GT"
                        />
                        <LegendItem
                          border={PRED_CHANGED_BORDER}
                          label="blue border = this board changed vs previous frame"
                        />
                        <LegendItem
                          border={GT_CHANGED_BORDER}
                          label="yellow border = GT changed vs previous frame"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="text-xs text-muted-foreground">
                    Stateless = direct per-frame board from the raw logits, with no temporal or legal continuity.
                    Decoded = legal sequence result after {decodedInputLabel}. Hover any GT/stateless/decoded square or inspect the 64-square evidence grid to see the projected piece-box crop and the raw stateless class distribution used on that frame. Use the pre frames to understand how the episode starts, but bucket the episode from the first FAIL frame in the tagging panel below.
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, frameWindow.length - 1)}
                    step={1}
                    value={frameIndexInWindow}
                    onChange={(event) =>
                      setFrameIndexInWindow(Number(event.target.value))
                    }
                    className="w-full"
                  />

                  <div className="flex gap-2 overflow-x-auto pb-1">
                    {frameWindow.map((frame, index) => {
                      const active = index === frameIndexInWindow;
                      return (
                        <button
                          key={`${frame.annotation_id}-${index}`}
                          type="button"
                          onClick={() => setFrameIndexInWindow(index)}
                          className={`relative shrink-0 overflow-hidden rounded-md border ${
                            active ? "border-foreground" : "border-border"
                          }`}
                        >
                          <img
                            src={physicalFailureStudyImageUrl(
                              study.path,
                              thumbnailImagePath(frame),
                            )}
                            alt={formatFrameIndex(frame.frame_index)}
                            className="h-20 w-28 object-cover"
                          />
                          <div
                            className={`absolute inset-x-0 bottom-0 px-1 py-0.5 text-[10px] text-white ${
                              frame.is_failing_frame ? "bg-red-600/80" : "bg-black/65"
                            }`}
                          >
                            {frame.is_failing_frame
                              ? "FAIL"
                              : frame.offset_from_failure > 0
                                ? `+${frame.offset_from_failure}`
                                : frame.offset_from_failure}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Tagging</CardTitle>
              <CardDescription>
                Classify the episode from its first FAIL frame. Use the pre frames only
                for context. Click a bucket or press <code>1–6</code> to save.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {likelyHint ? (
                <div className="rounded-md border bg-muted/10 p-3 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">Heuristic read</span>
                    <Badge variant={badgeVariantForBucket(likelyHint.bucket)}>
                      {likelyHint.bucket}
                    </Badge>
                  </div>
                  <div className="mt-2 font-medium">{likelyHint.title}</div>
                  <div className="mt-1 text-muted-foreground">{likelyHint.body}</div>
                </div>
              ) : null}

              <div className="rounded-md border bg-muted/10 p-3 text-sm">
                <div className="font-medium">How to decide</div>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                  <li>
                    <strong>{RECTIFICATION_BUCKET}</strong>: the crop/grid is visibly
                    warped, shifted, clipped, or on the wrong board.
                  </li>
                  <li>
                    <strong>{CLASSIFIER_BUCKET}</strong>: geometry is okay, but the
                    stateless board is already wrong on a frame that looks stable.
                  </li>
                  <li>
                    <strong>{TEMPORAL_BUCKET}</strong>: hand or piece is between
                    squares, or decoded is basically one frame early/late.
                  </li>
                  <li>
                    <strong>{DECODER_BUCKET}</strong>: square evidence looks usable,
                    but the wrong legal move/hypothesis was chosen.
                  </li>
                  <li>
                    <strong>{EVAL_BUCKET}</strong>: GT itself looks wrong or too
                    ambiguous to call.
                  </li>
                </ul>
              </div>

              <div className="grid gap-2 md:grid-cols-2">
                {(study?.bucket_options ?? []).map((bucket, index) => {
                  const selected = draftBucket === bucket;
                  const isLikely = likelyHint?.bucket === bucket;
                  return (
                    <button
                      key={bucket}
                      type="button"
                      disabled={!selectedEntry || saving}
                      onClick={() => selectBucket(bucket)}
                      className={`rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                        selected
                          ? "border-foreground bg-muted/30"
                          : "border-border hover:bg-muted/20"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">
                          {index + 1}
                        </span>
                        <span className="font-medium">{bucket}</span>
                        {isLikely ? (
                          <Badge variant="outline" className="ml-auto">
                            likely
                          </Badge>
                        ) : null}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {BUCKET_DESCRIPTIONS[bucket]}
                      </div>
                    </button>
                  );
                })}
              </div>

              <label className="space-y-1 text-sm">
                <span className="font-medium">Notes</span>
                <textarea
                  ref={notesRef}
                  value={draftNotes}
                  onChange={(event) => setDraftNotes(event.target.value)}
                  rows={3}
                  disabled={!selectedEntry}
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                  placeholder="Why does this episode belong in that bucket?"
                />
              </label>

              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  onClick={() => void handleSave(draftBucket, draftNotes)}
                  disabled={!selectedEntry || saving}
                >
                  {saving ? "Saving…" : "Save notes + bucket"}
                </Button>
                {selectedEntry?.bucket_updated_at ? (
                  <span className="text-xs text-muted-foreground">
                    last saved {selectedEntry.bucket_updated_at}
                  </span>
                ) : null}
              </div>

              {selectedEntry ? (
                <div className="grid gap-4 text-sm xl:grid-cols-3">
                  <div className="space-y-2 rounded-md border p-3">
                    <div className="font-medium">Failing-frame summary</div>
                    <div className="text-muted-foreground">
                      decoded {selectedEntry.failing_frame.decoded_error_count} wrong ·
                      stateless {selectedEntry.failing_frame.stateless_error_count} wrong ·
                      episode length {selectedEntry.length}
                    </div>
                    <div className="text-muted-foreground">
                      prevGT=
                      {String(
                        Boolean(selectedEntry.failing_frame.decoded_matches_previous_gt),
                      )}
                      {" · "}
                      nextGT=
                      {String(
                        Boolean(selectedEntry.failing_frame.decoded_matches_next_gt),
                      )}
                      {" · "}
                      bestLegalGT=
                      {String(
                        Boolean(
                          selectedEntry.failing_frame.legal_from_previous_decoded
                            ?.best_legal_matches_gt,
                        ),
                      )}
                    </div>
                    <div className="text-muted-foreground">
                      suggested: {selectedEntry.suggested_bucket ?? "-"}
                    </div>
                  </div>

                  <div className="space-y-2 rounded-md border p-3">
                    <div className="font-medium">Squares on FAIL frame</div>
                    <div className="text-muted-foreground">
                      GT changed:{" "}
                      {(selectedEntry.failing_frame.gt_changed_squares ?? []).join(", ") || "-"}
                    </div>
                    <div className="text-muted-foreground">
                      Stateless changed:{" "}
                      {(selectedEntry.failing_frame.stateless_changed_squares ?? []).join(
                        ", ",
                      ) || "-"}
                    </div>
                    <div className="text-muted-foreground">
                      Decoded changed:{" "}
                      {(selectedEntry.failing_frame.decoded_changed_squares ?? []).join(
                        ", ",
                      ) || "-"}
                    </div>
                    <div className="text-muted-foreground">
                      Decoded errors:{" "}
                      {(selectedEntry.failing_frame.decoded_error_squares ?? []).join(
                        ", ",
                      ) || "-"}
                    </div>
                  </div>

                  <div className="space-y-2 rounded-md border p-3">
                    <div className="font-medium">Legal candidates from previous decoded</div>
                    {topLegalCandidates.length > 0 ? (
                      <div className="space-y-1 text-muted-foreground">
                        {topLegalCandidates.map((candidate, index) => (
                          <div
                            key={`${candidate.move_uci ?? "stay"}-${index}`}
                            className="flex items-center justify-between gap-2 rounded border px-2 py-1"
                          >
                            <span className="font-mono text-xs">
                              {candidate.move_uci ?? "stay"}
                            </span>
                            <span className="text-xs">score {candidate.score.toFixed(2)}</span>
                            {candidate.matches_gt ? (
                              <Badge variant="outline">GT</Badge>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-muted-foreground">
                        No legal-candidate breakdown for this frame.
                      </div>
                    )}
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
