"use client";

import { chessPieceSvg } from "./chess-pieces";

const LIGHT = "#F0D9B5";
const DARK = "#B58863";
const MOVE_HIGHLIGHT = "#F6D66A";
const MOVE_ARROW = "#E8C547";

interface ChessArrow {
  from: string;
  to: string;
  color?: string;
}

interface Props {
  fen: string;
  size?: number;
  flipped?: boolean;
  highlightedSquares?: string[];
  arrows?: ChessArrow[];
}

type Square = {
  row: number;
  col: number;
  piece: string | null;
};

type Point = {
  x: number;
  y: number;
};

export function ChessBoard({
  fen,
  size = 280,
  flipped = false,
  highlightedSquares = [],
  arrows = [],
}: Props) {
  const sqSize = size / 8;
  const boardFen = fen.split(" ")[0] ?? fen;
  const rows = boardFen.split("/");
  const squares: Square[] = [];

  for (let row = 0; row < 8; row++) {
    let col = 0;
    for (const ch of rows[row] || "") {
      if (ch >= "1" && ch <= "8") {
        const emptyCount = parseInt(ch, 10);
        for (let i = 0; i < emptyCount; i += 1) {
          squares.push({ row, col, piece: null });
          col += 1;
        }
      } else {
        squares.push({ row, col, piece: ch });
        col += 1;
      }
    }
  }

  const highlightSet = new Set(highlightedSquares);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="border rounded">
      {squares.map(({ row, col }) => {
        const { x, y } = squareTopLeft(row, col, sqSize, flipped);
        const isLight = (row + col) % 2 === 0;
        const squareName = boardCoordsToSquare(row, col);
        return (
          <g key={`square-${row}-${col}`}>
            <rect x={x} y={y} width={sqSize} height={sqSize} fill={isLight ? LIGHT : DARK} />
            {highlightSet.has(squareName) && (
              <rect
                x={x}
                y={y}
                width={sqSize}
                height={sqSize}
                fill={MOVE_HIGHLIGHT}
                fillOpacity={0.5}
              />
            )}
          </g>
        );
      })}

      {arrows.map((arrow, index) => {
        const points = getArrowPoints(arrow.from, arrow.to, sqSize, flipped);
        if (!points) {
          return null;
        }
        const color = arrow.color ?? MOVE_ARROW;
        return (
          <g key={`arrow-${arrow.from}-${arrow.to}-${index}`} opacity={0.9}>
            <line
              x1={points.start.x}
              y1={points.start.y}
              x2={points.shaftEnd.x}
              y2={points.shaftEnd.y}
              stroke={color}
              strokeWidth={sqSize * 0.18}
              strokeLinecap="round"
            />
            <polygon
              points={`${points.tip.x},${points.tip.y} ${points.left.x},${points.left.y} ${points.right.x},${points.right.y}`}
              fill={color}
            />
          </g>
        );
      })}

      {squares.map(({ row, col, piece }) => {
        if (!piece) {
          return null;
        }
        const { x, y } = squareTopLeft(row, col, sqSize, flipped);
        return (
          <g key={`piece-${row}-${col}`} transform={`translate(${x}, ${y}) scale(${sqSize / 45})`}>
            {chessPieceSvg(piece)}
          </g>
        );
      })}
    </svg>
  );
}

function squareTopLeft(row: number, col: number, sqSize: number, flipped: boolean): Point {
  const displayRow = flipped ? 7 - row : row;
  const displayCol = flipped ? 7 - col : col;
  return {
    x: displayCol * sqSize,
    y: displayRow * sqSize,
  };
}

function boardCoordsToSquare(row: number, col: number): string {
  const file = String.fromCharCode("a".charCodeAt(0) + col);
  const rank = String(8 - row);
  return `${file}${rank}`;
}

function squareToBoardCoords(square: string): { row: number; col: number } | null {
  if (!/^[a-h][1-8]$/i.test(square)) {
    return null;
  }
  const file = square[0]!.toLowerCase().charCodeAt(0) - "a".charCodeAt(0);
  const rank = parseInt(square[1]!, 10);
  return {
    row: 8 - rank,
    col: file,
  };
}

function squareCenter(square: string, sqSize: number, flipped: boolean): Point | null {
  const coords = squareToBoardCoords(square);
  if (!coords) {
    return null;
  }
  const topLeft = squareTopLeft(coords.row, coords.col, sqSize, flipped);
  return {
    x: topLeft.x + sqSize / 2,
    y: topLeft.y + sqSize / 2,
  };
}

function getArrowPoints(
  from: string,
  to: string,
  sqSize: number,
  flipped: boolean,
): {
  start: Point;
  shaftEnd: Point;
  tip: Point;
  left: Point;
  right: Point;
} | null {
  const start = squareCenter(from, sqSize, flipped);
  const tip = squareCenter(to, sqSize, flipped);
  if (!start || !tip) {
    return null;
  }

  const dx = tip.x - start.x;
  const dy = tip.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length === 0) {
    return null;
  }

  const ux = dx / length;
  const uy = dy / length;
  const headLength = sqSize * 0.42;
  const headWidth = sqSize * 0.34;
  const shaftEnd = {
    x: tip.x - ux * headLength,
    y: tip.y - uy * headLength,
  };
  const px = -uy;
  const py = ux;

  return {
    start,
    shaftEnd,
    tip,
    left: {
      x: shaftEnd.x + px * (headWidth / 2),
      y: shaftEnd.y + py * (headWidth / 2),
    },
    right: {
      x: shaftEnd.x - px * (headWidth / 2),
      y: shaftEnd.y - py * (headWidth / 2),
    },
  };
}
