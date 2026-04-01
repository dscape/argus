"use client";

import { chessPieceSvg } from "./chess-pieces";

const LIGHT = "#F0D9B5";
const DARK = "#B58863";

interface Props {
  fen: string;
  size?: number;
  flipped?: boolean;
}

export function ChessBoard({ fen, size = 280, flipped = false }: Props) {
  const sqSize = size / 8;
  const rows = fen.split("/");
  const squares: { row: number; col: number; piece: string | null }[] = [];

  for (let r = 0; r < 8; r++) {
    let col = 0;
    for (const ch of rows[r] || "") {
      if (ch >= "1" && ch <= "8") {
        for (let i = 0; i < parseInt(ch); i++) {
          squares.push({ row: r, col, piece: null });
          col++;
        }
      } else {
        squares.push({ row: r, col, piece: ch });
        col++;
      }
    }
  }

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="border rounded">
      {squares.map(({ row, col, piece }) => {
        const dr = flipped ? 7 - row : row;
        const dc = flipped ? 7 - col : col;
        const isLight = (dr + dc) % 2 === 0;
        const x = col * sqSize;
        const y = row * sqSize;
        return (
          <g key={`${row}-${col}`}>
            <rect x={x} y={y} width={sqSize} height={sqSize} fill={isLight ? LIGHT : DARK} />
            {piece && (
              <g transform={`translate(${x}, ${y}) scale(${sqSize / 45})`}>
                {chessPieceSvg(piece)}
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
