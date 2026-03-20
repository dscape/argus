"use client";

const PIECE_UNICODE: Record<string, string> = {
  K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};

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
              <text
                x={x + sqSize / 2}
                y={y + sqSize * 0.62}
                textAnchor="middle"
                fontSize={sqSize * 0.75}
                className="select-none"
              >
                {PIECE_UNICODE[piece] || piece}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
