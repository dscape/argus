import React from "react";

// Standard chess piece SVG paths in a 45×45 coordinate space (Cburnett-style).
// White pieces: white fill, black outline.
// Black pieces: black fill, black outline, white internal details.

const w = {
  fill: "#fff",
  stroke: "#000",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const b = {
  fill: "#000",
  stroke: "#000",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const detail = (color: string) => ({
  fill: "none",
  stroke: color,
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

// ── Pawn ──────────────────────────────────────────────

const PAWN_PATH =
  "M 22.5 9 C 19.79 9 17.609 11.18 17.609 13.89 C 17.609 15.293 18.254 16.536 19.246 17.393 C 13.738 19.5 8.5 25.719 8.5 36 L 36.5 36 C 36.5 25.719 31.262 19.5 25.754 17.393 C 26.746 16.536 27.391 15.293 27.391 13.89 C 27.391 11.18 25.21 9 22.5 9 z";

// ── Rook ──────────────────────────────────────────────

function rookPaths(style: typeof w) {
  return (
    <g {...style}>
      <path d="M 9 39 L 36 39 L 36 36 L 9 36 z" />
      <path d="M 12.5 32 L 14 29.5 L 31 29.5 L 32.5 32 z" />
      <path d="M 12 36 L 12 32 L 33 32 L 33 36 z" />
      <path d="M 14 29.5 L 14 16.5 L 31 16.5 L 31 29.5 z" />
      <path d="M 14 16.5 L 11 14 L 34 14 L 31 16.5 z" />
      <path d="M 11 14 L 11 9 L 15 9 L 15 11 L 20 11 L 20 9 L 25 9 L 25 11 L 30 11 L 30 9 L 34 9 L 34 14 z" />
    </g>
  );
}

// ── Knight ────────────────────────────────────────────

const KNIGHT_BODY =
  "M 22 10 C 32.5 11 38.5 18 38 39 L 15 39 C 15 30 25 32.5 23 18";
const KNIGHT_HEAD =
  "M 24 18 C 24.38 20.91 18.45 25.37 16 27 C 13 29 13.18 31.34 11 31 C 9.958 30.06 12.41 27.96 11 28 C 10 28 11.19 29.23 10 30 C 9 30 5.997 31 6 26 C 6 24 12 14 12 14 C 12 14 13.89 12.1 14 10.5 C 13.27 9.506 13.5 8.5 13.5 7.5 C 14.5 6.5 16.5 10 16.5 10 L 18.5 10 C 18.5 10 19.28 8.008 21 7 C 22 7 22 10 22 10";

// ── Bishop ────────────────────────────────────────────

const BISHOP_BASE =
  "M 9 36 C 12.39 35.03 19.11 36.43 22.5 34 C 25.89 36.43 32.61 35.03 36 36 C 36 36 37.65 36.54 39 38 C 38.32 38.97 37.35 38.99 36 38.5 C 32.61 37.53 25.89 38.96 22.5 37.5 C 19.11 38.96 12.39 37.53 9 38.5 C 7.646 38.99 6.677 38.97 6 38 C 7.354 36.06 9 36 9 36 z";
const BISHOP_BODY =
  "M 15 32 C 17.5 34.5 27.5 34.5 30 32 C 30.5 30.5 30 30 30 30 C 30 27.5 27.5 26 27.5 26 C 33 24.5 33.5 14.5 22.5 10.5 C 11.5 14.5 12 24.5 17.5 26 C 17.5 26 15 27.5 15 30 C 15 30 14.5 30.5 15 32 z";
const BISHOP_TOP = "M 25 8 A 2.5 2.5 0 1 1 20 8 A 2.5 2.5 0 1 1 25 8 z";
const BISHOP_DETAILS =
  "M 17.5 26 L 27.5 26 M 15 30 L 30 30 M 22.5 15.5 L 22.5 20.5 M 20 18 L 25 18";

// ── Queen ─────────────────────────────────────────────

const QUEEN_CROWN =
  "M 9 26 C 17.5 24.5 30 24.5 36 26 L 38.5 13.5 L 31 25 L 30.7 10.9 L 25.5 24.5 L 22.5 10 L 19.5 24.5 L 14.3 10.9 L 14 25 L 6.5 13.5 L 9 26 z";
const QUEEN_COLLAR =
  "M 9 26 C 9 28 10.5 28.5 12.5 30 C 14.5 31.5 16.5 31 16.5 31 C 18.5 30 19.5 30.5 20 31 C 20.5 30.5 21.5 30 23.5 31 C 23.5 31 25.5 31.5 27.5 30 C 29.5 28.5 36 28 36 26 C 27.5 24.5 17.5 24.5 9 26 z";
const QUEEN_BASE =
  "M 9 39 L 36 39 L 36 36 C 27.5 34.5 17.5 34.5 9 36 z";
const QUEEN_BASE_DETAIL = "M 12 36 C 16 37.5 29 37.5 33 36";
const QUEEN_CIRCLES: [number, number][] = [
  [6, 12],
  [14, 9],
  [22.5, 8],
  [31, 9],
  [39, 12],
];

// ── King ──────────────────────────────────────────────

const KING_BODY =
  "M 12.5 37 C 18 40.5 27 40.5 32.5 37 L 32.5 30 C 32.5 30 41.5 25.5 38.5 19.5 C 34.5 13 25 16 22.5 23.5 L 22.5 27 L 22.5 23.5 C 20 16 10.5 13 6.5 19.5 C 3.5 25.5 12.5 30 12.5 30 z";
const KING_CROWN_DETAIL = "M 12.5 30 C 18 27 27 27 32.5 30";
const KING_BASE_DETAIL = "M 12.5 37 C 18 40.5 27 40.5 32.5 37";

// ── Piece definitions ─────────────────────────────────

const PIECES: Record<string, () => React.ReactNode> = {
  // White Pawn
  P: () => <path d={PAWN_PATH} {...w} />,

  // Black Pawn
  p: () => <path d={PAWN_PATH} {...b} />,

  // White Rook
  R: () => rookPaths(w),

  // Black Rook
  r: () => (
    <g>
      {rookPaths(b)}
      <path
        d="M 14 29.5 L 14 16.5 M 31 29.5 L 31 16.5 M 12 36 L 12 32 M 33 36 L 33 32 M 11 14 L 34 14"
        {...detail("#fff")}
        strokeWidth={1}
      />
    </g>
  ),

  // White Knight
  N: () => (
    <g>
      <path d={KNIGHT_BODY} {...w} />
      <path d={KNIGHT_HEAD} {...w} />
      <circle cx={8.5} cy={25.5} r={0.5} fill="#000" stroke="none" />
      <ellipse
        cx={14.5}
        cy={15.5}
        rx={0.5}
        ry={1.5}
        fill="#000"
        stroke="none"
        transform="rotate(30 14.5 15.5)"
      />
    </g>
  ),

  // Black Knight
  n: () => (
    <g>
      <path d={KNIGHT_BODY} {...b} />
      <path d={KNIGHT_HEAD} {...b} />
      <circle cx={8.5} cy={25.5} r={0.5} fill="#fff" stroke="none" />
      <ellipse
        cx={14.5}
        cy={15.5}
        rx={0.5}
        ry={1.5}
        fill="#fff"
        stroke="none"
        transform="rotate(30 14.5 15.5)"
      />
    </g>
  ),

  // White Bishop
  B: () => (
    <g>
      <g {...w}>
        <path d={BISHOP_BASE} />
        <path d={BISHOP_BODY} />
        <path d={BISHOP_TOP} />
      </g>
      <path d={BISHOP_DETAILS} {...detail("#000")} />
    </g>
  ),

  // Black Bishop
  b: () => (
    <g>
      <g {...b}>
        <path d={BISHOP_BASE} />
        <path d={BISHOP_BODY} />
        <path d={BISHOP_TOP} />
      </g>
      <path d={BISHOP_DETAILS} {...detail("#fff")} />
    </g>
  ),

  // White Queen
  Q: () => (
    <g {...w}>
      <path d={QUEEN_CROWN} />
      <path d={QUEEN_COLLAR} />
      <path d={QUEEN_BASE} />
      <path d={QUEEN_BASE_DETAIL} fill="none" />
      {QUEEN_CIRCLES.map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r={2} />
      ))}
    </g>
  ),

  // Black Queen
  q: () => (
    <g>
      <g {...b}>
        <path d={QUEEN_CROWN} />
        <path d={QUEEN_COLLAR} />
        <path d={QUEEN_BASE} />
        {QUEEN_CIRCLES.map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r={2} />
        ))}
      </g>
      <path d={QUEEN_BASE_DETAIL} {...detail("#fff")} />
      <path
        d="M 9 26 C 9 28 10.5 28.5 12.5 30 C 14.5 31.5 16.5 31 16.5 31 C 18.5 30 19.5 30.5 20 31 C 20.5 30.5 21.5 30 23.5 31 C 23.5 31 25.5 31.5 27.5 30 C 29.5 28.5 36 28 36 26"
        {...detail("#fff")}
      />
    </g>
  ),

  // White King
  K: () => (
    <g>
      <path
        d="M 22.5 11.63 L 22.5 6"
        fill="none"
        stroke="#000"
        strokeWidth={1.5}
        strokeLinejoin="miter"
      />
      <path
        d="M 20 8 L 25 8"
        fill="none"
        stroke="#000"
        strokeWidth={1.5}
        strokeLinejoin="miter"
      />
      <path d={KING_BODY} {...w} />
      <path d={KING_CROWN_DETAIL} {...detail("#000")} />
      <path d={KING_BASE_DETAIL} {...detail("#000")} />
    </g>
  ),

  // Black King
  k: () => (
    <g>
      <path
        d="M 22.5 11.63 L 22.5 6"
        fill="none"
        stroke="#000"
        strokeWidth={1.5}
        strokeLinejoin="miter"
      />
      <path
        d="M 20 8 L 25 8"
        fill="none"
        stroke="#000"
        strokeWidth={1.5}
        strokeLinejoin="miter"
      />
      <path d={KING_BODY} {...b} />
      <path d={KING_CROWN_DETAIL} {...detail("#fff")} />
      <path d={KING_BASE_DETAIL} {...detail("#fff")} />
    </g>
  ),
};

export function chessPieceSvg(piece: string): React.ReactNode {
  const fn = PIECES[piece];
  return fn ? fn() : null;
}
