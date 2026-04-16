const LIGHT_SQUARE: [number, number, number] = [240, 217, 181];
const DARK_SQUARE: [number, number, number] = [181, 136, 99];
const CORRECT_COLOR: [number, number, number] = [51, 171, 88];
const ERROR_COLOR: [number, number, number] = [204, 65, 65];

export const GT_CHANGED_BORDER = "#F4D03F";
export const PRED_CHANGED_BORDER = "#49A0FF";

export interface PhysicalBoardSquareStyle {
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
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

export function groundTruthSquareStyles(
  changedSquares?: string[],
): Record<string, PhysicalBoardSquareStyle> {
  const styles: Record<string, PhysicalBoardSquareStyle> = {};
  for (const square of changedSquares ?? []) {
    styles[square] = { stroke: GT_CHANGED_BORDER, strokeWidth: 3 };
  }
  return styles;
}

export function predictionSquareStyles(
  errorSquares?: string[],
  changedSquares?: string[],
  confidences?: number[],
): Record<string, PhysicalBoardSquareStyle> {
  const styles: Record<string, PhysicalBoardSquareStyle> = {};
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
