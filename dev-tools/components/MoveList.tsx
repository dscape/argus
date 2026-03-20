"use client";

import { Badge } from "@/components/ui/badge";

interface Move {
  frame_index: number;
  uci: string;
  san?: string | null;
  detect_value?: number | null;
}

interface Props {
  moves: Move[];
  onMoveClick?: (frameIndex: number) => void;
  activeFrame?: number;
}

export function MoveList({ moves, onMoveClick, activeFrame }: Props) {
  if (moves.length === 0) {
    return <p className="text-sm text-muted-foreground">No moves detected</p>;
  }

  return (
    <div className="space-y-1 max-h-96 overflow-y-auto">
      {moves.map((move, i) => {
        const isWhite = i % 2 === 0;
        const moveNum = Math.floor(i / 2) + 1;
        const isActive = activeFrame === move.frame_index;
        return (
          <div
            key={i}
            onClick={() => onMoveClick?.(move.frame_index)}
            className={`flex items-center gap-2 px-2 py-1 rounded text-sm cursor-pointer hover:bg-accent ${
              isActive ? "bg-accent" : ""
            }`}
          >
            {isWhite && (
              <span className="text-muted-foreground w-6">{moveNum}.</span>
            )}
            {!isWhite && <span className="w-6" />}
            <span className="font-mono">{move.san || move.uci}</span>
            <Badge variant="outline" className="text-xs ml-auto">
              f{move.frame_index}
            </Badge>
          </div>
        );
      })}
    </div>
  );
}
