"use client";

import { useRef, useCallback, useEffect, useState } from "react";
import type { VideoClip } from "@/lib/types";
import { fmtTime } from "@/lib/format";

export interface DragState {
  clipId: number;
  edge: "start" | "end";
  originalTime: number;
  currentTime: number;
}

const SEGMENT_COLORS = [
  { bg: "bg-blue-500", bgAlpha: "bg-blue-500/50", bgHover: "bg-blue-500/60", bgSelected: "bg-blue-500/70", ring: "ring-blue-500", dot: "bg-blue-500" },
  { bg: "bg-emerald-500", bgAlpha: "bg-emerald-500/50", bgHover: "bg-emerald-500/60", bgSelected: "bg-emerald-500/70", ring: "ring-emerald-500", dot: "bg-emerald-500" },
  { bg: "bg-violet-500", bgAlpha: "bg-violet-500/50", bgHover: "bg-violet-500/60", bgSelected: "bg-violet-500/70", ring: "ring-violet-500", dot: "bg-violet-500" },
  { bg: "bg-amber-500", bgAlpha: "bg-amber-500/50", bgHover: "bg-amber-500/60", bgSelected: "bg-amber-500/70", ring: "ring-amber-500", dot: "bg-amber-500" },
  { bg: "bg-rose-500", bgAlpha: "bg-rose-500/50", bgHover: "bg-rose-500/60", bgSelected: "bg-rose-500/70", ring: "ring-rose-500", dot: "bg-rose-500" },
  { bg: "bg-cyan-500", bgAlpha: "bg-cyan-500/50", bgHover: "bg-cyan-500/60", bgSelected: "bg-cyan-500/70", ring: "ring-cyan-500", dot: "bg-cyan-500" },
  { bg: "bg-orange-500", bgAlpha: "bg-orange-500/50", bgHover: "bg-orange-500/60", bgSelected: "bg-orange-500/70", ring: "ring-orange-500", dot: "bg-orange-500" },
  { bg: "bg-fuchsia-500", bgAlpha: "bg-fuchsia-500/50", bgHover: "bg-fuchsia-500/60", bgSelected: "bg-fuchsia-500/70", ring: "ring-fuchsia-500", dot: "bg-fuchsia-500" },
] as const;

const GAP_COLOR = { bgAlpha: "bg-yellow-500/30", bgHover: "bg-yellow-500/40", bgSelected: "bg-yellow-500/50", ring: "ring-yellow-500", dot: "bg-yellow-500" };

function getSegmentColor(clipIndex: number) {
  return SEGMENT_COLORS[clipIndex % SEGMENT_COLORS.length];
}

interface SegmentTimelineProps {
  clips: VideoClip[];
  duration: number;
  frameIdx: number;
  totalFrames: number;
  fps: number;
  selectedClipIds: Set<number>;
  dragging: DragState | null;
  onFrameChange: (frameIdx: number) => void;
  onSelectClip: (clipId: number) => void;
  onDragStart: (clipId: number, edge: "start" | "end") => void;
  onDragMove: (time: number) => void;
  onDragEnd: () => void;
}

export function SegmentTimeline({
  clips,
  duration,
  frameIdx,
  totalFrames,
  fps,
  selectedClipIds,
  dragging,
  onFrameChange,
  onSelectClip,
  onDragStart,
  onDragMove,
  onDragEnd,
}: SegmentTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  const timeToPercent = (time: number) => duration > 0 ? (time / duration) * 100 : 0;
  const playheadPct = totalFrames > 1 ? (frameIdx / (totalFrames - 1)) * 100 : 0;

  const posToTime = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || duration <= 0) return 0;
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return pct * duration;
  }, [duration]);

  // Register document-level mouse handlers during drag
  useEffect(() => {
    if (!dragging) return;
    const handleMouseMove = (e: MouseEvent) => {
      onDragMove(posToTime(e.clientX));
    };
    const handleMouseUp = () => {
      onDragEnd();
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragging, onDragMove, onDragEnd, posToTime]);

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (dragging) return;
    const time = posToTime(e.clientX);
    const frame = Math.round(time * fps);
    onFrameChange(Math.max(0, Math.min(totalFrames - 1, frame)));
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragging) return;
    setHoverTime(posToTime(e.clientX));
  };

  const handleMouseLeave = () => {
    if (!dragging) setHoverTime(null);
  };

  // Get effective time for a clip edge during drag
  const getEffectiveTime = (clip: VideoClip, edge: "start" | "end") => {
    if (dragging && dragging.clipId === clip.id && dragging.edge === edge) {
      return dragging.currentTime;
    }
    return edge === "start" ? clip.start_time : (clip.end_time ?? duration);
  };

  return (
    <div className="space-y-1">
      <div
        ref={containerRef}
        className="relative w-full h-10 bg-muted rounded-lg overflow-hidden cursor-crosshair select-none"
        onClick={handleTimelineClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Segment regions (gaps are invisible — they're just empty space) */}
        {clips.filter((c) => !c.is_gap).map((c) => {
          const startTime = getEffectiveTime(c, "start");
          const endTime = getEffectiveTime(c, "end");
          const startPct = timeToPercent(startTime);
          const widthPct = timeToPercent(endTime - startTime);
          const isSelected = selectedClipIds.has(c.id);
          const color = getSegmentColor(c.clip_index);

          return (
            <div
              key={c.id}
              className={`absolute top-0 h-full transition-colors ${
                isSelected
                  ? `${color.bgSelected} ring-2 ${color.ring}`
                  : `${color.bgAlpha} hover:${color.bgHover}`
              }`}
              style={{ left: `${startPct}%`, width: `${Math.max(widthPct, 0.3)}%` }}
              title={`${c.label || `Segment ${c.clip_index + 1}`}: ${fmtTime(startTime)} \u2014 ${fmtTime(endTime)}`}
              onClick={(e) => {
                e.stopPropagation();
                const time = posToTime(e.clientX);
                const frame = Math.round(time * fps);
                onFrameChange(Math.max(0, Math.min(totalFrames - 1, frame)));
              }}
            >
              {/* Left boundary handle */}
              <div
                className="absolute left-0 top-0 w-2 h-full cursor-col-resize z-10 hover:bg-foreground/20"
                onMouseDown={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  onDragStart(c.id, "start");
                }}
              />
              {/* Right boundary handle */}
              <div
                className="absolute right-0 top-0 w-2 h-full cursor-col-resize z-10 hover:bg-foreground/20"
                onMouseDown={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  onDragStart(c.id, "end");
                }}
              />
              {/* Segment label if wide enough */}
              {widthPct > 5 && (
                <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-white/80 pointer-events-none truncate px-3 drop-shadow-sm">
                  {c.clip_index + 1}
                </span>
              )}
            </div>
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 w-0.5 h-full bg-foreground pointer-events-none z-20"
          style={{ left: `${playheadPct}%` }}
        />

        {/* Hover indicator */}
        {hoverTime !== null && !dragging && (
          <div
            className="absolute top-0 w-px h-full bg-foreground/30 pointer-events-none z-10"
            style={{ left: `${timeToPercent(hoverTime)}%` }}
          />
        )}
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-[10px] text-muted-foreground px-0.5">
        <span>0:00</span>
        <span>{fmtTime(duration)}</span>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
        {clips.map((c) => {
          const isGap = c.is_gap;
          const color = isGap ? GAP_COLOR : getSegmentColor(c.clip_index);
          const isSelected = selectedClipIds.has(c.id);
          const endTime = c.end_time ?? duration;
          return (
            <button
              key={c.id}
              onClick={() => onSelectClip(c.id)}
              className={`flex items-center gap-1.5 px-1.5 py-0.5 rounded text-[10px] transition-colors ${
                isSelected
                  ? `ring-1 ${color.ring} bg-muted/50`
                  : "hover:bg-muted/30"
              }`}
            >
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${color.dot}`} />
              <span className="text-muted-foreground tabular-nums">
                {fmtTime(c.start_time)}&ndash;{fmtTime(endTime)}
              </span>
              {isGap && <span className="text-yellow-600">(gap)</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
