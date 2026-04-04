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

interface SegmentTimelineProps {
  clips: VideoClip[];
  duration: number;
  frameIdx: number;
  totalFrames: number;
  fps: number;
  selectedClipIds: Set<number>;
  dragging: DragState | null;
  onFrameChange: (frameIdx: number) => void;
  onSelectClip: (clipId: number, multi: boolean) => void;
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
        {/* Segment regions */}
        {clips.map((c) => {
          const startTime = getEffectiveTime(c, "start");
          const endTime = getEffectiveTime(c, "end");
          const startPct = timeToPercent(startTime);
          const widthPct = timeToPercent(endTime - startTime);
          const isSelected = selectedClipIds.has(c.id);
          const isGap = c.is_gap;

          return (
            <div
              key={c.id}
              className={`absolute top-0 h-full transition-colors ${
                isGap
                  ? isSelected ? "bg-yellow-500/50 ring-2 ring-yellow-500" : "bg-yellow-500/30 hover:bg-yellow-500/40"
                  : isSelected ? "bg-primary/70 ring-2 ring-primary" : "bg-primary/50 hover:bg-primary/60"
              }`}
              style={{ left: `${startPct}%`, width: `${Math.max(widthPct, 0.3)}%` }}
              title={`${c.label || `Segment ${c.clip_index + 1}`}${isGap ? " (gap)" : ""}: ${fmtTime(startTime)} \u2014 ${fmtTime(endTime)}`}
              onClick={(e) => {
                e.stopPropagation();
                onSelectClip(c.id, e.metaKey || e.ctrlKey);
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
                <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-foreground/70 pointer-events-none truncate px-3">
                  {c.label || `${c.clip_index + 1}`}
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
    </div>
  );
}
