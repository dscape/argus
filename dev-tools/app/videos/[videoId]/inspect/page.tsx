"use client";

import { useState } from "react";
import { toast } from "sonner";
import { loadClip, getClipInfo, clipFrameUrl } from "@/lib/api";
import type { GeneratedClip, ClipInspectResponse } from "@/lib/types";
import { useVideoWorkbench } from "../_context";

export default function InspectPage() {
  const { video } = useVideoWorkbench();
  const [clips] = useState<GeneratedClip[]>([]);
  const [inspecting, setInspecting] = useState<number | null>(null);
  const [clipInfo, setClipInfo] = useState<Map<number, ClipInspectResponse>>(new Map());
  const [clipSessions, setClipSessions] = useState<Map<number, string>>(new Map());
  const [frameIndices, setFrameIndices] = useState<Map<number, number>>(new Map());

  const handleInspectClip = async (clip: GeneratedClip) => {
    setInspecting(clip.game_index);
    try {
      const response = await fetch(clip.filepath);
      const blob = await response.blob();
      const file = new File([blob], clip.filepath.split("/").pop() || "clip.pt");
      const { session_id } = await loadClip(file);
      setClipSessions((prev) => new Map(prev).set(clip.game_index, session_id));

      const info = await getClipInfo(session_id);
      if (info.replay_error) toast.error(info.replay_error);
      setClipInfo((prev) => new Map(prev).set(clip.game_index, info));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to inspect clip");
    } finally {
      setInspecting(null);
    }
  };

  if (!video) return null;

  return (
    <div className="space-y-4 pt-2 max-w-4xl">
      <p className="text-sm text-muted-foreground">
        Generate clips first (Step 6), then inspect them here to verify training data quality.
      </p>

      {clips.length === 0 && (
        <p className="text-xs text-muted-foreground">No clips to inspect yet. Run clip generation first.</p>
      )}

      {clips.map((clip) => {
        const info = clipInfo.get(clip.game_index);
        const sessionId = clipSessions.get(clip.game_index);
        const currentFrame = frameIndices.get(clip.game_index) ?? 0;

        return (
          <div key={clip.game_index} className="border rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium">Game {clip.game_index + 1}</span>
              <span className="text-xs text-muted-foreground">{clip.num_moves} moves, {clip.num_frames} frames</span>
              <button
                onClick={() => handleInspectClip(clip)}
                disabled={inspecting === clip.game_index}
                className="text-xs text-primary hover:underline"
              >
                {inspecting === clip.game_index ? "Loading..." : "Inspect"}
              </button>
            </div>

            {info && sessionId && (
              <div className="space-y-2">
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Replay:</span>{" "}
                    <span className={info.replay_valid ? "text-green-600" : "text-destructive"}>
                      {info.replay_valid ? "Valid" : "Invalid"}
                    </span>
                  </div>
                  <div><span className="text-muted-foreground">Moves:</span> {info.total_moves}</div>
                  <div><span className="text-muted-foreground">No-move frames:</span> {info.no_move_frames}</div>
                  <div><span className="text-muted-foreground">Avg legal:</span> {info.avg_legal_moves?.toFixed(1)}</div>
                </div>

                <div>
                  <input
                    type="range"
                    min={0}
                    max={info.num_frames - 1}
                    value={currentFrame}
                    onChange={(e) => setFrameIndices((prev) => new Map(prev).set(clip.game_index, Number(e.target.value)))}
                    className="w-full"
                  />
                  <div className="flex gap-3">
                    <img
                      src={clipFrameUrl(sessionId, currentFrame)}
                      alt={`Frame ${currentFrame}`}
                      className="w-56 rounded border"
                    />
                    <div className="text-xs space-y-1">
                      <div>Frame {currentFrame} / {info.num_frames}</div>
                      {info.moves.filter((m) => m.frame_index === currentFrame).map((m) => (
                        <div key={m.frame_index} className="font-medium text-primary">
                          Move: {m.san || m.uci}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}

    </div>
  );
}
