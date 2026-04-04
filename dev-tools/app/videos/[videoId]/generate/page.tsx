"use client";

import { useState } from "react";
import { generateClips } from "@/lib/api";
import type { GeneratedClip } from "@/lib/types";
import { fmtTime } from "@/lib/format";
import { useVideoWorkbench } from "../_context";

export default function GeneratePage() {
  const { session, clips, downloadStatus } = useVideoWorkbench();
  const [generatingId, setGeneratingId] = useState<number | null>(null);
  const [generated, setGenerated] = useState<Map<number, GeneratedClip[]>>(new Map());
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (clipId?: number) => {
    if (!session) return;
    setGeneratingId(clipId ?? -1);
    setError(null);
    try {
      const result = await generateClips(session.session_id, clipId);
      setGenerated((prev) => new Map(prev).set(clipId ?? -1, result.clips));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGeneratingId(null);
    }
  };

  if (!downloadStatus?.downloaded) {
    return <p className="text-sm text-muted-foreground pt-2">Video must be downloaded first.</p>;
  }

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video session...</p>;
  }

  const hasClips = clips.length > 0;
  if (!hasClips && !session.has_calibration) {
    return <p className="text-sm text-muted-foreground pt-2">No clips or calibration found. Go to the Calibrate step first.</p>;
  }

  return (
    <div className="space-y-4 pt-2 max-w-3xl">
      {hasClips ? (
        <div className="space-y-4">
          {clips.map((vc) => {
            const gen = generated.get(vc.id) || [];
            const isGenerating = generatingId === vc.id;
            return (
              <div key={vc.id} className="border rounded-lg p-3 space-y-2">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium">{vc.label || `Clip ${vc.clip_index + 1}`}</span>
                  <span className="text-[10px] text-muted-foreground">
                    {fmtTime(vc.start_time)} &mdash; {vc.end_time != null ? fmtTime(vc.end_time) : "end"}
                  </span>
                  <button
                    onClick={() => handleGenerate(vc.id)}
                    disabled={isGenerating}
                    className="px-3 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
                  >
                    {isGenerating ? "Generating..." : "Generate"}
                  </button>
                </div>
                {gen.length > 0 && (
                  <div className="space-y-1 pl-2">
                    <div className="text-xs text-muted-foreground">{gen.length} training clip(s)</div>
                    {gen.map((g) => (
                      <div key={g.game_index} className="text-xs">
                        <span className="font-medium">Game {g.game_index + 1}</span>
                        <span className="text-muted-foreground ml-2">{g.num_moves} moves, {g.num_frames} frames</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="space-y-3">
          <button
            onClick={() => handleGenerate()}
            disabled={generatingId !== null}
            className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {generatingId !== null ? "Generating clips..." : "Generate Training Clips"}
          </button>
          {(() => {
            const gen = generated.get(-1) || [];
            if (gen.length === 0) return null;
            return (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Generated {gen.length} clip(s)</h3>
                {gen.map((clip) => (
                  <div key={clip.game_index} className="border rounded-lg p-3 space-y-1">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-medium">Game {clip.game_index + 1}</span>
                      <span className="text-xs text-muted-foreground">{clip.num_moves} moves, {clip.num_frames} frames</span>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground">{clip.filepath}</div>
                  </div>
                ))}
              </div>
            );
          })()}
        </div>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
