"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  startGenerateClipsJob,
  getGenerateClipsJobStatus,
} from "@/lib/api";
import type { GeneratedClip } from "@/lib/types";
import { fmtTime } from "@/lib/format";
import { useVideoWorkbench } from "../_context";

interface GenerateJob {
  job_id: string;
  status: string;
  clips: GeneratedClip[];
  error: string | null;
}

export default function GeneratePage() {
  const { session, activeClips: clips, downloadStatus } = useVideoWorkbench();
  const [job, setJob] = useState<GenerateJob | null>(null);
  const [starting, setStarting] = useState(false);

  const isRunning = job?.status === "running";

  // Poll job status
  useEffect(() => {
    if (!job || job.status !== "running" || !session) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getGenerateClipsJobStatus(session.session_id, job.job_id);
        if (cancelled) return;
        setJob(status);
        if (status.status === "failed") {
          toast.error(status.error ?? "Generation failed");
        }
      } catch (e) {
        if (cancelled) return;
        toast.error(e instanceof Error ? e.message : "Failed to poll job status");
      }
    };

    void poll();
    const id = window.setInterval(poll, 2000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [job?.job_id, job?.status, session]);

  const handleGenerate = async (clipId?: number) => {
    if (!session) return;
    setStarting(true);
    setJob(null);
    try {
      const j = await startGenerateClipsJob(session.session_id, clipId);
      setJob(j);
      if (j.status === "failed") {
        toast.error(j.error ?? "Generation failed");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setStarting(false);
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

  const generatedClips = job?.clips ?? [];

  return (
    <div className="space-y-4 pt-2 max-w-3xl">
      {hasClips ? (
        <div className="space-y-4">
          {clips.map((vc) => (
            <div key={vc.id} className="border rounded-lg p-3 space-y-2">
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium">{vc.label || `Clip ${vc.clip_index + 1}`}</span>
                <span className="text-[10px] text-muted-foreground">
                  {fmtTime(vc.start_time)} &mdash; {vc.end_time != null ? fmtTime(vc.end_time) : "end"}
                </span>
                <button
                  onClick={() => handleGenerate(vc.id)}
                  disabled={isRunning || starting}
                  className="px-3 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
                >
                  {isRunning ? "Generating..." : "Generate"}
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <button
          onClick={() => handleGenerate()}
          disabled={isRunning || starting}
          className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {isRunning ? "Generating clips..." : starting ? "Starting..." : "Generate Training Clips"}
        </button>
      )}

      {/* Progress / running state */}
      {isRunning && (
        <div className="space-y-2 border rounded-lg p-3 bg-muted/10">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-xs text-muted-foreground">
              Generating training clips...
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-primary animate-pulse" style={{ width: "100%" }} />
          </div>
        </div>
      )}

      {/* Generated clips (shown progressively) */}
      {generatedClips.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">
            {job?.status === "done" ? `Generated ${generatedClips.length} clip(s)` : `${generatedClips.length} clip(s) so far...`}
          </h3>
          {generatedClips.map((clip) => (
            <div key={clip.game_index} className="border rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium">Game {clip.game_index + 1}</span>
                <span className="text-xs text-muted-foreground">{clip.num_moves} moves, {clip.num_frames} frames</span>
              </div>
              <div className="text-xs font-mono text-muted-foreground">{clip.filepath}</div>
            </div>
          ))}
        </div>
      )}

      {/* Done state */}
      {job?.status === "done" && generatedClips.length === 0 && (
        <p className="text-xs text-muted-foreground">Generation complete but no clips were produced.</p>
      )}

      {/* Failed state */}
      {job?.status === "failed" && (
        <div className="border border-destructive/50 rounded-lg p-3 text-xs text-destructive">
          {job.error || "Generation failed"}
        </div>
      )}
    </div>
  );
}
