"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  listGeneratedClips,
  getClipInfo,
  clipFrameUrl,
  saveClipToTraining,
} from "@/lib/api";
import type { ClipInspectResponse } from "@/lib/types";
import { useVideoWorkbench } from "../_context";

interface GeneratedClipFile {
  filepath: string;
  filename: string;
}

export default function InspectPage() {
  const { session, video } = useVideoWorkbench();
  const [clips, setClips] = useState<GeneratedClipFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<Map<string, ClipInspectResponse>>(new Map());
  const [clipSessions, setClipSessions] = useState<Map<string, string>>(new Map());
  const [frameIndices, setFrameIndices] = useState<Map<string, number>>(new Map());
  const [saving, setSaving] = useState<string | null>(null);
  const [saved, setSaved] = useState<Set<string>>(new Set());

  // Load generated clips when session is available
  useEffect(() => {
    if (!session) return;
    setLoading(true);
    listGeneratedClips(session.session_id)
      .then(({ clips: c }) => setClips(c))
      .catch((e) => toast.error(e instanceof Error ? e.message : "Failed to list clips"))
      .finally(() => setLoading(false));
  }, [session]);

  const handleInspect = async (clip: GeneratedClipFile) => {
    if (!session) return;
    setInspecting(clip.filename);
    try {
      // Load clip via server-side path
      const res = await fetch("/api/clips/load-from-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: clip.filepath }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { session_id } = await res.json();
      setClipSessions((prev) => new Map(prev).set(clip.filename, session_id));

      const info = await getClipInfo(session_id);
      if (info.replay_error) toast.error(info.replay_error);
      setClipInfo((prev) => new Map(prev).set(clip.filename, info));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to inspect clip");
    } finally {
      setInspecting(null);
    }
  };

  const handleSave = async (clip: GeneratedClipFile) => {
    if (!session) return;
    setSaving(clip.filename);
    try {
      await saveClipToTraining(session.session_id, clip.filepath);
      setSaved((prev) => new Set(prev).add(clip.filename));
      toast.success(`Saved ${clip.filename} to training data`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save clip");
    } finally {
      setSaving(null);
    }
  };

  if (!video) return null;

  if (!session) {
    return <p className="text-sm text-muted-foreground pt-2">Opening video session...</p>;
  }

  return (
    <div className="space-y-4 pt-2 max-w-4xl">
      <p className="text-sm text-muted-foreground">
        Inspect generated clips and save confirmed ones to training data.
      </p>

      {loading && (
        <p className="text-xs text-muted-foreground">Scanning for generated clips...</p>
      )}

      {!loading && clips.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No clips found for this video. Run clip generation first (Step 6).
        </p>
      )}

      {clips.map((clip) => {
        const info = clipInfo.get(clip.filename);
        const sessionId = clipSessions.get(clip.filename);
        const currentFrame = frameIndices.get(clip.filename) ?? 0;
        const isSaved = saved.has(clip.filename);

        return (
          <div key={clip.filename} className="border rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium font-mono">{clip.filename}</span>
              <button
                onClick={() => handleInspect(clip)}
                disabled={inspecting === clip.filename}
                className="text-xs text-primary hover:underline"
              >
                {inspecting === clip.filename ? "Loading..." : info ? "Re-inspect" : "Inspect"}
              </button>
              <button
                onClick={() => handleSave(clip)}
                disabled={saving === clip.filename || isSaved}
                className={`text-xs px-2 py-0.5 rounded ${
                  isSaved
                    ? "bg-green-500/20 text-green-700"
                    : "bg-primary/10 text-primary hover:bg-primary/20"
                } disabled:opacity-50`}
              >
                {isSaved ? "Saved" : saving === clip.filename ? "Saving..." : "Save to training"}
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
                  <div><span className="text-muted-foreground">Frames:</span> {info.num_frames}</div>
                  <div><span className="text-muted-foreground">Avg legal:</span> {info.avg_legal_moves?.toFixed(1)}</div>
                </div>

                <div>
                  <input
                    type="range"
                    min={0}
                    max={info.num_frames - 1}
                    value={currentFrame}
                    onChange={(e) => setFrameIndices((prev) => new Map(prev).set(clip.filename, Number(e.target.value)))}
                    className="w-full"
                  />
                  <div className="flex gap-3">
                    <img
                      src={clipFrameUrl(sessionId, currentFrame)}
                      alt={`Frame ${currentFrame}`}
                      className="w-56 rounded border"
                    />
                    <div className="space-y-2">
                      <div className="text-xs">Frame {currentFrame} / {info.num_frames}</div>
                      {info.moves.filter((m) => m.frame_index === currentFrame).map((m) => (
                        <div key={m.frame_index}>
                          <div className="text-xs font-medium text-primary">
                            Move: {m.san || m.uci}
                          </div>
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
