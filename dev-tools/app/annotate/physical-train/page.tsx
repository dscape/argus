"use client";

import { CheckCircle2 } from "lucide-react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";

import { listPhysicalTrainClips, type PhysicalEvalClip } from "@/lib/api";

export default function PhysicalTrainAnnotationIndexPage() {
  const [clips, setClips] = useState<PhysicalEvalClip[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        const data = await listPhysicalTrainClips();
        setClips(data.clips);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to load clips");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading clips&hellip;</div>;
  }

  if (clips.length === 0) {
    return <div className="text-sm text-muted-foreground">No eligible non-held-out clips found in data/argus/train_real.</div>;
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        {clips.length} non-held-out clips in <code>data/argus/train_real</code>
      </p>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {clips.map((clip) => (
          <Link
            key={clip.clip_path}
            href={`/annotate/physical-train/${encodeURIComponent(clip.filename)}`}
            className="rounded border p-3 text-sm hover:bg-muted/40 transition-colors block"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="font-medium font-mono truncate">{clip.filename}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {clip.source_video_id ?? "unknown"} &middot; clip {clip.clip_id ?? "-"} &middot; {clip.size_mb.toFixed(1)} MB
                </div>
                {clip.annotated_frame_count > 0 && (
                  <div className="mt-1 text-xs text-muted-foreground">
                    {clip.num_frames !== null
                      ? `${clip.annotated_frame_count}/${clip.num_frames} frames saved`
                      : `${clip.annotated_frame_count} frames saved`}
                  </div>
                )}
              </div>
              {clip.fully_annotated && (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" aria-label="Fully annotated" />
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
