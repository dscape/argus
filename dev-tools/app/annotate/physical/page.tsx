"use client";

import { CheckCircle2 } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { listPhysicalEvalClips, listPhysicalTrainClips, type PhysicalEvalClip } from "@/lib/api";

type PhysicalAnnotationSplit = "val" | "train";

function normalizeSplit(value: string | null): PhysicalAnnotationSplit {
  return value === "train" ? "train" : "val";
}

export default function PhysicalAnnotationIndexPage() {
  const searchParams = useSearchParams();
  const split = normalizeSplit(searchParams.get("split"));
  const [clips, setClips] = useState<PhysicalEvalClip[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const data = split === "train"
          ? await listPhysicalTrainClips()
          : await listPhysicalEvalClips();
        if (!cancelled) {
          setClips(data.clips);
        }
      } catch (error) {
        if (!cancelled) {
          toast.error(error instanceof Error ? error.message : "Failed to load clips");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [split]);

  const splitLabel = split === "train" ? "Train" : "Validation";
  const emptyMessage = split === "train"
    ? "No eligible training clips found in data/argus/train_real."
    : "No eligible validation clips found in data/argus/train_real.";
  const helperText = split === "train"
    ? `${clips.length} training-eligible clips in `
    : `${clips.length} validation-eligible clips in `;

  const splitDescription = useMemo(() => {
    if (split === "train") {
      return "Shows clips whose source video is assigned to train or still unassigned.";
    }
    return "Shows clips whose source video is assigned to val or still unassigned.";
  }, [split]);

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading clips…</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button asChild variant={split === "val" ? "default" : "outline"}>
          <Link href="/annotate/physical?split=val">Validation</Link>
        </Button>
        <Button asChild variant={split === "train" ? "default" : "outline"}>
          <Link href="/annotate/physical?split=train">Train</Link>
        </Button>
      </div>

      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          {helperText}
          <code>data/argus/train_real</code>
        </p>
        <p>{splitDescription}</p>
      </div>

      {clips.length === 0 ? (
        <div className="text-sm text-muted-foreground">{emptyMessage}</div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {clips.map((clip) => {
            const splitBadge = clip.assigned_split ?? "unassigned";
            return (
              <Link
                key={clip.clip_path}
                href={`/annotate/physical/${encodeURIComponent(clip.filename)}?split=${split}`}
                className="rounded border p-3 text-sm transition-colors hover:bg-muted/40 block"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="font-medium font-mono truncate">{clip.filename}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {clip.source_video_id ?? "unknown"} · clip {clip.clip_id ?? "-"} · {clip.size_mb.toFixed(1)} MB
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      split: <span className="font-medium text-foreground">{splitBadge}</span>
                    </div>
                    {clip.annotated_frame_count > 0 && (
                      <div className="mt-1 text-xs text-muted-foreground">
                        {clip.num_frames !== null
                          ? `${clip.annotated_frame_count}/${clip.num_frames} frames saved`
                          : `${clip.annotated_frame_count} frames saved`}
                      </div>
                    )}
                  </div>
                  {clip.fully_annotated ? (
                    <CheckCircle2
                      className="h-4 w-4 shrink-0 text-green-600"
                      aria-label={`Fully annotated ${splitLabel.toLowerCase()} clip`}
                    />
                  ) : null}
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
