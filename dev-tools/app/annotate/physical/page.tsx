"use client";

import { CheckCircle2 } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { listPhysicalEvalClips, listPhysicalTrainClips, type PhysicalEvalClip } from "@/lib/api";

type PhysicalAnnotationSplit = "val" | "train";

function normalizeSplit(value: string | null): PhysicalAnnotationSplit {
  return value === "train" ? "train" : "val";
}

function buildSplitUrl(
  pathname: string,
  searchParams: ReturnType<typeof useSearchParams>,
  split: PhysicalAnnotationSplit,
): string {
  const params = new URLSearchParams(searchParams.toString());
  params.set("split", split);
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
}

export default function PhysicalAnnotationIndexPage() {
  const pathname = usePathname();
  const router = useRouter();
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
    ? "No training clips found in data/argus/train_real."
    : "No validation clips found in data/argus/train_real.";

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading clips…</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 rounded-2xl border bg-muted/30 p-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Split</span>
          <select
            value={split}
            onChange={(event) => {
              router.replace(
                buildSplitUrl(pathname, searchParams, normalizeSplit(event.target.value)),
              );
            }}
            className="h-9 rounded-md border bg-background px-3 text-sm shadow-sm outline-none transition-colors focus:border-ring"
          >
            <option value="train">Train</option>
            <option value="val">Validation</option>
          </select>
        </label>
      </div>

      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          {clips.length} {split === "train" ? "training" : "validation"} clips in <code>data/argus/train_real</code>
        </p>
        <p>
          Source videos are assigned once with a stable pseudo-random 80/20 train/val split.
        </p>
      </div>

      {clips.length === 0 ? (
        <div className="text-sm text-muted-foreground">{emptyMessage}</div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {clips.map((clip) => {
            const splitBadge = clip.assigned_split ?? "unknown";
            return (
              <Link
                key={clip.clip_path}
                href={`/annotate/physical/${encodeURIComponent(clip.filename)}?split=${split}`}
                className="block rounded border p-3 text-sm transition-colors hover:bg-muted/40"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-mono font-medium">{clip.filename}</div>
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
