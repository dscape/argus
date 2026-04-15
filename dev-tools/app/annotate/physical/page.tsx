"use client";

import { CheckCircle2 } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { ClipGallery } from "@/components/data/ClipGallery";
import { listPhysicalEvalClips, listPhysicalTrainClips, type PhysicalEvalClip } from "@/lib/api";
import type { SyntheticClipFile } from "@/lib/types";

type PhysicalAnnotationSplit = "val" | "train";
type StatusFilter = "all" | "complete" | "incomplete";
type FrameFilter = "all" | "lt100" | "lt500" | "gt500";

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
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [frameFilter, setFrameFilter] = useState<FrameFilter>("all");

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

  const filteredClips = useMemo(() => clips.filter((clip) => {
    if (statusFilter === "complete" && !clip.fully_annotated) return false;
    if (statusFilter === "incomplete" && clip.fully_annotated) return false;
    const nf = clip.num_frames ?? 0;
    if (frameFilter === "lt100" && nf >= 100) return false;
    if (frameFilter === "lt500" && nf >= 500) return false;
    if (frameFilter === "gt500" && nf < 500) return false;
    return true;
  }), [clips, statusFilter, frameFilter]);

  const galleryClips = useMemo(
    () => filteredClips.map((clip) => ({ filename: clip.filename, size_mb: clip.size_mb, modified: clip.modified_at })),
    [filteredClips],
  );

  const clipMetaByFilename = useMemo(() => {
    const map = new Map<string, PhysicalEvalClip>();
    for (const clip of clips) map.set(clip.filename, clip);
    return map;
  }, [clips]);

  const renderOverlay = useCallback((clip: SyntheticClipFile) => {
    const meta = clipMetaByFilename.get(clip.filename);
    if (!meta) return null;
    return (
      <>
        {meta.fully_annotated && (
          <CheckCircle2 className="absolute left-1.5 top-1.5 h-4 w-4 text-green-500 drop-shadow" />
        )}
        {meta.annotated_frame_count > 0 && !meta.fully_annotated && (
          <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1 py-0.5 text-[9px] font-medium text-white">
            {meta.annotated_frame_count}/{meta.num_frames ?? "?"}
          </span>
        )}
      </>
    );
  }, [clipMetaByFilename]);

  const completedCount = useMemo(() => clips.filter((c) => c.fully_annotated).length, [clips]);

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading clips...</div>;
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
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Status</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
            className="h-9 rounded-md border bg-background px-3 text-sm shadow-sm outline-none transition-colors focus:border-ring"
          >
            <option value="all">All</option>
            <option value="incomplete">Incomplete</option>
            <option value="complete">Complete</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Frames</span>
          <select
            value={frameFilter}
            onChange={(e) => setFrameFilter(e.target.value as FrameFilter)}
            className="h-9 rounded-md border bg-background px-3 text-sm shadow-sm outline-none transition-colors focus:border-ring"
          >
            <option value="all">All</option>
            <option value="lt100">&lt; 100</option>
            <option value="lt500">&lt; 500</option>
            <option value="gt500">&gt; 500</option>
          </select>
        </label>
        <span className="text-sm text-muted-foreground">
          {filteredClips.length}/{clips.length} clips ({completedCount} complete)
        </span>
      </div>

      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer hover:text-foreground">How to add more clips</summary>
        <p className="mt-1 pl-4">
          Open a video from the <strong>Videos</strong> page, use the <strong>Calibrate</strong> tab to
          auto-calibrate clips, then go to the <strong>Inspect</strong> tab and click <strong>Save to training</strong> on
          clips you want to annotate. They will appear here.
        </p>
      </details>

      <ClipGallery
        clips={galleryClips}
        directory="data/argus/train_real"
        detailBasePath="/annotate/physical"
        detailQueryString={`?split=${split}`}
        renderOverlay={renderOverlay}
      />
    </div>
  );
}
