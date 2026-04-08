"use client";

import { useState, useEffect } from "react";
import { youtubeThumb } from "@/components/video-shared";
import VideoCard, { computeAgreement, type InspectResult } from "@/components/evaluate/VideoCard";
import { useVideoWorkbench } from "../_context";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs text-muted-foreground w-24 flex-shrink-0">{label}</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}

export default function InfoPage() {
  const { video, downloadStatus } = useVideoWorkbench();
  const [aiResult, setAiResult] = useState<InspectResult | null>(null);
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    if (!video) return;
    let cancelled = false;
    (async () => {
      setAiLoading(true);
      try {
        const res = await fetch("/api/models/ai-screening/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: video.video_id }),
        });
        if (!res.ok) throw new Error(await res.text());
        if (!cancelled) setAiResult(await res.json());
      } catch {
        // silently fail
      }
      if (!cancelled) setAiLoading(false);
    })();
    return () => { cancelled = true; };
  }, [video?.video_id]);

  if (!video) return null;

  const fmtDuration = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const agreement = aiResult ? computeAgreement(aiResult) : null;
  const showAiResult = aiResult && (agreement === false || agreement === null);

  return (
    <div className="space-y-4 pt-2 max-w-4xl">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Field label="Video ID" value={video.video_id} />
          <Field label="Channel" value={video.channel_handle || "\u2014"} />
          <Field label="Published" value={video.published_at ? new Date(video.published_at).toLocaleDateString() : "\u2014"} />
          <Field label="Duration" value={downloadStatus?.duration_seconds != null ? fmtDuration(downloadStatus.duration_seconds) : "\u2014"} />
          <Field label="File Size" value={downloadStatus?.file_size_mb != null ? `${downloadStatus.file_size_mb} MB` : "\u2014"} />
          <Field label="Status" value={video.screening_status || "unscreened"} />
          <Field label="Layout" value={video.layout_type || "\u2014"} />
          <div>
            <a
              href={`https://www.youtube.com/watch?v=${video.video_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:underline"
            >
              Open on YouTube &rarr;
            </a>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-1">
          {[1, 2, 3].map((i) => (
            <img
              key={i}
              src={youtubeThumb(video.video_id, i)}
              alt={`${["", "25%", "50%", "75%"][i]}`}
              className="w-full aspect-video object-cover rounded border"
            />
          ))}
        </div>
      </div>

      <div className="border-t pt-4">
        <div className="flex items-center gap-2 mb-2">
          <h3 className="text-sm font-medium">Screening</h3>
          {aiLoading && <span className="text-xs text-muted-foreground">Running AI...</span>}
          {agreement === true && (
            <span className="text-xs text-green-600 font-medium">AI agrees</span>
          )}
        </div>
        {showAiResult && <VideoCard result={aiResult} />}
      </div>
    </div>
  );
}
