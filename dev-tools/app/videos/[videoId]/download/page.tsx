"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { SpinnerIcon } from "@/components/video-shared";
import { useVideoWorkbench } from "../_context";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs text-muted-foreground w-24 flex-shrink-0">{label}</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}

function AssetRow({ label, ready, loading }: { label: string; ready: boolean; loading: boolean }) {
  return (
    <div className="flex items-center gap-2">
      {ready ? (
        <span className="w-4 h-4 rounded-full bg-green-500 text-white text-[10px] flex items-center justify-center">&#10003;</span>
      ) : loading ? (
        <SpinnerIcon className="w-4 h-4 animate-spin text-muted-foreground" />
      ) : (
        <span className="w-4 h-4 rounded-full border border-muted-foreground text-[10px] flex items-center justify-center">&mdash;</span>
      )}
      <span className="text-sm">{label}</span>
    </div>
  );
}

export default function DownloadPage() {
  const router = useRouter();
  const {
    video,
    videoId,
    assets,
    assetsLoading,
    downloadStatus,
    downloadBusy,
    startDownload,
  } = useVideoWorkbench();
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!downloadBusy) return;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [downloadBusy]);

  const handleDownload = async () => {
    setError(null);
    try {
      await startDownload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed");
    }
  };

  const fmtElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  if (!video) return null;

  const loresReady = assets != null && assets.lores.length >= 3;
  const hiresReady = assets != null && assets.hires.length >= 3;
  const videoReady = Boolean(assets?.video || downloadStatus?.downloaded);

  return (
    <div className="space-y-4 pt-2 max-w-xl">
      <div className="space-y-1.5 p-3 rounded-lg border bg-muted/20">
        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Assets</h4>
        <AssetRow label="Low-res frames (480p)" ready={loresReady} loading={assetsLoading && !loresReady} />
        <AssetRow label="Hi-res frames (720p)" ready={hiresReady} loading={assetsLoading && !hiresReady} />
        <AssetRow label="Video file" ready={videoReady} loading={downloadBusy || (assetsLoading && !videoReady)} />
      </div>

      {downloadStatus === null ? (
        <p className="text-sm text-muted-foreground">Checking download status...</p>
      ) : downloadStatus.downloaded ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-sm font-medium">Downloaded</span>
          </div>
          <Field label="Path" value={downloadStatus.path!} />
          <Field label="Size" value={`${downloadStatus.file_size_mb} MB`} />
          <button
            onClick={() => router.push(`/videos/${videoId}/segment`)}
            className="text-sm text-primary hover:underline"
          >
            Continue to Segment &rarr;
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-yellow-500" />
            <span className="text-sm">Not downloaded yet</span>
          </div>
          {downloadBusy ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: "100%" }} />
                </div>
                <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">{fmtElapsed(elapsed)}</span>
              </div>
              <p className="text-xs text-muted-foreground">Downloading via yt-dlp...</p>
            </div>
          ) : (
            <button
              onClick={handleDownload}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90"
            >
              Download Video
            </button>
          )}
        </div>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
