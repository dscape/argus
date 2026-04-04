"use client";

import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";
import { VideoWorkbenchProvider, useVideoWorkbench } from "./_context";
import { scoreColor, StatusDropdown, SpinnerIcon } from "@/components/video-shared";
import type { VideoWithReason } from "@/components/video-shared";

const STEPS = [
  { id: "info", label: "Info" },
  { id: "download", label: "Download" },
  { id: "segment", label: "Segment" },
  { id: "calibrate", label: "Calibrate" },
  { id: "extract", label: "Extract" },
  { id: "generate", label: "Generate" },
  { id: "inspect", label: "Inspect" },
] as const;

function WorkbenchHeader() {
  const router = useRouter();
  const pathname = usePathname();
  const { videoId, video, downloadReady, assetsLoading, handleStatusChange } = useVideoWorkbench();

  const activeStep = pathname.split("/").pop() || "info";

  const goBack = () => {
    try {
      const saved = sessionStorage.getItem("videoBrowserState");
      if (saved) {
        const s = JSON.parse(saved);
        const params = new URLSearchParams();
        if (s.status && s.status !== "approved:!otb_only") params.set("status", s.status);
        if (s.sort && s.sort !== "published_at_desc") params.set("sort", s.sort);
        if (s.channel) params.set("channel", s.channel);
        if (s.page > 0) params.set("page", String(s.page));
        const qs = params.toString();
        router.push(`/videos/browse${qs ? `?${qs}` : ""}`);
        return;
      }
    } catch { /* ignore */ }
    router.push("/videos/browse");
  };

  if (!video) return null;

  return (
    <div className="flex-shrink-0 px-4 pt-3 pb-2">
      <button onClick={goBack} className="text-xs text-muted-foreground hover:text-foreground mb-1 block">
        &larr; Videos
      </button>
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-lg font-semibold line-clamp-1 flex-1">{video.title}</h2>
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${scoreColor(video.title_score)}`} />
        <StatusDropdown video={video as VideoWithReason} onStatusChange={handleStatusChange} />
      </div>

      <div className="flex items-center gap-1 rounded-2xl border bg-muted/30 p-1">
        {STEPS.map((s, i) => {
          const isDownloadReady = s.id === "download" && downloadReady;
          const isActive = activeStep === s.id;
          return (
            <Link
              key={s.id}
              href={`/videos/${videoId}/${s.id}`}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-150 ${
                isActive
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-background/60"
              }`}
            >
              {isDownloadReady ? (
                <span className="w-4 h-4 rounded-full bg-green-500 text-white text-[10px] flex items-center justify-center font-bold">
                  &#10003;
                </span>
              ) : (
                <span className={`w-4 h-4 rounded-full border text-[10px] flex items-center justify-center font-bold tabular-nums ${
                  s.id === "download" && assetsLoading ? "animate-pulse" : ""
                }`}>
                  {i + 1}
                </span>
              )}
              {s.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}

function WorkbenchContent({ children }: { children: React.ReactNode }) {
  const { video, error } = useVideoWorkbench();
  const router = useRouter();

  if (error) {
    return (
      <div className="p-6">
        <button onClick={() => router.push("/videos/browse")} className="text-sm text-primary hover:underline mb-4 block">
          &larr; Back to Videos
        </button>
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  if (!video) {
    return <div className="p-6 text-sm text-muted-foreground">Loading...</div>;
  }

  return (
    <>
      <WorkbenchHeader />
      <div className="flex-1 overflow-auto px-4 pb-4">
        {children}
      </div>
    </>
  );
}

export default function VideoWorkbenchLayout({ children }: { children: React.ReactNode }) {
  const params = useParams();
  const videoId = params.videoId as string;

  return (
    <VideoWorkbenchProvider videoId={videoId}>
      <div className="h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <WorkbenchContent>{children}</WorkbenchContent>
      </div>
    </VideoWorkbenchProvider>
  );
}
