"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ClipGallery } from "@/components/data/ClipGallery";
import { RealClipInspector } from "@/components/data/RealClipInspector";
import { usePolledScan } from "@/hooks/usePolledScan";
import {
  getRealDataOverview,
  getRealVideoProcessingStatus,
  startRealVideoProcessing,
  stopRealVideoProcessing,
} from "@/lib/api";
import type {
  ClipInspectResponse,
  RealDataOverview,
  RealVideoProcessingStatus,
} from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

const CLIPS_DIR = "data/argus/train_real";
const PROCESS_LIMIT = 10;

export function RealDataPane() {
  const [overview, setOverview] = useState<RealDataOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [jobStatus, setJobStatus] = useState<RealVideoProcessingStatus>({ status: "idle" });
  const [jobLoading, setJobLoading] = useState(false);
  const [inspectedClip, setInspectedClip] = useState<{
    filename: string;
    sessionId: string;
    clipInfo: ClipInspectResponse;
  } | null>(null);
  const lastRefreshClipCount = useRef(0);

  const refreshOverview = useCallback(() => {
    setOverviewLoading(true);
    getRealDataOverview({
      clipsDir: CLIPS_DIR,
      limit: 100,
    })
      .then(setOverview)
      .catch((e: unknown) => {
        toast.error(e instanceof Error ? e.message : "Failed to load real data overview");
      })
      .finally(() => setOverviewLoading(false));
  }, []);

  const handleClipCountChange = useCallback(
    (count: number) => {
      const prev = lastRefreshClipCount.current;
      if (count === 0 || count !== prev) {
        lastRefreshClipCount.current = count;
        refreshOverview();
      }
    },
    [refreshOverview]
  );

  const {
    scan,
    error: scanError,
    isPolling,
    setPolling,
  } = usePolledScan({
    directory: CLIPS_DIR,
    intervalMs: 2000,
    onClipCountChange: handleClipCountChange,
  });

  useEffect(() => {
    if (scanError) toast.error(scanError);
  }, [scanError]);

  useEffect(() => {
    refreshOverview();
    getRealVideoProcessingStatus().then(setJobStatus).catch(() => {});
  }, [refreshOverview]);

  useEffect(() => {
    if (jobStatus.status !== "running") return;
    const id = setInterval(() => {
      getRealVideoProcessingStatus().then(setJobStatus).catch(() => {});
    }, 2000);
    return () => clearInterval(id);
  }, [jobStatus.status]);

  useEffect(() => {
    if (jobStatus.status === "done" || jobStatus.status === "stopped") {
      refreshOverview();
    }
  }, [jobStatus.status, refreshOverview]);

  useEffect(() => {
    if (jobStatus.status === "failed" && jobStatus.error) {
      toast.error(jobStatus.error);
    }
  }, [jobStatus.status, jobStatus.error]);

  const handleProcess = async () => {
    setJobLoading(true);
    try {
      const status = await startRealVideoProcessing({
        limit: PROCESS_LIMIT,
        clipsDir: CLIPS_DIR,
        minMoves: 5,
      });
      setJobStatus(status);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to start processing");
    } finally {
      setJobLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      const status = await stopRealVideoProcessing();
      setJobStatus(status);
    } catch {
      // ignore
    }
  };

  const clipStats = overview?.clip_stats;
  const isProcessing = jobStatus.status === "running";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold">Real-footage clips</h3>
          <p className="text-sm text-muted-foreground">
            Process downloaded local videos into training clips under <code>{CLIPS_DIR}</code>.
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
          {isProcessing ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-all duration-150"
            >
              Stop ({jobStatus.completed_videos ?? 0}/{jobStatus.total_videos ?? 0})
            </button>
          ) : (
            <button
              onClick={handleProcess}
              disabled={jobLoading}
              className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
            >
              {jobLoading ? "Starting..." : "Generate more"}
            </button>
          )}
          <button
            onClick={refreshOverview}
            disabled={overviewLoading}
            className="text-xs underline hover:text-foreground"
          >
            Refresh
          </button>
          <span className="text-muted-foreground/40">|</span>
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              isPolling ? "bg-green-500 animate-pulse" : "bg-gray-400"
            }`}
          />
          <span>{isPolling ? "Live" : "Paused"}</span>
          <button onClick={() => setPolling(!isPolling)} className="text-xs underline ml-1">
            {isPolling ? "Pause" : "Resume"}
          </button>
        </div>
      </div>

      {isProcessing && (
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Processing real videos</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge variant="secondary">
                {jobStatus.completed_videos ?? 0}/{jobStatus.total_videos ?? 0} videos
              </Badge>
              <Badge variant="outline">{jobStatus.generated_clips ?? 0} clips generated</Badge>
              {jobStatus.current_video_id && (
                <span className="text-muted-foreground">
                  Current: {jobStatus.current_video_title || jobStatus.current_video_id}
                </span>
              )}
            </div>
            {jobStatus.results && jobStatus.results.length > 0 && (
              <div className="space-y-1">
                {jobStatus.results.slice().reverse().slice(0, 5).map((result) => (
                  <div key={result.video_id} className="text-xs text-muted-foreground">
                    <span className="font-medium text-foreground mr-2">{result.video_id}</span>
                    {result.status === "generated"
                      ? `generated ${result.generated_clip_count} clip(s)`
                      : result.status === "no_clips"
                        ? "processed but produced no legal clips"
                        : result.error || "failed"}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {overviewLoading || !overview || !clipStats ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Array.from({ length: 4 }, (_, index) => (
            <Card key={index}>
              <CardHeader className="pb-1 pt-3 px-4">
                <Skeleton className="h-3 w-20" />
              </CardHeader>
              <CardContent className="px-4 pb-3">
                <Skeleton className="h-6 w-16 mt-1" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Real clips" value={clipStats.clip_count} />
          <StatCard label="Source videos" value={overview.source_video_count} />
          <StatCard label="Total moves" value={clipStats.total_moves} />
          <StatCard label="Total size" value={`${clipStats.total_size_mb} MB`} />
        </div>
      )}

      <div className="space-y-3">
        <div>
          <h4 className="text-sm font-medium">Generated real clips</h4>
          <p className="text-sm text-muted-foreground">
            Existing .pt clips available for inspection and spot-checking.
          </p>
        </div>
        {scan ? (
          <ClipGallery
            clips={scan.clips}
            directory={scan.directory}
            onClipClick={(clip, sessionId, clipInfo) =>
              setInspectedClip({
                filename: clip.filename,
                sessionId,
                clipInfo,
              })
            }
          />
        ) : !scanError ? (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {Array.from({ length: 8 }, (_, index) => (
              <div key={index} className="rounded-lg border bg-card overflow-hidden">
                <Skeleton className="aspect-square w-full rounded-none" />
                <div className="p-2 space-y-1">
                  <Skeleton className="h-3 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {inspectedClip && (
        <RealClipInspector
          open={!!inspectedClip}
          onClose={() => setInspectedClip(null)}
          filename={inspectedClip.filename}
          sessionId={inspectedClip.sessionId}
          clipInfo={inspectedClip.clipInfo}
        />
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardHeader className="pb-1 pt-3 px-4">
        <CardDescription className="text-xs">{label}</CardDescription>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        <p className="text-xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}
