"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getSyntheticStats,
  startGeneration,
  getGenerationStatus,
  stopGeneration,
} from "@/lib/api";
import type {
  SyntheticStatsResponse,
  ClipInspectResponse,
  GenerationStatus,
} from "@/lib/types";
import { ClipGallery } from "@/components/ClipGallery";
import { ClipInspector } from "@/components/ClipInspector";
import { usePolledScan } from "@/hooks/usePolledScan";
import { toast } from "sonner";

const DIRECTORY = "data/train";

export default function DataPage() {
  const [stats, setStats] = useState<SyntheticStatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const lastStatsRefreshCount = useRef(0);

  const refreshStats = useCallback(() => {
    setStatsLoading(true);
    getSyntheticStats(DIRECTORY)
      .then((result) => setStats(result))
      .catch(() => {})
      .finally(() => setStatsLoading(false));
  }, []);

  const handleClipCountChange = useCallback(
    (count: number) => {
      const prev = lastStatsRefreshCount.current;
      if (count === 0 || Math.floor(count / 20) > Math.floor(prev / 20)) {
        lastStatsRefreshCount.current = count;
        refreshStats();
      }
    },
    [refreshStats]
  );

  const {
    scan,
    error: scanError,
    isPolling,
    setPolling,
  } = usePolledScan({
    directory: DIRECTORY,
    intervalMs: 2000,
    onClipCountChange: handleClipCountChange,
  });

  useEffect(() => {
    if (scanError) toast.error(scanError);
  }, [scanError]);

  const [validCount, setValidCount] = useState(0);
  const [invalidCount, setInvalidCount] = useState(0);

  const [inspectedClip, setInspectedClip] = useState<{
    filename: string;
    sessionId: string;
    clipInfo: ClipInspectResponse;
  } | null>(null);

  const handleClipInspected = useCallback((clipInfo: ClipInspectResponse) => {
    if (clipInfo.replay_valid) {
      setValidCount((count) => count + 1);
      return;
    }

    setInvalidCount((count) => count + 1);
  }, []);

  const [genStatus, setGenStatus] = useState<GenerationStatus>({ status: "idle" });
  const [genLoading, setGenLoading] = useState(false);

  useEffect(() => {
    getGenerationStatus().then(setGenStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (genStatus.status !== "running") return;

    const id = setInterval(() => {
      getGenerationStatus().then(setGenStatus).catch(() => {});
    }, 2000);

    return () => clearInterval(id);
  }, [genStatus.status]);

  useEffect(() => {
    if (genStatus.status === "failed") {
      toast.error(genStatus.error || "Generation failed");
    }
  }, [genStatus.status, genStatus.error]);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  const handleGenerate10 = async () => {
    setGenLoading(true);
    try {
      const status = await startGeneration({ num_clips: 10, output_dir: DIRECTORY });
      setGenStatus(status);
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : "Generation failed");
    } finally {
      setGenLoading(false);
    }
  };

  const handleStopGeneration = async () => {
    try {
      const status = await stopGeneration();
      setGenStatus(status);
    } catch {
      // ignore
    }
  };

  const isGenerating = genStatus.status === "running";
  const genCompleted = genStatus.completed ?? 0;
  const genTotal = genStatus.num_clips ?? 0;

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold">Synthetic</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {isGenerating ? (
            <button
              onClick={handleStopGeneration}
              className="flex h-8 items-center gap-1.5 rounded-xl bg-destructive px-3 text-xs font-medium text-destructive-foreground transition-all duration-150 hover:bg-destructive/90"
            >
              Stop ({genCompleted}/{genTotal})
            </button>
          ) : (
            <button
              onClick={handleGenerate10}
              disabled={genLoading}
              className="flex h-8 items-center gap-1.5 rounded-xl bg-secondary px-3 text-xs font-medium text-secondary-foreground transition-all duration-150 hover:bg-secondary/80 disabled:pointer-events-none disabled:opacity-50"
            >
              {genLoading ? "Starting..." : "Generate 10"}
            </button>
          )}
          <span className="text-muted-foreground/40">|</span>
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              isPolling ? "animate-pulse bg-green-500" : "bg-gray-400"
            }`}
          />
          <span>{isPolling ? "Live" : "Paused"}</span>
          <button onClick={() => setPolling(!isPolling)} className="ml-1 text-xs underline">
            {isPolling ? "Pause" : "Resume"}
          </button>
        </div>
      </div>

      <div className="mb-6">
        {statsLoading || !stats ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {Array.from({ length: 8 }, (_, index) => (
              <Card key={index}>
                <CardHeader className="px-4 pb-1 pt-3">
                  <Skeleton className="h-3 w-20" />
                </CardHeader>
                <CardContent className="px-4 pb-3">
                  <Skeleton className="mt-1 h-6 w-16" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <>
            <div className="mb-2 flex items-center gap-2">
              <button
                onClick={refreshStats}
                disabled={statsLoading}
                className="text-xs text-muted-foreground underline hover:text-foreground"
              >
                Refresh stats
              </button>
            </div>
            <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatCard label="Clips" value={stats.clip_count} />
              <StatCard label="Total frames" value={stats.total_frames} />
              <StatCard label="Avg frames/clip" value={stats.avg_frames_per_clip} />
              <StatCard label="Total moves" value={stats.total_moves} />
              <StatCard label="Avg moves/clip" value={stats.avg_moves_per_clip} />
              <StatCard label="Avg file size" value={`${stats.avg_file_size_mb} MB`} />
              <StatCard label="Total size" value={`${stats.total_size_mb} MB`} />
              {stats.avg_legal_moves !== null && (
                <StatCard label="Avg legal moves" value={stats.avg_legal_moves} />
              )}
              {stats.image_size && (
                <StatCard
                  label="Image size"
                  value={`${stats.image_size[0]}x${stats.image_size[1]}`}
                />
              )}
              {stats.clip_length !== null && (
                <StatCard label="Clip length" value={`${stats.clip_length} frames`} />
              )}
              {(validCount > 0 || invalidCount > 0) && (
                <ValidityStatCard valid={validCount} invalid={invalidCount} />
              )}
            </div>
          </>
        )}
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
          onClipInspected={handleClipInspected}
        />
      ) : !scanError ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {Array.from({ length: 8 }, (_, index) => (
            <div key={index} className="overflow-hidden rounded-lg border bg-card">
              <Skeleton className="aspect-square w-full rounded-none" />
              <div className="space-y-1 p-2">
                <Skeleton className="h-3 w-3/4" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {inspectedClip && (
        <ClipInspector
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
      <CardHeader className="px-4 pb-1 pt-3">
        <CardDescription className="text-xs">{label}</CardDescription>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        <p className="text-xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}

function ValidityStatCard({ valid, invalid }: { valid: number; invalid: number }) {
  const total = valid + invalid;
  if (total === 0) return null;

  const pct = Math.round((valid / total) * 100);

  return (
    <Card>
      <CardHeader className="px-4 pb-1 pt-3">
        <CardDescription className="text-xs">Clip validity</CardDescription>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        <p className="text-xl font-bold">{pct}%</p>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-red-500/20">
          <div
            className="h-full rounded-full bg-green-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {valid} valid / {invalid} invalid
        </p>
      </CardContent>
    </Card>
  );
}
