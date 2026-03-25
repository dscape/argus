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

export default function SyntheticPage() {
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

  // Auto-refresh stats every 20 clips
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
      setValidCount((c) => c + 1);
    } else {
      setInvalidCount((c) => c + 1);
    }
  }, []);

  // ── Generation state ────────────────────────────────────
  const [genStatus, setGenStatus] = useState<GenerationStatus>({ status: "idle" });
  const [genLoading, setGenLoading] = useState(false);

  useEffect(() => {
    getGenerationStatus().then(setGenStatus).catch(() => {});
  }, []);

  // Poll while running
  useEffect(() => {
    if (genStatus.status !== "running") return;
    const id = setInterval(() => {
      getGenerationStatus().then(setGenStatus).catch(() => {});
    }, 2000);
    return () => clearInterval(id);
  }, [genStatus.status]);

  // Toast on generation failure
  useEffect(() => {
    if (genStatus.status === "failed") {
      toast.error(genStatus.error || "Generation failed");
    }
  }, [genStatus.status, genStatus.error]);

  const handleGenerate10 = async () => {
    setGenLoading(true);
    try {
      const s = await startGeneration({ num_clips: 10, output_dir: DIRECTORY });
      setGenStatus(s);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGenLoading(false);
    }
  };

  const handleStopGeneration = async () => {
    try {
      const s = await stopGeneration();
      setGenStatus(s);
    } catch {
      // ignore
    }
  };

  const isGenerating = genStatus.status === "running";
  const genCompleted = genStatus.completed ?? 0;
  const genTotal = genStatus.num_clips ?? 0;

  // Initial stats load
  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Synthetic</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {/* Generation control — pill button matching app style */}
          {isGenerating ? (
            <button
              onClick={handleStopGeneration}
              className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-all duration-150"
            >
              Stop ({genCompleted}/{genTotal})
            </button>
          ) : (
            <button
              onClick={handleGenerate10}
              disabled={genLoading}
              className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
            >
              {genLoading ? "Starting..." : "Generate 10"}
            </button>
          )}
          <span className="text-muted-foreground/40">|</span>
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              isPolling ? "bg-green-500 animate-pulse" : "bg-gray-400"
            }`}
          />
          <span>{isPolling ? "Live" : "Paused"}</span>
          <button
            onClick={() => setPolling(!isPolling)}
            className="text-xs underline ml-1"
          >
            {isPolling ? "Pause" : "Resume"}
          </button>
        </div>
      </div>

      {/* Stats — always visible */}
      <div className="mb-6">
        {statsLoading || !stats ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: 8 }, (_, i) => (
              <Card key={i}>
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
          <>
            <div className="flex items-center gap-2 mb-2">
              <button
                onClick={refreshStats}
                disabled={statsLoading}
                className="text-xs text-muted-foreground underline hover:text-foreground"
              >
                Refresh stats
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <StatCard label="Clips" value={stats.clip_count} />
              <StatCard label="Total frames" value={stats.total_frames} />
              <StatCard
                label="Avg frames/clip"
                value={stats.avg_frames_per_clip}
              />
              <StatCard label="Total moves" value={stats.total_moves} />
              <StatCard
                label="Avg moves/clip"
                value={stats.avg_moves_per_clip}
              />
              <StatCard
                label="Avg file size"
                value={`${stats.avg_file_size_mb} MB`}
              />
              <StatCard
                label="Total size"
                value={`${stats.total_size_mb} MB`}
              />
              {stats.avg_legal_moves !== null && (
                <StatCard
                  label="Avg legal moves"
                  value={stats.avg_legal_moves}
                />
              )}
              {stats.image_size && (
                <StatCard
                  label="Image size"
                  value={`${stats.image_size[0]}x${stats.image_size[1]}`}
                />
              )}
              {stats.clip_length !== null && (
                <StatCard
                  label="Clip length"
                  value={`${stats.clip_length} frames`}
                />
              )}
              {(validCount > 0 || invalidCount > 0) && (
                <ValidityStatCard valid={validCount} invalid={invalidCount} />
              )}
            </div>
          </>
        )}
      </div>

      {/* Gallery */}
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
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 8 }, (_, i) => (
            <div
              key={i}
              className="rounded-lg border bg-card overflow-hidden"
            >
              <Skeleton className="aspect-square w-full rounded-none" />
              <div className="p-2 space-y-1">
                <Skeleton className="h-3 w-3/4" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Fullscreen inspector */}
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

// ── Stat Cards ────────────────────────────────────────────────

function StatCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
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

function ValidityStatCard({ valid, invalid }: { valid: number; invalid: number }) {
  const total = valid + invalid;
  if (total === 0) return null;
  const pct = Math.round((valid / total) * 100);
  return (
    <Card>
      <CardHeader className="pb-1 pt-3 px-4">
        <CardDescription className="text-xs">Clip validity</CardDescription>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        <p className="text-xl font-bold">{pct}%</p>
        <div className="mt-1 h-1.5 rounded-full bg-red-500/20 overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">
          {valid} valid / {invalid} invalid
        </p>
      </CardContent>
    </Card>
  );
}
