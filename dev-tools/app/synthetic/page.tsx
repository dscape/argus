"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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

  // Initial stats load
  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Synthetic</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
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

      {scanError && (
        <p className="text-sm text-destructive mb-4">{scanError}</p>
      )}

      {/* Generation Controls */}
      <GenerationControls />

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

// ── Generation Controls ──────────────────────────────────────

function GenerationControls() {
  const [status, setStatus] = useState<GenerationStatus>({ status: "idle" });
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState({
    num_clips: 100,
    output_dir: "data/train",
    image_size: 224,
    clip_length: 16,
    frames_per_move: 4,
    seed: 42,
    quality: "training",
  });

  // Fetch initial status
  useEffect(() => {
    getGenerationStatus().then(setStatus).catch(() => {});
  }, []);

  // Poll status while running
  useEffect(() => {
    if (status.status !== "running") return;
    const id = setInterval(async () => {
      try {
        const s = await getGenerationStatus();
        setStatus(s);
      } catch {
        // ignore poll errors
      }
    }, 2000);
    return () => clearInterval(id);
  }, [status.status]);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await startGeneration(params);
      setStatus(s);
      setShowForm(false);
    } catch (e: any) {
      setError(e.message || "Failed to start generation");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      const s = await stopGeneration();
      setStatus(s);
    } catch (e: any) {
      setError(e.message || "Failed to stop generation");
    } finally {
      setLoading(false);
    }
  };

  const isRunning = status.status === "running";
  const completed = status.completed ?? 0;
  const total = status.num_clips ?? 0;
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Generation</CardTitle>
          {!isRunning && status.status !== "running" && (
            <Button
              size="sm"
              onClick={() => setShowForm(!showForm)}
              variant={showForm ? "outline" : "default"}
            >
              {showForm ? "Cancel" : "Generate"}
            </Button>
          )}
          {isRunning && (
            <Button
              size="sm"
              variant="destructive"
              onClick={handleStop}
              disabled={loading}
            >
              Stop
            </Button>
          )}
        </div>
        {status.status !== "idle" && status.status !== "no_job_running" && (
          <CardDescription className="text-xs mt-1">
            {isRunning
              ? `Generating: ${completed}/${total} clips (${pct}%)`
              : status.status === "done"
              ? `Completed: ${completed} clips generated`
              : status.status === "stopped"
              ? `Stopped at ${completed}/${total} clips`
              : status.status === "failed"
              ? `Failed: ${status.error}`
              : null}
          </CardDescription>
        )}
      </CardHeader>

      {/* Progress bar */}
      {isRunning && total > 0 && (
        <CardContent className="pt-0 pb-3">
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </CardContent>
      )}

      {/* Parameters form */}
      {showForm && !isRunning && (
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <FormField
              label="Clips"
              value={params.num_clips}
              onChange={(v) => setParams({ ...params, num_clips: Number(v) })}
              type="number"
            />
            <FormField
              label="Output dir"
              value={params.output_dir}
              onChange={(v) => setParams({ ...params, output_dir: v })}
            />
            <FormField
              label="Image size"
              value={params.image_size}
              onChange={(v) => setParams({ ...params, image_size: Number(v) })}
              type="number"
            />
            <FormField
              label="Clip length"
              value={params.clip_length}
              onChange={(v) => setParams({ ...params, clip_length: Number(v) })}
              type="number"
            />
            <FormField
              label="Frames/move"
              value={params.frames_per_move}
              onChange={(v) =>
                setParams({ ...params, frames_per_move: Number(v) })
              }
              type="number"
            />
            <FormField
              label="Seed"
              value={params.seed}
              onChange={(v) => setParams({ ...params, seed: Number(v) })}
              type="number"
            />
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Quality</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                value={params.quality}
                onChange={(e) =>
                  setParams({ ...params, quality: e.target.value })
                }
              >
                <option value="training">Training</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
          <Button onClick={handleStart} disabled={loading} size="sm">
            {loading ? "Starting..." : "Start Generation"}
          </Button>
          {error && (
            <p className="text-xs text-destructive mt-2">{error}</p>
          )}
        </CardContent>
      )}
    </Card>
  );
}

function FormField({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  type?: string;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-muted-foreground">{label}</label>
      <input
        type={type}
        className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
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
