"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getSyntheticStats } from "@/lib/api";
import type { SyntheticStatsResponse, ClipInspectResponse } from "@/lib/types";
import { ClipGallery } from "@/components/ClipGallery";
import { ClipInspector } from "@/components/ClipInspector";
import { ValidityChart } from "@/components/ValidityChart";
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
            </div>

            {(validCount > 0 || invalidCount > 0) && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Clip validity</CardTitle>
                </CardHeader>
                <CardContent>
                  <ValidityChart valid={validCount} invalid={invalidCount} />
                </CardContent>
              </Card>
            )}
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
