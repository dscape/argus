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
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  scanSyntheticDir,
  getSyntheticStats,
  inspectSyntheticClip,
  getClipInfo,
  clipFrameUrl,
  deleteClipSession,
} from "@/lib/api";
import type {
  SyntheticScanResponse,
  SyntheticStatsResponse,
  ClipInspectResponse,
} from "@/lib/types";

export default function SyntheticPage() {
  const [directory, setDirectory] = useState("data/train");
  const [expectedClips, setExpectedClips] = useState<number | undefined>(
    undefined
  );
  const [watching, setWatching] = useState(true);
  const [scan, setScan] = useState<SyntheticScanResponse | null>(null);
  const [stats, setStats] = useState<SyntheticStatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Clip viewer state
  const [activeTab, setActiveTab] = useState("files");
  const [clipSessionId, setClipSessionId] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<ClipInspectResponse | null>(null);
  const [inspectingFile, setInspectingFile] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const doScan = useCallback(async () => {
    try {
      const result = await scanSyntheticDir(directory, expectedClips);
      setScan(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan failed");
    }
  }, [directory, expectedClips]);

  // Polling
  useEffect(() => {
    if (watching) {
      doScan();
      intervalRef.current = setInterval(doScan, 2000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [watching, doScan]);

  const handleComputeStats = async () => {
    setStatsLoading(true);
    try {
      const result = await getSyntheticStats(directory);
      setStats(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stats failed");
    }
    setStatsLoading(false);
  };

  const handleInspect = async (filename: string) => {
    // Clean up previous session
    if (clipSessionId) {
      deleteClipSession(clipSessionId).catch(() => {});
    }

    const filepath = `${scan?.directory}/${filename}`;
    try {
      const { session_id } = await inspectSyntheticClip(filepath);
      setClipSessionId(session_id);
      setInspectingFile(filename);
      const info = await getClipInfo(session_id);
      setClipInfo(info);
      setActiveTab("viewer");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inspect failed");
    }
  };

  const progress =
    scan && scan.expected_clips
      ? Math.min(100, Math.round((scan.clip_count / scan.expected_clips) * 100))
      : null;

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Synthetic Monitor</h2>
      <p className="text-muted-foreground mb-6">
        Watch synthetic data generation in real time. Browse generated clips and
        view aggregated stats.
      </p>

      {/* Directory config bar */}
      <div className="flex items-end gap-3 mb-6">
        <div>
          <label className="text-xs font-medium text-muted-foreground block mb-1">
            Directory
          </label>
          <input
            type="text"
            value={directory}
            onChange={(e) => setDirectory(e.target.value)}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm"
            placeholder="data/train"
          />
        </div>
        <div>
          <label className="text-xs font-medium text-muted-foreground block mb-1">
            Expected clips
          </label>
          <input
            type="number"
            value={expectedClips ?? ""}
            onChange={(e) =>
              setExpectedClips(
                e.target.value ? parseInt(e.target.value, 10) : undefined
              )
            }
            className="h-9 w-24 rounded-md border border-input bg-background px-3 text-sm"
            placeholder="100"
          />
        </div>
        <Button
          variant={watching ? "destructive" : "default"}
          onClick={() => setWatching(!watching)}
          className="h-9"
        >
          {watching ? "Stop" : "Watch"}
        </Button>
        {!watching && (
          <Button variant="outline" onClick={doScan} className="h-9">
            Scan once
          </Button>
        )}
      </div>

      {/* Progress bar */}
      {scan && (
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-sm font-medium">
              {scan.clip_count} clips
              {scan.expected_clips ? ` / ${scan.expected_clips}` : ""}
            </span>
            <span className="text-xs text-muted-foreground">
              {scan.total_size_mb} MB
            </span>
            {watching && (
              <Badge variant="outline" className="text-xs">
                live
              </Badge>
            )}
            {progress !== null && progress >= 100 && (
              <Badge className="text-xs">complete</Badge>
            )}
          </div>
          {progress !== null && (
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive mb-4">{error}</p>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="stats">Stats</TabsTrigger>
          {clipInfo && <TabsTrigger value="viewer">Clip Viewer</TabsTrigger>}
        </TabsList>

        {/* Files tab */}
        <TabsContent value="files">
          {scan && scan.clips.length > 0 ? (
            <div className="border rounded-md overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left px-3 py-2 font-medium">File</th>
                    <th className="text-right px-3 py-2 font-medium">Size</th>
                    <th className="text-right px-3 py-2 font-medium">
                      Modified
                    </th>
                    <th className="text-right px-3 py-2 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {scan.clips.map((clip) => (
                    <tr
                      key={clip.filename}
                      className="border-b last:border-0 hover:bg-muted/30"
                    >
                      <td className="px-3 py-2 font-mono text-xs">
                        {clip.filename}
                      </td>
                      <td className="px-3 py-2 text-right text-muted-foreground">
                        {clip.size_mb} MB
                      </td>
                      <td className="px-3 py-2 text-right text-muted-foreground text-xs">
                        {new Date(clip.modified).toLocaleTimeString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleInspect(clip.filename)}
                        >
                          Inspect
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : scan ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              {scan.exists
                ? "No .pt files found in this directory."
                : "Directory does not exist yet. Start generation and it will be created."}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground py-8 text-center">
              Scanning...
            </p>
          )}
        </TabsContent>

        {/* Stats tab */}
        <TabsContent value="stats">
          <div className="mb-4">
            <Button
              onClick={handleComputeStats}
              disabled={statsLoading}
            >
              {statsLoading ? "Computing..." : "Compute Stats"}
            </Button>
            <span className="text-xs text-muted-foreground ml-2">
              Loads all .pt files — may take a moment for large datasets
            </span>
          </div>
          {stats && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
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

              {/* Moves distribution */}
              {stats.moves_per_clip_distribution.length > 0 && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">
                      Moves per clip distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <MovesDistribution
                      data={stats.moves_per_clip_distribution}
                    />
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        {/* Clip viewer tab */}
        <TabsContent value="viewer">
          {clipInfo && clipSessionId && (
            <div>
              <div className="flex items-center gap-3 mb-4">
                <h3 className="text-lg font-semibold">{inspectingFile}</h3>
                <Badge variant="outline">{clipInfo.file_size_mb} MB</Badge>
                <Badge variant={clipInfo.replay_valid ? "default" : "destructive"}>
                  {clipInfo.replay_valid ? "valid" : "invalid"}
                </Badge>
              </div>

              {/* Tensor info */}
              <div className="mb-4">
                <h4 className="text-sm font-medium mb-2">Tensors</h4>
                <div className="border rounded-md overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left px-3 py-1 font-medium">
                          Name
                        </th>
                        <th className="text-left px-3 py-1 font-medium">
                          Shape
                        </th>
                        <th className="text-left px-3 py-1 font-medium">
                          Dtype
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {clipInfo.tensors.map((t) => (
                        <tr key={t.name} className="border-b last:border-0">
                          <td className="px-3 py-1 font-mono text-xs">
                            {t.name}
                          </td>
                          <td className="px-3 py-1 font-mono text-xs">
                            [{t.shape.join(", ")}]
                          </td>
                          <td className="px-3 py-1 text-xs text-muted-foreground">
                            {t.dtype}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Moves */}
              {clipInfo.moves.length > 0 && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-2">
                    Moves ({clipInfo.total_moves})
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {clipInfo.moves.map((m, i) => (
                      <span
                        key={i}
                        className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded"
                        title={`Frame ${m.frame_index}`}
                      >
                        {m.san || m.uci}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Frames grid */}
              <div className="mb-4">
                <h4 className="text-sm font-medium mb-2">
                  Frames ({clipInfo.num_frames})
                </h4>
                <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
                  {Array.from({ length: clipInfo.num_frames }, (_, i) => (
                    <div key={i} className="relative">
                      <img
                        src={clipFrameUrl(clipSessionId, i)}
                        alt={`Frame ${i}`}
                        className="w-full rounded border"
                      />
                      <span className="absolute bottom-0 right-0 text-[10px] bg-black/60 text-white px-1 rounded-tl">
                        {i}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {clipInfo.replay_error && (
                <p className="text-sm text-destructive">
                  {clipInfo.replay_error}
                </p>
              )}
            </div>
          )}
        </TabsContent>
      </Tabs>
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

function MovesDistribution({ data }: { data: number[] }) {
  // Bucket into a histogram
  const buckets: Record<number, number> = {};
  for (const v of data) {
    buckets[v] = (buckets[v] || 0) + 1;
  }
  const sorted = Object.entries(buckets)
    .map(([k, v]) => [Number(k), v] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const maxCount = Math.max(...sorted.map((s) => s[1]), 1);

  return (
    <div className="flex items-end gap-1 h-24">
      {sorted.map(([moves, count]) => {
        const height = Math.max(4, (count / maxCount) * 100);
        return (
          <div key={moves} className="flex flex-col items-center flex-1">
            <div
              className="w-full bg-primary/70 rounded-t min-w-[4px]"
              style={{ height: `${height}%` }}
              title={`${moves} moves: ${count} clips`}
            />
            <span className="text-[9px] text-muted-foreground mt-1">
              {moves}
            </span>
          </div>
        );
      })}
    </div>
  );
}
