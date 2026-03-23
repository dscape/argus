"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  listCrawlVideos,
  updateVideoStatus,
  batchUpdateVideoStatus,
  getVideoCounts,
  inspectVideo,
  batchInspectVideos,
  getInspectJobStatus,
} from "@/lib/api";
import type { CrawlChannel, CrawlVideo, InspectResult } from "@/lib/types";

interface VideoInspectorProps {
  channels: CrawlChannel[];
  initialChannelId: string | null;
}

const STATUS_FILTERS = [
  { label: "All", value: null },
  { label: "Approved", value: "approved" },
  { label: "Rejected", value: "rejected" },
  { label: "Unscreened", value: "unscreened" },
];

const PAGE_SIZE = 20;

function statusBadge(status: string | null) {
  switch (status) {
    case "approved":
      return <Badge className="bg-green-600 text-white text-xs">approved</Badge>;
    case "rejected":
      return <Badge variant="destructive" className="text-xs">rejected</Badge>;
    default:
      return <Badge variant="outline" className="text-xs">unscreened</Badge>;
  }
}

function scoreColor(score: number): string {
  if (score >= 0.6) return "bg-green-500";
  if (score >= 0.3) return "bg-yellow-500";
  return "bg-muted-foreground/30";
}

function youtubeThumb(videoId: string, index: number): string {
  return `https://img.youtube.com/vi/${videoId}/${index}.jpg`;
}

function InspectBadges({ result }: { result: InspectResult }) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <Badge className={result.has_overlay ? "bg-green-600 text-white text-xs" : "bg-red-500 text-white text-xs"}>
        OVR {result.has_overlay ? `${result.overlay_score.toFixed(2)}` : "No"}
      </Badge>
      <Badge className={result.has_otb ? "bg-green-600 text-white text-xs" : "bg-red-500 text-white text-xs"}>
        OTB {result.has_otb ? `${result.otb_confidence.toFixed(2)}` : "No"}
      </Badge>
      <Badge className={result.has_person ? "bg-green-600 text-white text-xs" : "bg-red-500 text-white text-xs"}>
        Person {result.has_person ? "Yes" : "No"}
      </Badge>
      <Badge className={result.approved ? "bg-green-600 text-white text-xs" : "bg-red-500 text-white text-xs"}>
        {result.approved ? "Approved" : "Rejected"}
      </Badge>
    </div>
  );
}

export default function VideoInspector({
  channels,
  initialChannelId,
}: VideoInspectorProps) {
  const [channelId, setChannelId] = useState<string | null>(initialChannelId);
  const [videos, setVideos] = useState<CrawlVideo[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string | null>("unscreened");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({});
  const [inspecting, setInspecting] = useState<Set<string>>(new Set());
  const [inspectResults, setInspectResults] = useState<Map<string, InspectResult>>(new Map());
  const [batchJob, setBatchJob] = useState<{ id: string; total: number; completed: number } | null>(null);
  const [batchResult, setBatchResult] = useState<string | null>(null);

  useEffect(() => {
    if (initialChannelId) setChannelId(initialChannelId);
  }, [initialChannelId]);

  const loadCounts = useCallback(async () => {
    try {
      setStatusCounts(await getVideoCounts(channelId ?? undefined));
    } catch {
      // best-effort
    }
  }, [channelId]);

  useEffect(() => {
    loadCounts();
  }, [loadCounts]);

  const loadVideos = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listCrawlVideos({
        channel_id: channelId ?? undefined,
        status: statusFilter ?? undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      });
      setVideos(data.videos);
      setTotal(data.total);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load videos");
    } finally {
      setLoading(false);
    }
  }, [channelId, statusFilter, page]);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  useEffect(() => {
    setPage(0);
    setSelected(new Set());
  }, [channelId, statusFilter]);

  const removeVideo = (videoId: string) => {
    setVideos((prev) => prev.filter((v) => v.video_id !== videoId));
    setTotal((prev) => prev - 1);
    setSelected((prev) => {
      const next = new Set(prev);
      next.delete(videoId);
      return next;
    });
  };

  const removeVideos = (videoIds: string[]) => {
    const idSet = new Set(videoIds);
    setVideos((prev) => prev.filter((v) => !idSet.has(v.video_id)));
    setTotal((prev) => prev - videoIds.length);
    setSelected((prev) => {
      const next = new Set(prev);
      videoIds.forEach((id) => next.delete(id));
      return next;
    });
  };

  const handleStatusChange = async (
    videoId: string,
    status: string | null,
    layoutType?: string
  ) => {
    try {
      await updateVideoStatus(videoId, status, layoutType);
      removeVideo(videoId);
      loadCounts();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update status");
    }
  };

  const handleBatchUpdate = async (status: string) => {
    if (selected.size === 0) return;
    const ids = Array.from(selected);
    try {
      await batchUpdateVideoStatus(ids, status);
      removeVideos(ids);
      loadCounts();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch update failed");
    }
  };

  const handleRejectAll = async () => {
    const rejectableIds = videos
      .filter((v) => v.screening_status !== "rejected")
      .map((v) => v.video_id);
    if (rejectableIds.length === 0) return;
    try {
      await batchUpdateVideoStatus(rejectableIds, "rejected");
      removeVideos(rejectableIds);
      loadCounts();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reject all failed");
    }
  };

  const handleInspect = async (videoId: string) => {
    setInspecting((prev) => new Set(prev).add(videoId));
    setError(null);
    try {
      const result = await inspectVideo(videoId);
      setInspectResults((prev) => new Map(prev).set(videoId, result));
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === videoId ? { ...v, screening_status: result.status } : v
        )
      );
      loadCounts();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inspection failed");
    } finally {
      setInspecting((prev) => {
        const next = new Set(prev);
        next.delete(videoId);
        return next;
      });
    }
  };

  const handleBatchInspect = async () => {
    if (selected.size === 0) return;
    setError(null);
    try {
      const { job_id } = await batchInspectVideos(Array.from(selected));
      setBatchJob({ id: job_id, total: selected.size, completed: 0 });
      setSelected(new Set());
      const poll = setInterval(async () => {
        try {
          const status = await getInspectJobStatus(job_id);
          setBatchJob({ id: job_id, total: status.total, completed: status.completed });
          if (status.status === "done") {
            clearInterval(poll);
            setBatchJob(null);
            setBatchResult(
              `Inspected ${status.total}: ${status.approved} approved, ${status.rejected} rejected${
                status.failed ? `, ${status.failed} failed` : ""
              }`
            );
            loadVideos();
            loadCounts();
          }
        } catch {
          clearInterval(poll);
          setBatchJob(null);
          setError("Failed to poll inspection job");
        }
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch inspection failed");
    }
  };

  const toggleSelect = (videoId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) next.delete(videoId);
      else next.add(videoId);
      return next;
    });
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <select
            value={channelId ?? ""}
            onChange={(e) => setChannelId(e.target.value || null)}
            className="rounded-md border bg-background px-3 py-2 text-sm min-w-[200px]"
          >
            <option value="">All Channels</option>
            {channels.map((ch) => (
              <option key={ch.channel_id} value={ch.channel_id}>
                {ch.channel_name} ({ch.video_count})
              </option>
            ))}
          </select>

          <div className="flex items-center gap-1">
            {STATUS_FILTERS.map((f) => (
              <button
                key={f.label}
                onClick={() => setStatusFilter(f.value)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  statusFilter === f.value
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {f.label}
                {statusCounts[f.value ?? "all"] !== undefined && (
                  <span className="ml-1 opacity-60">
                    ({statusCounts[f.value ?? "all"]})
                  </span>
                )}
              </button>
            ))}
          </div>

        </div>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
          {error}
          <button onClick={() => setError(null)} className="ml-2 underline text-xs">
            dismiss
          </button>
        </div>
      )}

      {batchResult && (
        <div className="rounded-md bg-primary/10 border border-primary/20 p-3 text-sm">
          {batchResult}
          <button onClick={() => setBatchResult(null)} className="ml-2 underline text-xs">
            dismiss
          </button>
        </div>
      )}

      {/* Batch actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50 border text-sm">
          <span className="text-muted-foreground">{selected.size} selected</span>
          <Button
            size="sm"
            className="h-7 text-xs bg-blue-600 hover:bg-blue-700"
            onClick={handleBatchInspect}
            disabled={!!batchJob}
          >
            Inspect
          </Button>
          <Button
            size="sm"
            className="h-7 text-xs bg-green-600 hover:bg-green-700"
            onClick={() => handleBatchUpdate("approved")}
          >
            Approve
          </Button>
          <Button
            size="sm"
            variant="destructive"
            className="h-7 text-xs"
            onClick={() => handleBatchUpdate("rejected")}
          >
            Reject
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => setSelected(new Set())}
          >
            Clear
          </Button>
        </div>
      )}

      {/* Batch inspection progress */}
      {batchJob && (
        <div className="rounded-md bg-blue-500/10 border border-blue-500/20 p-3 text-sm">
          <div className="flex items-center gap-2">
            <div className="h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span>Inspecting {batchJob.completed}/{batchJob.total} videos...</span>
          </div>
          <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(batchJob.completed / batchJob.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Loading videos...
        </div>
      ) : videos.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          No videos found. Try a different filter or crawl this channel first.
        </div>
      ) : (
        /* Thumbnail card grid */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {videos.map((v) => (
            <div
              key={v.video_id}
              className={`border rounded-lg overflow-hidden bg-card ${
                selected.has(v.video_id) ? "ring-2 ring-primary" : ""
              }`}
            >
              {/* Header: checkbox + title + score + status */}
              <div className="px-3 py-2 border-b bg-muted/30 flex items-start gap-2">
                <input
                  type="checkbox"
                  checked={selected.has(v.video_id)}
                  onChange={() => toggleSelect(v.video_id)}
                  className="rounded mt-1 flex-shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <a
                    href={`https://www.youtube.com/watch?v=${v.video_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium line-clamp-1 hover:text-primary transition-colors"
                  >
                    {v.title}
                  </a>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="flex items-center gap-1">
                      <div className={`w-2 h-2 rounded-full ${scoreColor(v.title_score)}`} />
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {v.title_score.toFixed(2)}
                      </span>
                    </div>
                    {statusBadge(v.screening_status)}
                  </div>
                </div>
              </div>

              {/* 2x2 thumbnail grid */}
              <div className="grid grid-cols-2 gap-px bg-muted">
                {[0, 1, 2, 3].map((i) => (
                  <img
                    key={i}
                    src={youtubeThumb(v.video_id, i)}
                    alt={`Frame ${i}`}
                    className="w-full aspect-video object-cover bg-background"
                    loading="lazy"
                  />
                ))}
              </div>

              {/* Inspect results (if available) */}
              {inspectResults.has(v.video_id) && (
                <div className="px-3 py-2 border-t bg-muted/20">
                  <InspectBadges result={inspectResults.get(v.video_id)!} />
                </div>
              )}

              {/* Actions */}
              <div className="px-3 py-2 border-t flex items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs text-blue-600 hover:text-blue-700"
                  onClick={() => handleInspect(v.video_id)}
                  disabled={inspecting.has(v.video_id)}
                >
                  {inspecting.has(v.video_id) ? (
                    <span className="flex items-center gap-1">
                      <span className="h-3 w-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    </span>
                  ) : inspectResults.has(v.video_id) ? (
                    "Re-inspect"
                  ) : (
                    "Inspect"
                  )}
                </Button>
                <div className="flex-1" />
                {v.screening_status !== "approved" && (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 text-xs text-green-600 border-green-600 hover:bg-green-50"
                      onClick={() => handleStatusChange(v.video_id, "approved", "otb_only")}
                    >
                      OTB Only
                    </Button>
                    <Button
                      size="sm"
                      className="h-7 text-xs bg-green-600 hover:bg-green-700"
                      onClick={() => handleStatusChange(v.video_id, "approved")}
                    >
                      Approve
                    </Button>
                  </>
                )}
                {v.screening_status !== "rejected" && (
                  <Button
                    size="sm"
                    variant="destructive"
                    className="h-7 text-xs"
                    onClick={() => handleStatusChange(v.video_id, "rejected")}
                  >
                    Reject
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-muted-foreground">
            Showing {page * PAGE_SIZE + 1}–
            {Math.min((page + 1) * PAGE_SIZE, total)} of {total}
          </span>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-xs text-muted-foreground">
              Page {page + 1} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}

      {/* Floating Reject All */}
      {videos.length > 0 && !videos.every((v) => v.screening_status === "rejected") && (
        <div className="fixed bottom-6 right-6 z-50">
          <Button
            variant="destructive"
            className="shadow-lg"
            onClick={handleRejectAll}
          >
            Reject All
          </Button>
        </div>
      )}
    </div>
  );
}
