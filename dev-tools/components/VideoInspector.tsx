"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  listCrawlVideos,
  updateVideoStatus,
  batchUpdateVideoStatus,
} from "@/lib/api";
import type { CrawlChannel, CrawlVideo } from "@/lib/types";

interface VideoInspectorProps {
  channels: CrawlChannel[];
  initialChannelId: string | null;
}

const STATUS_FILTERS = [
  { label: "All", value: null },
  { label: "Candidates", value: "candidate" },
  { label: "Approved", value: "approved" },
  { label: "Rejected", value: "rejected" },
  { label: "Unscreened", value: "unscreened" },
];

const PAGE_SIZE = 30;

function statusBadge(status: string | null) {
  switch (status) {
    case "approved":
      return <Badge className="bg-green-600 text-white text-xs">approved</Badge>;
    case "rejected":
      return <Badge variant="destructive" className="text-xs">rejected</Badge>;
    case "candidate":
      return <Badge className="bg-yellow-500 text-white text-xs">candidate</Badge>;
    default:
      return <Badge variant="outline" className="text-xs">unscreened</Badge>;
  }
}

function scoreColor(score: number): string {
  if (score >= 0.6) return "bg-green-500";
  if (score >= 0.3) return "bg-yellow-500";
  return "bg-muted-foreground/30";
}

export default function VideoInspector({
  channels,
  initialChannelId,
}: VideoInspectorProps) {
  const [channelId, setChannelId] = useState<string | null>(
    initialChannelId
  );
  const [videos, setVideos] = useState<CrawlVideo[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"table" | "gallery">("table");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialChannelId) setChannelId(initialChannelId);
  }, [initialChannelId]);

  const loadVideos = useCallback(async () => {
    if (!channelId) {
      setVideos([]);
      setTotal(0);
      return;
    }
    setLoading(true);
    try {
      const data = await listCrawlVideos({
        channel_id: channelId,
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

  const handleStatusChange = async (
    videoId: string,
    status: string | null
  ) => {
    try {
      await updateVideoStatus(videoId, status);
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === videoId
            ? { ...v, screening_status: status }
            : v
        )
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update status");
    }
  };

  const handleBatchUpdate = async (status: string) => {
    if (selected.size === 0) return;
    try {
      await batchUpdateVideoStatus(Array.from(selected), status);
      setVideos((prev) =>
        prev.map((v) =>
          selected.has(v.video_id)
            ? { ...v, screening_status: status }
            : v
        )
      );
      setSelected(new Set());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch update failed");
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

  const toggleSelectAll = () => {
    if (selected.size === videos.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(videos.map((v) => v.video_id)));
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const formatDate = (d: string | null) => {
    if (!d) return "";
    return new Date(d).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={channelId ?? ""}
          onChange={(e) => setChannelId(e.target.value || null)}
          className="rounded-md border bg-background px-3 py-2 text-sm min-w-[200px]"
        >
          <option value="">Select a channel...</option>
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
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={() => setViewMode("table")}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              viewMode === "table"
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground"
            }`}
          >
            Table
          </button>
          <button
            onClick={() => setViewMode("gallery")}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              viewMode === "gallery"
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground"
            }`}
          >
            Gallery
          </button>
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

      {/* Batch actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50 border text-sm">
          <span className="text-muted-foreground">
            {selected.size} selected
          </span>
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

      {!channelId ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Select a channel to browse videos.
        </div>
      ) : loading ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Loading videos...
        </div>
      ) : videos.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          No videos found. Try a different filter or crawl this channel first.
        </div>
      ) : viewMode === "table" ? (
        /* Table view */
        <div className="border rounded-md overflow-x-auto">
          <table className="w-full text-sm min-w-[700px]">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-3 py-2 w-8">
                  <input
                    type="checkbox"
                    checked={selected.size === videos.length && videos.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded"
                  />
                </th>
                <th className="text-left px-3 py-2 font-medium">Title</th>
                <th className="text-left px-3 py-2 font-medium w-28">Published</th>
                <th className="text-center px-3 py-2 font-medium w-16">Score</th>
                <th className="text-center px-3 py-2 font-medium w-24">Status</th>
                <th className="text-right px-3 py-2 font-medium w-32">Actions</th>
              </tr>
            </thead>
            <tbody>
              {videos.map((v) => (
                <tr
                  key={v.video_id}
                  className={`border-b last:border-b-0 hover:bg-muted/30 ${
                    selected.has(v.video_id) ? "bg-primary/5" : ""
                  }`}
                >
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selected.has(v.video_id)}
                      onChange={() => toggleSelect(v.video_id)}
                      className="rounded"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <a
                      href={`https://www.youtube.com/watch?v=${v.video_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-primary transition-colors"
                    >
                      {v.title}
                    </a>
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {formatDate(v.published_at)}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center justify-center gap-1.5">
                      <div
                        className={`w-2 h-2 rounded-full ${scoreColor(v.title_score)}`}
                      />
                      <span className="text-xs tabular-nums">
                        {v.title_score.toFixed(1)}
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-center">
                    {statusBadge(v.screening_status)}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {v.screening_status !== "approved" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-xs text-green-600 hover:text-green-700"
                          onClick={() =>
                            handleStatusChange(v.video_id, "approved")
                          }
                        >
                          Approve
                        </Button>
                      )}
                      {v.screening_status !== "rejected" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-xs text-destructive"
                          onClick={() =>
                            handleStatusChange(v.video_id, "rejected")
                          }
                        >
                          Reject
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        /* Gallery view */
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {videos.map((v) => (
            <div
              key={v.video_id}
              className={`border rounded-lg overflow-hidden bg-card ${
                selected.has(v.video_id) ? "ring-2 ring-primary" : ""
              }`}
            >
              <div className="aspect-video">
                <iframe
                  src={`https://www.youtube.com/embed/${v.video_id}`}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="w-full h-full"
                />
              </div>
              <div className="p-3 space-y-2">
                <div className="flex items-start gap-2">
                  <input
                    type="checkbox"
                    checked={selected.has(v.video_id)}
                    onChange={() => toggleSelect(v.video_id)}
                    className="rounded mt-0.5"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium line-clamp-2">
                      {v.title}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex items-center gap-1">
                        <div
                          className={`w-2 h-2 rounded-full ${scoreColor(v.title_score)}`}
                        />
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {v.title_score.toFixed(1)}
                        </span>
                      </div>
                      {statusBadge(v.screening_status)}
                      <span className="text-xs text-muted-foreground">
                        {formatDate(v.published_at)}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-1">
                  {v.screening_status !== "approved" && (
                    <Button
                      size="sm"
                      className="h-7 text-xs flex-1 bg-green-600 hover:bg-green-700"
                      onClick={() =>
                        handleStatusChange(v.video_id, "approved")
                      }
                    >
                      Approve
                    </Button>
                  )}
                  {v.screening_status !== "rejected" && (
                    <Button
                      size="sm"
                      variant="destructive"
                      className="h-7 text-xs flex-1"
                      onClick={() =>
                        handleStatusChange(v.video_id, "rejected")
                      }
                    >
                      Reject
                    </Button>
                  )}
                </div>
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
    </div>
  );
}
