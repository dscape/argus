"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  listCrawlVideos,
  updateVideoStatus,
  getVideoCounts,
} from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";
import {
  VideoWithReason,
  SortIcon,
  FilterIcon,
  scoreColor,
  youtubeThumb,
  useToasts,
  ToastContainer,
  ToolbarIconButton,
  ToolbarDropdown,
  StatusDropdown,
} from "@/components/video-shared";

interface VideoBrowserProps {
  channels: CrawlChannel[];
  initialChannelId: string | null;
}

const STATUS_OPTIONS = [
  { label: "All", value: "screened" },
  { label: "Approved", value: "approved" },
  { label: "Rejected", value: "rejected" },
];

const SORT_OPTIONS = [
  { label: "Title Score", value: "title_score_desc" },
  { label: "Date (newest)", value: "published_at_desc" },
  { label: "Channel", value: "channel_name" },
];

const PAGE_SIZE = 18;

function ChevronLeftIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

function ChevronRightIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M9 18l6-6-6-6" />
    </svg>
  );
}

export default function VideoBrowser({
  channels,
  initialChannelId,
}: VideoBrowserProps) {
  const [channelId, setChannelId] = useState<string | null>(initialChannelId);
  const [videos, setVideos] = useState<VideoWithReason[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>("screened");
  const [orderBy, setOrderBy] = useState("title_score_desc");
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({});
  const [page, setPage] = useState(0);
  const { toasts, addToast, removeToast } = useToasts();

  // Dropdown state
  const [sortOpen, setSortOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);

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
        status: statusFilter,
        order_by: orderBy === "title_score_desc" ? undefined : orderBy,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      });
      setVideos(data.videos);
      setTotal(data.total);
    } catch (e) {
      addToast({
        message: e instanceof Error ? e.message : "Failed to load videos",
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  }, [channelId, statusFilter, orderBy, page, addToast]);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  // Reset page when filters change
  useEffect(() => {
    setPage(0);
  }, [channelId, statusFilter, orderBy]);

  const updateVideoInPlace = useCallback(
    (videoId: string, newStatus: string | null, layoutType?: string | null, inspectReason?: string) => {
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === videoId
            ? {
                ...v,
                screening_status: newStatus,
                layout_type: layoutType !== undefined ? layoutType : v.layout_type,
                inspect_reason: inspectReason !== undefined ? inspectReason : v.inspect_reason,
              }
            : v
        )
      );
    },
    []
  );

  const handleStatusChange = useCallback(
    async (videoId: string, status: string | null, layoutType?: string) => {
      const video = videos.find((v) => v.video_id === videoId);
      const previousStatus = video?.screening_status ?? null;
      const previousLayout = video?.layout_type ?? null;

      try {
        await updateVideoStatus(videoId, status, layoutType);
        // If unscreening, remove from the list since we're viewing screened only
        if (status === null) {
          setVideos((prev) => prev.filter((v) => v.video_id !== videoId));
          setTotal((prev) => prev - 1);
        } else {
          updateVideoInPlace(videoId, status, layoutType ?? null);
        }
        loadCounts();

        const label =
          status === null
            ? "unscreened"
            : layoutType === "otb_only"
            ? "marked OTB only"
            : status === "approved"
            ? "approved"
            : status === "rejected"
            ? "rejected"
            : "updated";

        addToast({
          message: `Video ${label}`,
          type: "success",
          undoAction: async () => {
            await updateVideoStatus(videoId, previousStatus);
            if (status === null) {
              // Re-fetch to get the video back
              loadVideos();
            } else {
              updateVideoInPlace(videoId, previousStatus, previousLayout);
            }
            loadCounts();
          },
        });
      } catch (e) {
        addToast({
          message: e instanceof Error ? e.message : "Failed to update status",
          type: "error",
        });
      }
    },
    [videos, updateVideoInPlace, loadCounts, addToast, loadVideos]
  );

  const sortedVideos = useMemo(() => {
    const sorted = [...videos];
    switch (orderBy) {
      case "title_score_desc":
        return sorted.sort((a, b) => b.title_score - a.title_score);
      case "published_at_desc":
        return sorted.sort((a, b) =>
          (b.published_at ?? "").localeCompare(a.published_at ?? "")
        );
      case "channel_name":
        return sorted.sort((a, b) =>
          (a.channel_handle ?? "").localeCompare(b.channel_handle ?? "")
        );
      default:
        return sorted;
    }
  }, [videos, orderBy]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const hasActiveFilter = statusFilter !== "screened" || channelId !== null;

  return (
    <div className="relative h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* Sticky Toolbar */}
      <div className="flex-shrink-0 z-40 bg-background/95 backdrop-blur pb-1 pt-1 mb-1">
        <div className="flex items-center gap-1 rounded-2xl border bg-muted/30 p-1.5">
          {/* Sort dropdown */}
          <div className="relative">
            <ToolbarIconButton
              active={sortOpen}
              label="Sort"
              onClick={() => { setSortOpen(!sortOpen); setFilterOpen(false); }}
            >
              <SortIcon className="w-4 h-4" />
            </ToolbarIconButton>
            <ToolbarDropdown open={sortOpen} onClose={() => setSortOpen(false)}>
              <div className="text-xs font-medium text-muted-foreground px-2 py-1">Sort by</div>
              {SORT_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => { setOrderBy(opt.value); setSortOpen(false); }}
                  className={`w-full text-left px-2 py-1.5 rounded text-sm hover:bg-muted transition-colors ${
                    orderBy === opt.value ? "bg-muted font-medium" : ""
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </ToolbarDropdown>
          </div>

          {/* Filter dropdown */}
          <div className="relative">
            <ToolbarIconButton
              active={filterOpen || hasActiveFilter}
              label="Filter"
              onClick={() => { setFilterOpen(!filterOpen); setSortOpen(false); }}
            >
              <FilterIcon className="w-4 h-4" />
              {hasActiveFilter && !filterOpen && (
                <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-primary" />
              )}
            </ToolbarIconButton>
            <ToolbarDropdown open={filterOpen} onClose={() => setFilterOpen(false)}>
              <div className="text-xs font-medium text-muted-foreground px-2 py-1">Status</div>
              <div className="flex flex-wrap gap-1 px-1 pb-2">
                {STATUS_OPTIONS.map((f) => (
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
                    {(() => {
                      const count = f.value === "screened"
                        ? (statusCounts.all ?? 0) - (statusCounts.unscreened ?? 0)
                        : statusCounts[f.value];
                      return count !== undefined ? (
                        <span className="ml-1 opacity-60">({count})</span>
                      ) : null;
                    })()}
                  </button>
                ))}
              </div>
              <div className="border-t pt-2 mt-1">
                <div className="text-xs font-medium text-muted-foreground px-2 py-1">Channel</div>
                <select
                  value={channelId ?? ""}
                  onChange={(e) => setChannelId(e.target.value || null)}
                  className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
                >
                  <option value="">All Channels</option>
                  {channels.map((ch) => (
                    <option key={ch.channel_id} value={ch.channel_id}>
                      {ch.channel_name} ({ch.video_count})
                    </option>
                  ))}
                </select>
              </div>
            </ToolbarDropdown>
          </div>

          {/* Showing X of N */}
          <span className="text-xs text-muted-foreground px-2 tabular-nums">
            Showing {videos.length} of {total.toLocaleString()}
          </span>

          <div className="flex-1" />

          {/* Pagination */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0 || loading}
              className="flex items-center justify-center w-8 h-8 rounded-xl text-muted-foreground hover:text-foreground hover:bg-background/60 transition-all duration-150 disabled:opacity-30 disabled:pointer-events-none"
            >
              <ChevronLeftIcon className="w-4 h-4" />
            </button>
            <span className="text-xs text-muted-foreground tabular-nums px-1">
              {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1 || loading}
              className="flex items-center justify-center w-8 h-8 rounded-xl text-muted-foreground hover:text-foreground hover:bg-background/60 transition-all duration-150 disabled:opacity-30 disabled:pointer-events-none"
            >
              <ChevronRightIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Video grid */}
      {loading ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Loading videos...
        </div>
      ) : videos.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          No videos found. Try a different filter or crawl channels first.
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 xl:grid-cols-3 gap-2 auto-rows-min content-start overflow-auto">
          {sortedVideos.map((v) => (
            <div
              key={v.video_id}
              className="border rounded-lg bg-card transition-all duration-300"
            >
              {/* Header: title + score + status */}
              <div className="px-2 py-1 flex items-center gap-1.5 min-w-0">
                <div
                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${scoreColor(v.title_score)}`}
                />
                <a
                  href={`https://www.youtube.com/watch?v=${v.video_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs font-medium line-clamp-1 flex-1 min-w-0 hover:text-primary transition-colors"
                >
                  {v.title}
                </a>
                <StatusDropdown video={v} onStatusChange={handleStatusChange} />
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
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
