"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
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

const DEFAULT_STATUS = "approved:!otb_only";
const DEFAULT_SORT = "published_at_desc";

interface VideoBrowserProps {
  channels: CrawlChannel[];
}

const STATUS_OPTIONS = [
  { label: "All", value: "screened" },
  { label: "Approved", value: "approved" },
  { label: "OTB Only", value: "approved:otb_only" },
  { label: "With Overlay", value: "approved:!otb_only" },
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

const SESSION_KEY = "videoBrowserState";

function getSavedState(): { status: string; sort: string; channel: string | null; page: number } {
  if (typeof window === "undefined") {
    return { status: DEFAULT_STATUS, sort: DEFAULT_SORT, channel: null, page: 0 };
  }
  try {
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) return JSON.parse(saved);
  } catch { /* ignore */ }
  return { status: DEFAULT_STATUS, sort: DEFAULT_SORT, channel: null, page: 0 };
}

function buildQs(statusFilter: string, orderBy: string, channelId: string | null, page: number) {
  const params = new URLSearchParams();
  if (statusFilter !== DEFAULT_STATUS) params.set("status", statusFilter);
  if (orderBy !== DEFAULT_SORT) params.set("sort", orderBy);
  if (channelId) params.set("channel", channelId);
  if (page > 0) params.set("page", String(page));
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

interface BrowseState {
  status: string;
  sort: string;
  channel: string | null;
  page: number;
}

export default function VideoBrowser({
  channels,
}: VideoBrowserProps) {
  const router = useRouter();

  const [browse, setBrowse] = useState<BrowseState>(getSavedState);
  const [videos, setVideos] = useState<VideoWithReason[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({});
  const { toasts, addToast, removeToast } = useToasts();

  // Convenience aliases
  const { status: statusFilter, sort: orderBy, channel: channelId, page } = browse;

  // Setters that reset page when filters change
  const setStatusFilter = (v: string) => setBrowse((s) => ({ ...s, status: v, page: 0 }));
  const setOrderBy = (v: string) => setBrowse((s) => ({ ...s, sort: v, page: 0 }));
  const setChannelId = (v: string | null) => setBrowse((s) => ({ ...s, channel: v, page: 0 }));
  const setPage = (v: number | ((p: number) => number)) =>
    setBrowse((s) => ({ ...s, page: typeof v === "function" ? v(s.page) : v }));

  // Dropdown state
  const [sortOpen, setSortOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);

  // Restore state from sessionStorage on mount (handles Next.js cache)
  useEffect(() => {
    setBrowse(getSavedState());
  }, []);

  // Sync state to URL (display) and sessionStorage (for back navigation)
  useEffect(() => {
    const qs = buildQs(statusFilter, orderBy, channelId, page);
    window.history.replaceState(window.history.state, "", `/videos/browse${qs}`);
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(browse));
  }, [browse, statusFilter, orderBy, channelId, page]);

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
      // Parse compound filter like "approved:otb_only" into status + layout_type
      let status = statusFilter;
      let layoutType: string | undefined;
      if (statusFilter.includes(":")) {
        const [s, l] = statusFilter.split(":");
        status = s;
        layoutType = l;
      }

      const data = await listCrawlVideos({
        channel_id: channelId ?? undefined,
        status,
        layout_type: layoutType,
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
  const hasActiveFilter = statusFilter !== DEFAULT_STATUS || channelId !== null;

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
                      let count: number | undefined;
                      if (f.value === "screened") {
                        count = (statusCounts.approved ?? 0) + (statusCounts.rejected ?? 0);
                      } else if (!f.value.includes(":")) {
                        count = statusCounts[f.value];
                      }
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
              onClick={() => router.push(`/videos/${v.video_id}`)}
              className="border rounded-lg bg-card transition-all duration-300 cursor-pointer hover:border-primary/50 hover:shadow-sm"
            >
              {/* Header: title + score + status */}
              <div className="px-2 py-1 flex items-center gap-1.5 min-w-0">
                <div
                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${scoreColor(v.title_score)}`}
                />
                <span
                  className="text-xs font-medium line-clamp-1 flex-1 min-w-0 hover:text-primary transition-colors"
                >
                  {v.title}
                </span>
                <div onClick={(e) => e.stopPropagation()}>
                  <StatusDropdown video={v} onStatusChange={handleStatusChange} />
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
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
