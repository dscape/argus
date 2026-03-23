"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import {
  listCrawlVideos,
  updateVideoStatus,
  batchUpdateVideoStatus,
  getVideoCounts,
  batchInspectVideos,
  getInspectJobStatus,
  undoAutoReject,
} from "@/lib/api";
import type { CrawlChannel, CrawlVideo } from "@/lib/types";

interface VideoScreenerProps {
  channels: CrawlChannel[];
  initialChannelId: string | null;
}

const STATUS_OPTIONS = [
  { label: "All", value: null },
  { label: "Unscreened", value: "unscreened" },
  { label: "Approved", value: "approved" },
  { label: "Rejected", value: "rejected" },
];

const SORT_OPTIONS = [
  { label: "Title Score", value: "title_score_desc" },
  { label: "Date (newest)", value: "published_at_desc" },
  { label: "Channel", value: "channel_name" },
];

interface Toast {
  id: string;
  message: string;
  type: "success" | "warning" | "error";
  undoAction?: () => Promise<void>;
}

// Extended video with local inspect reason
interface VideoWithReason extends CrawlVideo {
  inspect_reason?: string;
}

// ── Icons ──────────────────────────────────────────────────

function SortIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18" />
      <path d="M3 12h12" />
      <path d="M3 18h6" />
    </svg>
  );
}

function FilterIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

function RobotIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect x="4" y="8" width="16" height="12" rx="2" />
      <path d="M12 8V4" />
      <circle cx="12" cy="3" r="1" />
      <circle cx="9" cy="14" r="1.5" fill="currentColor" />
      <circle cx="15" cy="14" r="1.5" fill="currentColor" />
      <path d="M9 18h6" />
      <path d="M2 14h2" />
      <path d="M20 14h2" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" />
      <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

// ── Helpers ─────────────────────────────────────────────────

function statusBadge(status: string | null, layoutType?: string | null) {
  switch (status) {
    case "approved":
      return (
        <Badge className="bg-green-600 text-white text-xs">
          {layoutType === "otb_only" ? "OTB only" : "approved"}
        </Badge>
      );
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

/** Compute how many video cards fit above the fold in a 3-col grid. */
function computePageSize(): number {
  if (typeof window === "undefined") return 6;
  const vh = window.innerHeight;
  const headerHeight = 130; // page title + toolbar
  const cardHeight = 308;   // card (~300px) + gap (8px)
  const cols = window.innerWidth >= 1280 ? 3 : 2;
  const rows = Math.max(1, Math.floor((vh - headerHeight) / cardHeight));
  return rows * cols;
}

// ── Toolbar Icon Button (matches sidebar style) ─────────────

function ToolbarIconButton({
  children,
  active,
  label,
  className,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  active?: boolean;
  label: string;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={onClick}
        disabled={disabled}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={`flex items-center justify-center w-8 h-8 rounded-xl transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none ${
          active
            ? "bg-background text-foreground shadow-sm"
            : "text-muted-foreground/50 hover:text-foreground hover:bg-background/60"
        } ${className ?? ""}`}
      >
        {children}
      </button>
      {hovered && !disabled && (
        <div className="absolute top-full mt-1.5 left-1/2 -translate-x-1/2 px-2 py-1 rounded-md bg-foreground text-background text-xs font-medium whitespace-nowrap z-[60] pointer-events-none">
          {label}
        </div>
      )}
    </div>
  );
}

// ── Dropdown Components ─────────────────────────────────────

function ToolbarDropdown({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={ref}
      className="absolute top-full mt-1 left-0 z-[60] rounded-lg border bg-background shadow-lg p-2 min-w-[200px]"
    >
      {children}
    </div>
  );
}

// ── Unscreened Actions (inline buttons) ─────────────────────

function UnscreenedActions({
  videoId,
  onStatusChange,
}: {
  videoId: string;
  onStatusChange: (videoId: string, status: string | null, layoutType?: string) => Promise<void>;
}) {
  return (
    <div className="flex items-center gap-1 flex-shrink-0">
      <button
        onClick={() => onStatusChange(videoId, "approved", "otb_only")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-amber-600 hover:bg-amber-700 text-white transition-colors"
      >
        OTB Only
      </button>
      <button
        onClick={() => onStatusChange(videoId, "approved")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-green-600 hover:bg-green-700 text-white transition-colors"
      >
        Approve
      </button>
      <button
        onClick={() => onStatusChange(videoId, "rejected")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-destructive hover:bg-destructive/90 text-destructive-foreground transition-colors"
      >
        Reject
      </button>
    </div>
  );
}

// ── Status Dropdown (for already-screened videos) ───────────

function StatusDropdown({
  video,
  onStatusChange,
}: {
  video: VideoWithReason;
  onStatusChange: (videoId: string, status: string | null, layoutType?: string) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const statusOptions = [
    {
      label: "Approve",
      status: "approved" as const,
      layoutType: undefined,
      disabled: video.screening_status === "approved" && video.layout_type !== "otb_only",
    },
    {
      label: "OTB Only",
      status: "approved" as const,
      layoutType: "otb_only",
      disabled: video.screening_status === "approved" && video.layout_type === "otb_only",
    },
    {
      label: "Reject",
      status: "rejected" as const,
      layoutType: undefined,
      disabled: video.screening_status === "rejected",
    },
    {
      label: "Unscreen",
      status: null,
      layoutType: undefined,
      disabled: video.screening_status === null,
    },
  ];

  return (
    <div className="relative flex-shrink-0" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 cursor-pointer"
      >
        {statusBadge(video.screening_status, video.layout_type)}
        {video.inspect_reason && (
          <span className="text-[10px] text-muted-foreground">{video.inspect_reason}</span>
        )}
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`w-3 h-3 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-[60] rounded-lg border bg-background shadow-lg p-1 min-w-[130px]">
          {statusOptions.map((opt) => (
            <button
              key={opt.label}
              disabled={opt.disabled}
              onClick={() => {
                onStatusChange(video.video_id, opt.status, opt.layoutType);
                setOpen(false);
              }}
              className="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-muted transition-colors disabled:opacity-40 disabled:pointer-events-none"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────

export default function VideoScreener({
  channels,
  initialChannelId,
}: VideoScreenerProps) {
  const [channelId, setChannelId] = useState<string | null>(initialChannelId);
  const [videos, setVideos] = useState<VideoWithReason[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string | null>("unscreened");
  const [orderBy, setOrderBy] = useState("title_score_desc");
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [batchJob, setBatchJob] = useState<{
    id: string;
    total: number;
    completed: number;
  } | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const [pageSize, setPageSize] = useState(() => computePageSize());

  // Dropdown state
  const [sortOpen, setSortOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);

  // Recompute page size on resize
  useEffect(() => {
    const onResize = () => setPageSize(computePageSize());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    if (initialChannelId) setChannelId(initialChannelId);
  }, [initialChannelId]);

  const addToast = useCallback((toast: Omit<Toast, "id">) => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { ...toast, id }]);
    const timer = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
      toastTimers.current.delete(id);
    }, 6000);
    toastTimers.current.set(id, timer);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = toastTimers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      toastTimers.current.delete(id);
    }
  }, []);

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
        order_by: orderBy === "title_score_desc" ? undefined : orderBy,
        limit: pageSize,
        offset: 0,
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
  }, [channelId, statusFilter, orderBy, pageSize, addToast]);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  useEffect(() => {
    setSelected(new Set());
  }, [channelId, statusFilter]);

  // In-place status update (videos don't disappear)
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
        updateVideoInPlace(videoId, status, layoutType ?? null);
        loadCounts();

        const label =
          layoutType === "otb_only"
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
            updateVideoInPlace(videoId, previousStatus, previousLayout);
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
    [videos, updateVideoInPlace, loadCounts, addToast]
  );

  const handleRejectAllUnmarked = useCallback(async () => {
    const unmarkedIds = videos
      .filter((v) => v.screening_status === null)
      .map((v) => v.video_id);
    if (unmarkedIds.length === 0) return;

    try {
      await batchUpdateVideoStatus(unmarkedIds, "rejected");
      const idSet = new Set(unmarkedIds);
      setVideos((prev) =>
        prev.map((v) =>
          idSet.has(v.video_id) ? { ...v, screening_status: "rejected" } : v
        )
      );
      loadCounts();

      addToast({
        message: `Rejected ${unmarkedIds.length} unmarked video${unmarkedIds.length !== 1 ? "s" : ""}`,
        type: "success",
        undoAction: async () => {
          await undoAutoReject(unmarkedIds);
          setVideos((prev) =>
            prev.map((v) =>
              idSet.has(v.video_id) ? { ...v, screening_status: null } : v
            )
          );
          loadCounts();
        },
      });
    } catch (e) {
      addToast({
        message: e instanceof Error ? e.message : "Reject all failed",
        type: "error",
      });
    }
  }, [videos, loadCounts, addToast]);

  const handleAutomaticInspect = useCallback(async () => {
    const allIds = videos
      .filter((v) => v.screening_status === null)
      .map((v) => v.video_id);
    if (allIds.length === 0) {
      addToast({ message: "No unscreened videos to inspect", type: "warning" });
      return;
    }
    try {
      const { job_id } = await batchInspectVideos(allIds);
      setBatchJob({ id: job_id, total: allIds.length, completed: 0 });

      const poll = setInterval(async () => {
        try {
          const status = await getInspectJobStatus(job_id);
          setBatchJob({
            id: job_id,
            total: status.total,
            completed: status.completed,
          });

          if (status.results) {
            for (const r of status.results) {
              let reason = "";
              if (r.approved) {
                reason = "";
              } else if (r.error) {
                reason = "inspect failed";
              } else if (!r.has_overlay) {
                reason = "no overlay";
              } else if (!r.has_otb) {
                reason = "no OTB";
              } else {
                reason = "rejected";
              }

              updateVideoInPlace(
                r.video_id,
                r.status ?? (r.approved ? "approved" : "rejected"),
                undefined,
                reason
              );
            }
          }

          if (status.status === "done") {
            clearInterval(poll);
            setBatchJob(null);
            addToast({
              message: `Inspected ${status.total}: ${status.approved} approved, ${status.rejected} rejected${status.failed ? `, ${status.failed} failed` : ""}`,
              type: "success",
            });
            loadCounts();
          }
        } catch {
          clearInterval(poll);
          setBatchJob(null);
          addToast({
            message: "Failed to poll inspection job",
            type: "error",
          });
        }
      }, 2000);
    } catch (e) {
      addToast({
        message: e instanceof Error ? e.message : "Batch inspection failed",
        type: "error",
      });
    }
  }, [videos, updateVideoInPlace, loadCounts, addToast]);

  const toggleSelect = (videoId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) next.delete(videoId);
      else next.add(videoId);
      return next;
    });
  };

  // Client-side sorting
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

  const allScreened =
    videos.length > 0 && videos.every((v) => v.screening_status !== null);

  const hasActiveFilter = statusFilter !== null || channelId !== null;

  return (
    <div className="relative h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
      {/* Floating Toasts */}
      {toasts.length > 0 && (
        <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className={`rounded-lg border shadow-lg p-3 text-sm flex items-center gap-3 animate-in slide-in-from-top-2 ${
                toast.type === "error"
                  ? "bg-destructive/10 border-destructive/30 text-destructive"
                  : toast.type === "warning"
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-400"
                  : "bg-background border-border"
              }`}
            >
              <span className="flex-1">{toast.message}</span>
              {toast.undoAction && (
                <button
                  onClick={async () => {
                    await toast.undoAction!();
                    removeToast(toast.id);
                  }}
                  className="text-xs font-medium text-primary underline whitespace-nowrap"
                >
                  Undo
                </button>
              )}
              <button
                onClick={() => removeToast(toast.id)}
                className="text-muted-foreground hover:text-foreground text-xs"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}

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
                    {statusCounts[f.value ?? "all"] !== undefined && (
                      <span className="ml-1 opacity-60">
                        ({statusCounts[f.value ?? "all"]})
                      </span>
                    )}
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

          {/* Accept (green pill) */}
          <button
            onClick={loadVideos}
            disabled={!allScreened && videos.length > 0}
            className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-green-600 text-white hover:bg-green-700 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
          >
            <CheckIcon className="w-3.5 h-3.5" />
            Accept
          </button>

          {/* Automatic (blue pill) */}
          <button
            onClick={handleAutomaticInspect}
            disabled={!!batchJob}
            className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
          >
            <RobotIcon className="w-3.5 h-3.5" />
            {batchJob ? `${batchJob.completed}/${batchJob.total}` : "Automatic"}
          </button>

          {/* Reject All (red pill) */}
          <button
            onClick={handleRejectAllUnmarked}
            disabled={!videos.some((v) => v.screening_status === null)}
            className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
          >
            <TrashIcon className="w-3.5 h-3.5" />
            Reject All
          </button>
        </div>

        {/* Batch inspection progress */}
        {batchJob && (
          <div className="mt-1 h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{
                width: `${(batchJob.completed / batchJob.total) * 100}%`,
              }}
            />
          </div>
        )}
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
        <div className="flex-1 grid grid-cols-2 xl:grid-cols-3 gap-2 auto-rows-min content-start overflow-hidden">
          {sortedVideos.map((v) => {
            const isScreened = v.screening_status !== null;
            const isUnscreened = v.screening_status === null;

            return (
              <div
                key={v.video_id}
                className={`border rounded-lg bg-card transition-opacity ${
                  isScreened && statusFilter === "unscreened" ? "opacity-60" : ""
                } ${selected.has(v.video_id) ? "ring-2 ring-primary" : ""}`}
              >
                {/* Header: title + score + status */}
                <div className="px-2 py-1 flex items-center gap-1.5 min-w-0">
                  <input
                    type="checkbox"
                    checked={selected.has(v.video_id)}
                    onChange={() => toggleSelect(v.video_id)}
                    className="rounded flex-shrink-0"
                  />
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
                  {/* Status (for screened videos) */}
                  {!isUnscreened && (
                    <StatusDropdown video={v} onStatusChange={handleStatusChange} />
                  )}
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

                {/* Actions (for unscreened videos) */}
                {isUnscreened && (
                  <div className="px-2 py-1 flex items-center gap-1 justify-end">
                    <UnscreenedActions videoId={v.video_id} onStatusChange={handleStatusChange} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
