"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  listCrawlVideos,
  updateVideoStatus,
  batchUpdateVideoStatus,
  getVideoCounts,
  undoAutoReject,
} from "@/lib/api";
import type { CrawlChannel, CrawlVideo } from "@/lib/types";
import {
  VideoWithReason,
  CheckIcon,
  RobotIcon,
  TrashIcon,
  SpinnerIcon,
  scoreColor,
  youtubeThumb,
  cardTintClass,
  computePageSize,
  useToasts,
  ToastContainer,
  UnscreenedActions,
  StatusDropdown,
} from "@/components/video-shared";

interface VideoScreenerPageProps {
  channels: CrawlChannel[];
}

export default function VideoScreenerPage({ channels }: VideoScreenerPageProps) {
  const [videos, setVideos] = useState<VideoWithReason[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [pageSize, setPageSize] = useState(() => computePageSize());
  const [autoRunning, setAutoRunning] = useState(false);
  const { toasts, addToast, removeToast } = useToasts();

  // Recompute page size on resize
  useEffect(() => {
    const onResize = () => setPageSize(computePageSize());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Ref to track whether automatic should run after load
  const shouldAutoRejectRef = useRef(false);

  const loadCounts = useCallback(async () => {
    try {
      await getVideoCounts();
    } catch {
      // best-effort
    }
  }, []);

  const runAutomaticReject = useCallback(async (vids: VideoWithReason[]) => {
    const unscreened = vids.filter((v) => v.screening_status === null);
    if (unscreened.length === 0) return;

    const toReject = unscreened.filter((v) => !v.title_is_candidate);
    const toKeep = unscreened.filter((v) => v.title_is_candidate);

    if (toReject.length === 0) {
      for (const v of toKeep) {
        setVideos((prev) =>
          prev.map((pv) =>
            pv.video_id === v.video_id ? { ...pv, inspect_reason: "title \u2713" } : pv
          )
        );
      }
      return;
    }

    const rejectIds = toReject.map((v) => v.video_id);

    try {
      await batchUpdateVideoStatus(rejectIds, "rejected");
      const idSet = new Set(rejectIds);

      setVideos((prev) =>
        prev.map((v) => {
          if (idSet.has(v.video_id)) {
            return { ...v, screening_status: "rejected", inspect_reason: `title ${v.title_score.toFixed(2)}` };
          }
          if (v.screening_status === null && v.title_is_candidate) {
            return { ...v, inspect_reason: "title \u2713" };
          }
          return v;
        })
      );

      await loadCounts();

      addToast({
        message: `Auto-rejected ${toReject.length} by title, ${toKeep.length} kept`,
        type: "success",
        undoAction: async () => {
          await undoAutoReject(rejectIds);
          setVideos((prev) =>
            prev.map((v) =>
              idSet.has(v.video_id) ? { ...v, screening_status: null, inspect_reason: undefined } : v
            )
          );
          loadCounts();
        },
      });
    } catch (e) {
      addToast({
        message: e instanceof Error ? e.message : "Auto reject failed",
        type: "error",
      });
    }
  }, [loadCounts, addToast]);

  const loadVideos = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listCrawlVideos({
        status: "unscreened",
        limit: pageSize,
        offset: 0,
      });
      setVideos(data.videos);
      setTotal(data.total);
      setSelected(new Set());

      if (data.videos.some((v: CrawlVideo) => v.screening_status === null)) {
        shouldAutoRejectRef.current = true;
      }
    } catch (e) {
      addToast({
        message: e instanceof Error ? e.message : "Failed to load videos",
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  }, [pageSize, addToast]);

  // After videos load, run automatic reject if flagged
  useEffect(() => {
    if (!loading && shouldAutoRejectRef.current && videos.length > 0) {
      shouldAutoRejectRef.current = false;
      setAutoRunning(true);
      runAutomaticReject(videos).finally(() => setAutoRunning(false));
    }
  }, [loading, videos, runAutomaticReject]);

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

  const handleRejectAction = useCallback(async () => {
    const hasSelection = selected.size > 1;

    if (hasSelection) {
      // Reject only selected unscreened videos
      const selectedIds = Array.from(selected).filter((id) => {
        const v = videos.find((v) => v.video_id === id);
        return v && v.screening_status === null;
      });
      if (selectedIds.length === 0) return;

      try {
        await batchUpdateVideoStatus(selectedIds, "rejected");
        const idSet = new Set(selectedIds);
        setVideos((prev) =>
          prev.map((v) =>
            idSet.has(v.video_id) ? { ...v, screening_status: "rejected" } : v
          )
        );
        setSelected(new Set());
        loadCounts();

        addToast({
          message: `Rejected ${selectedIds.length} selected video${selectedIds.length !== 1 ? "s" : ""}`,
          type: "success",
          undoAction: async () => {
            await undoAutoReject(selectedIds);
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
          message: e instanceof Error ? e.message : "Reject selected failed",
          type: "error",
        });
      }
    } else {
      // Reject all unmarked
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
    }
  }, [selected, videos, loadCounts, addToast]);

  const handleManualAutoReject = useCallback(async () => {
    setAutoRunning(true);
    await runAutomaticReject(videos);
    setAutoRunning(false);
  }, [videos, runAutomaticReject]);

  const toggleSelect = (videoId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) next.delete(videoId);
      else next.add(videoId);
      return next;
    });
  };

  const sortedVideos = useMemo(() => {
    const sorted = [...videos];
    return sorted.sort((a, b) => b.title_score - a.title_score);
  }, [videos]);

  const allScreened =
    videos.length > 0 && videos.every((v) => v.screening_status !== null);

  const hasSelection = selected.size > 1;

  return (
    <div className="relative h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
      {/* Pulse animation for commit button */}
      <style>{`
        @keyframes commit-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.5); }
          50% { box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
        }
        .commit-ready {
          animation: commit-pulse 1.5s ease-in-out infinite;
        }
      `}</style>

      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* Sticky Toolbar */}
      <div className="flex-shrink-0 z-40 bg-background/95 backdrop-blur pb-1 pt-1 mb-1">
        <div className="flex items-center gap-1 rounded-2xl border bg-muted/30 p-1.5">
          {/* Showing X of N */}
          <span className="text-xs text-muted-foreground px-2 tabular-nums">
            Showing {videos.length} of {total.toLocaleString()}
          </span>

          <div className="flex-1" />

          {/* Commit (green pill) — loads next batch */}
          <button
            onClick={loadVideos}
            disabled={(!allScreened && videos.length > 0) || loading}
            className={`flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-green-600 text-white hover:bg-green-700 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none ${
              allScreened ? "commit-ready" : ""
            }`}
          >
            <CheckIcon className="w-3.5 h-3.5" />
            Commit
          </button>

          {/* Automatic (blue pill) — reject by title only */}
          <button
            onClick={handleManualAutoReject}
            className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
            disabled={autoRunning || !videos.some((v) => v.screening_status === null)}
          >
            {autoRunning ? (
              <SpinnerIcon className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <RobotIcon className="w-3.5 h-3.5" />
            )}
            {autoRunning ? "Running..." : "Automatic"}
          </button>

          {/* Reject Remaining / Reject Selected (red pill) */}
          <button
            onClick={handleRejectAction}
            disabled={!videos.some((v) => v.screening_status === null)}
            className="flex items-center gap-1.5 px-3 h-8 rounded-xl text-xs font-medium bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none"
          >
            <TrashIcon className="w-3.5 h-3.5" />
            {hasSelection ? "Reject Selected" : "Reject Remaining"}
          </button>
        </div>
      </div>

      {/* Video grid */}
      {loading ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Loading videos...
        </div>
      ) : videos.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center">
          No videos to screen. Crawl channels first.
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 xl:grid-cols-3 gap-2 auto-rows-min content-start overflow-hidden">
          {sortedVideos.map((v) => {
            const isUnscreened = v.screening_status === null;
            const tint = cardTintClass(v.screening_status, v.layout_type, true);

            return (
              <div
                key={v.video_id}
                className={`border rounded-lg bg-card transition-all duration-300 ${tint} ${
                  selected.has(v.video_id) ? "ring-2 ring-primary" : ""
                }`}
              >
                {/* Header: checkbox + title + score + status */}
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
