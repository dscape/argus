"use client";

import { createContext, useContext, useState, useEffect, useCallback, useMemo } from "react";
import {
  getVideo,
  getDownloadStatus,
  downloadVideo as requestVideoDownload,
  openVideo,
  listVideoClips,
  updateVideoStatus,
  getAssetStatus,
  fetchAssets,
} from "@/lib/api";
import type { AssetStatus } from "@/lib/api";
import type { CrawlVideo, DownloadStatus, VideoSession, VideoClip } from "@/lib/types";

const activeVideoDownloads = new Map<string, Promise<void>>();

interface VideoWorkbenchContextValue {
  videoId: string;
  video: CrawlVideo | null;
  session: VideoSession | null;
  clips: VideoClip[];
  activeClips: VideoClip[];
  assets: AssetStatus | null;
  assetsLoading: boolean;
  downloadReady: boolean;
  downloadBusy: boolean;
  downloadStatus: DownloadStatus | null;
  error: string | null;
  refreshClips: () => Promise<void>;
  refreshDownloadStatus: () => Promise<DownloadStatus>;
  startDownload: () => Promise<void>;
  setClips: React.Dispatch<React.SetStateAction<VideoClip[]>>;
  handleStatusChange: (vid: string, status: string | null, layoutType?: string) => Promise<void>;
}

const VideoWorkbenchContext = createContext<VideoWorkbenchContextValue | null>(null);

export function useVideoWorkbench() {
  const ctx = useContext(VideoWorkbenchContext);
  if (!ctx) throw new Error("useVideoWorkbench must be used within VideoWorkbenchProvider");
  return ctx;
}

export function VideoWorkbenchProvider({ videoId, children }: { videoId: string; children: React.ReactNode }) {
  const [video, setVideo] = useState<CrawlVideo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [assets, setAssets] = useState<AssetStatus | null>(null);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [downloadBusy, setDownloadBusy] = useState(activeVideoDownloads.has(videoId));
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [clips, setClips] = useState<VideoClip[]>([]);

  const refreshAssets = useCallback(async (): Promise<AssetStatus> => {
    const status = await getAssetStatus(videoId);
    setAssets(status);
    return status;
  }, [videoId]);

  const refreshDownloadStatus = useCallback(async (): Promise<DownloadStatus> => {
    const status = await getDownloadStatus(videoId);
    setDownloadStatus(status);
    return status;
  }, [videoId]);

  const startDownload = useCallback(async () => {
    const existing = activeVideoDownloads.get(videoId);
    if (existing) {
      setDownloadBusy(true);
      try {
        await existing;
        await Promise.allSettled([refreshAssets(), refreshDownloadStatus()]);
      } finally {
        setDownloadBusy(false);
      }
      return;
    }

    const request = (async () => {
      const result = await requestVideoDownload(videoId);
      setDownloadStatus({
        downloaded: true,
        path: result.path,
        file_size_mb: result.file_size_mb,
        duration_seconds: null,
      });
      await Promise.allSettled([refreshAssets(), refreshDownloadStatus()]);
    })();

    activeVideoDownloads.set(videoId, request);
    setDownloadBusy(true);

    try {
      await request;
    } finally {
      if (activeVideoDownloads.get(videoId) === request) {
        activeVideoDownloads.delete(videoId);
      }
      setDownloadBusy(false);
    }
  }, [videoId, refreshAssets, refreshDownloadStatus]);

  // Fetch video metadata
  useEffect(() => {
    getVideo(videoId)
      .then(setVideo)
      .catch((e) => setError(e.message));
  }, [videoId]);

  // Auto-download assets on visit
  useEffect(() => {
    if (!video) return;
    let cancelled = false;
    (async () => {
      setAssetsLoading(true);
      try {
        let status = await refreshAssets();
        if (cancelled) return;
        if (status.lores.length < 3 || status.hires.length < 3) {
          await fetchAssets(videoId);
          if (cancelled) return;
          status = await refreshAssets();
          if (cancelled) return;
        }
        if (!status.video) {
          try {
            await startDownload();
            if (cancelled) return;
            status = await refreshAssets();
            if (cancelled) return;
          } catch {
            // video download may fail, that's ok
          }
        }
        await refreshDownloadStatus();
      } catch {
        // ignore asset fetch errors
      }
      if (!cancelled) setAssetsLoading(false);
    })();
    return () => { cancelled = true; };
  }, [video, videoId, refreshAssets, refreshDownloadStatus, startDownload]);

  // Fetch download status
  useEffect(() => {
    refreshDownloadStatus().catch(() => undefined);
  }, [refreshDownloadStatus]);

  // Open video session when download is ready
  useEffect(() => {
    if (downloadStatus?.downloaded && downloadStatus.path && !session) {
      openVideo(downloadStatus.path, video?.channel_handle || undefined)
        .then(setSession)
        .catch((e) => setError(e.message));
    }
  }, [downloadStatus, video?.channel_handle, session]);

  // Fetch clips
  useEffect(() => {
    listVideoClips(videoId).then(setClips);
  }, [videoId]);

  const refreshClips = useCallback(async () => {
    const updated = await listVideoClips(videoId);
    setClips(updated);
  }, [videoId]);

  const handleStatusChange = useCallback(async (vid: string, status: string | null, layoutType?: string) => {
    await updateVideoStatus(vid, status, layoutType);
    setVideo((prev) => prev ? { ...prev, screening_status: status, layout_type: layoutType ?? null } : prev);
  }, []);

  const downloadReady = Boolean(
    (assets?.video || downloadStatus?.downloaded) &&
    assets?.lores.length === 3 &&
    assets?.hires.length === 3
  );
  const activeClips = useMemo(() => clips.filter((c) => !c.is_gap), [clips]);

  return (
    <VideoWorkbenchContext.Provider value={{
      videoId,
      video,
      session,
      clips,
      activeClips,
      assets,
      assetsLoading,
      downloadReady,
      downloadBusy,
      downloadStatus,
      error,
      refreshClips,
      refreshDownloadStatus,
      startDownload,
      setClips,
      handleStatusChange,
    }}>
      {children}
    </VideoWorkbenchContext.Provider>
  );
}
