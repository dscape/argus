"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import {
  getVideo,
  getDownloadStatus,
  downloadVideo,
  openVideo,
  listVideoClips,
  updateVideoStatus,
  getAssetStatus,
  fetchAssets,
} from "@/lib/api";
import type { AssetStatus } from "@/lib/api";
import type { CrawlVideo, DownloadStatus, VideoSession, VideoClip } from "@/lib/types";

interface VideoWorkbenchContextValue {
  videoId: string;
  video: CrawlVideo | null;
  session: VideoSession | null;
  clips: VideoClip[];
  assets: AssetStatus | null;
  assetsLoading: boolean;
  downloadReady: boolean;
  downloadStatus: DownloadStatus | null;
  error: string | null;
  refreshClips: () => Promise<void>;
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
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [session, setSession] = useState<VideoSession | null>(null);
  const [clips, setClips] = useState<VideoClip[]>([]);

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
        let status = await getAssetStatus(videoId);
        if (cancelled) return;
        setAssets(status);
        if (status.lores.length < 3 || status.hires.length < 3) {
          await fetchAssets(videoId);
          if (cancelled) return;
          status = await getAssetStatus(videoId);
          if (cancelled) return;
          setAssets(status);
        }
        if (!status.video) {
          try {
            await downloadVideo(videoId);
            if (cancelled) return;
            status = await getAssetStatus(videoId);
            if (cancelled) return;
            setAssets(status);
          } catch { /* video download may fail, that's ok */ }
        }
      } catch { /* ignore asset fetch errors */ }
      if (!cancelled) setAssetsLoading(false);
    })();
    return () => { cancelled = true; };
  }, [video, videoId]);

  // Fetch download status
  useEffect(() => {
    getDownloadStatus(videoId).then(setDownloadStatus);
  }, [videoId]);

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

  const downloadReady = assets != null && assets.video && assets.lores.length >= 3 && assets.hires.length >= 3;

  return (
    <VideoWorkbenchContext.Provider value={{
      videoId,
      video,
      session,
      clips,
      assets,
      assetsLoading,
      downloadReady,
      downloadStatus,
      error,
      refreshClips,
      setClips,
      handleStatusChange,
    }}>
      {children}
    </VideoWorkbenchContext.Provider>
  );
}
