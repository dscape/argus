import type {
  OverlayTestResponse,
  CalibrationEntry,
  ClipInspectResponse,
  VideoSession,
  FrameOverlayResponse,
  VideoMoveDetectionResponse,
  SyntheticScanResponse,
  SyntheticStatsResponse,
  CrawlChannel,
  CrawlChannelDetail,
  CrawlVideosResponse,
  QuotaStatus,
  InspectResult,
  InspectJobStatus,
} from "./types";

// ── Overlay ─────────────────────────────────────────────────

export async function testOverlayImage(
  image: File,
  options?: {
    overlay_bbox?: string;
    flipped?: boolean;
    theme?: string;
  }
): Promise<OverlayTestResponse> {
  const form = new FormData();
  form.append("image", image);
  if (options?.overlay_bbox) form.append("overlay_bbox", options.overlay_bbox);
  if (options?.flipped) form.append("flipped", "true");
  if (options?.theme) form.append("theme", options.theme);

  const res = await fetch("/api/overlay/test-image", {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Calibration ─────────────────────────────────────────────

export async function listCalibrations(): Promise<CalibrationEntry[]> {
  const res = await fetch("/api/calibration/");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.calibrations;
}

export async function saveCalibration(
  channel: string,
  calibration: Omit<CalibrationEntry, "channel_handle">
): Promise<void> {
  const res = await fetch(`/api/calibration/${encodeURIComponent(channel)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(calibration),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function deleteCalibration(channel: string): Promise<void> {
  const res = await fetch(`/api/calibration/${encodeURIComponent(channel)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

// ── Clips ───────────────────────────────────────────────────

export async function loadClip(
  file: File
): Promise<{ session_id: string }> {
  const form = new FormData();
  form.append("clip_file", file);
  const res = await fetch("/api/clips/load", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getClipInfo(
  sessionId: string
): Promise<ClipInspectResponse> {
  const res = await fetch(`/api/clips/${sessionId}/info`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function clipFrameUrl(sessionId: string, index: number): string {
  return `/api/clips/${sessionId}/frame/${index}`;
}

export async function deleteClipSession(sessionId: string): Promise<void> {
  await fetch(`/api/clips/${sessionId}`, { method: "DELETE" });
}

// ── Video ───────────────────────────────────────────────────

export async function openVideo(
  videoPath: string,
  channelHandle?: string
): Promise<VideoSession> {
  const res = await fetch("/api/video/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_path: videoPath,
      channel_handle: channelHandle,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function videoFrameUrl(sessionId: string, index: number): string {
  return `/api/video/${sessionId}/frame?index=${index}`;
}

export async function readOverlayFrame(
  sessionId: string,
  index: number
): Promise<FrameOverlayResponse> {
  const res = await fetch(
    `/api/video/${sessionId}/overlay-read?index=${index}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function detectVideoMoves(
  sessionId: string,
  sampleFps: number = 2.0
): Promise<VideoMoveDetectionResponse> {
  const res = await fetch(`/api/video/${sessionId}/detect-moves`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_fps: sampleFps }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteVideoSession(sessionId: string): Promise<void> {
  await fetch(`/api/video/${sessionId}`, { method: "DELETE" });
}

// ── Crawl ───────────────────────────────────────────────────

export async function listCrawlChannels(): Promise<CrawlChannel[]> {
  const res = await fetch("/api/crawl/channels");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function addCrawlChannel(handle: string): Promise<CrawlChannel> {
  const res = await fetch("/api/crawl/channels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ handle }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function toggleCrawlChannel(
  channelId: string,
  enabled: boolean
): Promise<CrawlChannel> {
  const res = await fetch(
    `/api/crawl/channels/${encodeURIComponent(channelId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getCrawlChannelDetail(
  channelId: string
): Promise<CrawlChannelDetail> {
  const res = await fetch(
    `/api/crawl/channels/${encodeURIComponent(channelId)}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function crawlChannel(
  channelId: string
): Promise<{ channel_id: string; new_videos: number }> {
  const res = await fetch(
    `/api/crawl/channels/${encodeURIComponent(channelId)}/crawl`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function crawlAllChannels(): Promise<{
  channels_crawled: number;
  total_new_videos: number;
}> {
  const res = await fetch("/api/crawl/crawl-all", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getVideoCounts(
  channelId?: string
): Promise<Record<string, number>> {
  const qs = new URLSearchParams();
  if (channelId) qs.set("channel_id", channelId);
  const res = await fetch(`/api/crawl/videos/counts?${qs}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listCrawlVideos(params: {
  channel_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<CrawlVideosResponse> {
  const qs = new URLSearchParams();
  if (params.channel_id) qs.set("channel_id", params.channel_id);
  if (params.status) qs.set("status", params.status);
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  if (params.offset !== undefined) qs.set("offset", String(params.offset));
  const res = await fetch(`/api/crawl/videos?${qs}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateVideoStatus(
  videoId: string,
  status: string | null
): Promise<void> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/status`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
}

export async function batchUpdateVideoStatus(
  videoIds: string[],
  status: string
): Promise<{ updated: number }> {
  const res = await fetch("/api/crawl/videos/batch-status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_ids: videoIds, status }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function screenVideos(params?: {
  channel_id?: string;
  limit?: number;
}): Promise<{ screened: number; candidates: number; rejected: number }> {
  const res = await fetch("/api/crawl/screen", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getQuotaStatus(): Promise<QuotaStatus> {
  const res = await fetch("/api/crawl/quota");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function autoClassifyTitles(params?: {
  channel_id?: string;
  limit?: number;
}): Promise<{ classified: number; candidates: number; rejected: number }> {
  const res = await fetch("/api/crawl/auto-classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Inspection ──────────────────────────────────────────────

export async function inspectVideo(
  videoId: string
): Promise<InspectResult> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/inspect`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function batchInspectVideos(
  videoIds: string[]
): Promise<{ job_id: string }> {
  const res = await fetch("/api/crawl/videos/batch-inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_ids: videoIds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getInspectJobStatus(
  jobId: string
): Promise<InspectJobStatus> {
  const res = await fetch(
    `/api/crawl/videos/inspect-job/${encodeURIComponent(jobId)}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── AI Classification ───────────────────────────────────────

export async function classifyTitles(
  videoIds: string[]
): Promise<{ prompt: string }> {
  const res = await fetch("/api/crawl/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_ids: videoIds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Synthetic ────────────────────────────────────────────────

export async function scanSyntheticDir(
  directory: string,
  expectedClips?: number
): Promise<SyntheticScanResponse> {
  const params = new URLSearchParams({ directory });
  if (expectedClips !== undefined) params.set("expected_clips", String(expectedClips));
  const res = await fetch(`/api/synthetic/scan?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSyntheticStats(
  directory: string
): Promise<SyntheticStatsResponse> {
  const params = new URLSearchParams({ directory });
  const res = await fetch(`/api/synthetic/stats?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function inspectSyntheticClip(
  filepath: string
): Promise<{ session_id: string }> {
  const res = await fetch("/api/synthetic/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filepath }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
