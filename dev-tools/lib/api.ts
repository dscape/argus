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
  CrawlVideo,
  CrawlVideosResponse,
  QuotaStatus,
  InspectResult,
  InspectJobStatus,
  VideoAnnotations,
  VideoClip,
  DownloadStatus,
  DownloadResult,
  GenerateClipsResponse,
  AiScreenResult,
  GenerationStatus,
  AutoSegmentResponse,
  AutoCalibrateResponse,
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

export async function testOverlayUrl(
  url: string,
  options?: {
    overlay_bbox?: string;
    flipped?: boolean;
    theme?: string;
  }
): Promise<OverlayTestResponse> {
  const body: Record<string, unknown> = { url };
  if (options?.overlay_bbox) body.overlay_bbox = options.overlay_bbox;
  if (options?.flipped) body.flipped = options.flipped;
  if (options?.theme) body.theme = options.theme;

  const res = await fetch("/api/overlay/test-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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
  index: number,
  clipId?: number
): Promise<FrameOverlayResponse> {
  let url = `/api/video/${sessionId}/overlay-read?index=${index}`;
  if (clipId !== undefined) url += `&clip_id=${clipId}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function detectVideoMoves(
  sessionId: string,
  sampleFps: number = 2.0,
  clipId?: number
): Promise<VideoMoveDetectionResponse> {
  const body: Record<string, unknown> = { sample_fps: sampleFps };
  if (clipId !== undefined) body.clip_id = clipId;
  const res = await fetch(`/api/video/${sessionId}/detect-moves`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteVideoSession(sessionId: string): Promise<void> {
  await fetch(`/api/video/${sessionId}`, { method: "DELETE" });
}

// ── Crawl ───────────────────────────────────────────────────

export async function listCrawlChannels(opts?: { screenedOnly?: boolean }): Promise<CrawlChannel[]> {
  const params = new URLSearchParams();
  if (opts?.screenedOnly) params.set("screened_only", "true");
  const qs = params.toString();
  const res = await fetch(`/api/crawl/channels${qs ? `?${qs}` : ""}`);
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
  layout_type?: string;
  order_by?: string;
  limit?: number;
  offset?: number;
  video_ids?: string[];
}): Promise<CrawlVideosResponse> {
  const qs = new URLSearchParams();
  if (params.channel_id) qs.set("channel_id", params.channel_id);
  if (params.status) qs.set("status", params.status);
  if (params.layout_type) qs.set("layout_type", params.layout_type);
  if (params.order_by) qs.set("order_by", params.order_by);
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  if (params.offset !== undefined) qs.set("offset", String(params.offset));
  if (params.video_ids && params.video_ids.length > 0) qs.set("video_ids", params.video_ids.join(","));
  const res = await fetch(`/api/crawl/videos?${qs}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateVideoStatus(
  videoId: string,
  status: string | null,
  layoutType?: string
): Promise<void> {
  const body: Record<string, unknown> = { status };
  if (layoutType) body.layout_type = layoutType;
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/status`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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

export async function undoAutoReject(
  videoIds: string[]
): Promise<{ restored: number }> {
  const res = await fetch("/api/crawl/videos/undo-auto-reject", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_ids: videoIds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getVideoAnnotations(
  videoId: string
): Promise<{ video_id: string; annotations: VideoAnnotations | null }> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/annotations`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function saveVideoAnnotations(
  videoId: string,
  annotations: VideoAnnotations
): Promise<{ video_id: string; annotations: VideoAnnotations }> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/annotations`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotations),
    }
  );
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
}): Promise<{ classified: number; approved: number; rejected: number }> {
  const res = await fetch("/api/crawl/auto-classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Single video ────────────────────────────────────────────

export async function getVideo(videoId: string): Promise<CrawlVideo> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDownloadStatus(
  videoId: string
): Promise<DownloadStatus> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/download-status`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Video Clips ────────────────────────────────────────────

export async function listVideoClips(videoId: string): Promise<VideoClip[]> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/clips`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createVideoClip(
  videoId: string,
  clip: Omit<VideoClip, "id" | "video_id" | "clip_index">
): Promise<VideoClip> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/clips`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clip),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateVideoClip(
  videoId: string,
  clipId: number,
  updates: Partial<VideoClip>
): Promise<VideoClip> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/clips/${clipId}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteVideoClip(
  videoId: string,
  clipId: number
): Promise<void> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/clips/${clipId}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(await res.text());
}

// ── Auto-segment & auto-calibrate ─────────────────────────

export async function autoSegmentVideo(
  videoId: string,
  opts?: { sampleIntervalSec?: number; replaceExisting?: boolean }
): Promise<AutoSegmentResponse> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/auto-segment`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_interval_sec: opts?.sampleIntervalSec ?? 30.0,
        replace_existing: opts?.replaceExisting ?? false,
      }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function autoCalibrateClip(
  videoId: string,
  clipId: number
): Promise<AutoCalibrateResponse> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/clips/${clipId}/auto-calibrate`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function downloadVideo(
  videoId: string
): Promise<DownloadResult> {
  const res = await fetch(
    `/api/crawl/videos/${encodeURIComponent(videoId)}/download`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Clip generation ─────────────────────────────────────────

export async function generateClips(
  sessionId: string,
  clipId?: number
): Promise<GenerateClipsResponse> {
  const body: Record<string, unknown> = {};
  if (clipId !== undefined) body.clip_id = clipId;
  const res = await fetch(`/api/video/${sessionId}/generate-clips`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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

// ── AI Screening ────────────────────────────────────────────

export async function aiScreenBatch(
  videoIds: string[],
  threshold?: number
): Promise<{ results: AiScreenResult[] }> {
  const body: Record<string, unknown> = { video_ids: videoIds };
  if (threshold !== undefined) body.threshold = threshold;
  const res = await fetch("/api/crawl/ai-screen", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Screening Sessions ─────────────────────────────────────

export interface ScreeningSession {
  id: string;
  created_at: string;
  sample_size: number;
  model_version: string | null;
  accuracy: number | null;
  per_class?: Record<string, { correct: number; total: number }> | null;
  results?: unknown[];
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}

export async function createScreeningSession(
  body: {
    results: unknown[];
    accuracy?: number | null;
    sample_size?: number;
    per_class?: Record<string, { correct: number; total: number }> | null;
    model_version?: string | null;
    pin_state?: Record<string, boolean>;
    evaluation_id?: number | null;
  }
): Promise<{ session_id: string; created_at: string }> {
  const res = await fetch("/api/models/ai-screening/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getScreeningSession(sessionId: string): Promise<ScreeningSession> {
  const res = await fetch(`/api/models/ai-screening/sessions/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listScreeningSessions(limit = 20): Promise<{ sessions: ScreeningSession[] }> {
  const res = await fetch(`/api/models/ai-screening/sessions?limit=${limit}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateSessionPins(
  sessionId: string,
  pinState: Record<string, boolean>
): Promise<{ pin_state: Record<string, boolean> }> {
  const res = await fetch(`/api/models/ai-screening/sessions/${sessionId}/pins`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pin_state: pinState }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Overlay Test Sessions ──────────────────────────────────

export interface OverlayTestResult {
  filename: string;
  /** "synthetic" = chess-positions dataset; "real" = pre-extracted video overlay crop */
  source?: "synthetic" | "real";
  expected_fen: string | null;
  predicted_fen: string | null;
  match: boolean;
  piece_accuracy: number;
  square_diffs: string[];
  board_image_b64?: string;
  elapsed_ms: number;
  overlay_detect_ms?: number;
  grid_detect_ms?: number;
  piece_classify_ms?: number;
  error?: string;
}

export interface OverlayTestSession {
  id: string;
  created_at: string;
  sample_size: number;
  accuracy: number | null;
  piece_accuracy: number | null;
  results?: OverlayTestResult[];
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}

export async function sampleOverlayBoards(
  limit: number,
  exclude?: string[]
): Promise<{ filenames: string[] }> {
  const excludeParam =
    exclude && exclude.length > 0 ? `&exclude=${exclude.join(",")}` : "";
  const res = await fetch(
    `/api/models/overlay-test/sample?limit=${limit}${excludeParam}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function inspectOverlayBoard(
  filename: string
): Promise<OverlayTestResult> {
  const res = await fetch("/api/models/overlay-test/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createOverlayTestSession(body: {
  results: OverlayTestResult[];
  accuracy?: number | null;
  sample_size?: number;
  piece_accuracy?: number | null;
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}): Promise<{ session_id: string }> {
  const res = await fetch("/api/models/overlay-test/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getOverlayTestSession(
  sessionId: string
): Promise<OverlayTestSession> {
  const res = await fetch(`/api/models/overlay-test/sessions/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listOverlayTestSessions(
  limit = 20
): Promise<{ sessions: OverlayTestSession[] }> {
  const res = await fetch(
    `/api/models/overlay-test/sessions?limit=${limit}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateOverlaySessionPins(
  sessionId: string,
  pinState: Record<string, boolean>
): Promise<{ pin_state: Record<string, boolean> }> {
  const res = await fetch(
    `/api/models/overlay-test/sessions/${sessionId}/pins`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin_state: pinState }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function overlayBoardImageUrl(filename: string): string {
  return `/api/models/overlay-test/board-image/${encodeURIComponent(filename)}`;
}

export interface OverlayValidationResult {
  video_id: string;
  channel: string;
  timestamp: number;
  found: boolean;
  score: number;
  bbox: [number, number, number, number] | null;
  time_ms: number;
  resolution: string;
}

export interface OverlayValidationResponse {
  total_frames: number;
  detected: number;
  detection_rate: number;
  avg_time_ms: number;
  p95_time_ms: number;
  per_channel: {
    channel: string;
    total: number;
    found: number;
    detection_rate: number;
    avg_time_ms: number;
  }[];
  num_channels: number;
  num_videos: number;
  elapsed_ms: number;
  results: OverlayValidationResult[];
  error?: string;
}

export async function validateOverlayDetection(
  limit = 100
): Promise<OverlayValidationResponse> {
  const res = await fetch(
    `/api/models/overlay-test/validate-real?limit=${limit}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function extractRealOverlaySamples(
  limit = 200
): Promise<{ processed: number; saved: number; skipped: number }> {
  const res = await fetch(
    `/api/models/overlay-test/extract-real-samples?limit=${limit}`,
    { method: "POST" }
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

// ── Synthetic Generation Control ─────────────────────────────

export async function startGeneration(params: {
  num_clips?: number;
  output_dir?: string;
  image_size?: number;
  clip_length?: number;
  frames_per_move?: number;
  seed?: number;
  quality?: string;
}): Promise<GenerationStatus> {
  const res = await fetch("/api/synthetic/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getGenerationStatus(): Promise<GenerationStatus> {
  const res = await fetch("/api/synthetic/generate/status");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stopGeneration(): Promise<GenerationStatus> {
  const res = await fetch("/api/synthetic/generate/stop", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Segmentation Evaluation ──────────────────────────────────

export interface SegmentEvalSessionSummary {
  id: string;
  created_at: string;
  sample_size: number;
  segment_consistency: number | null;
  gap_consistency: number | null;
  piece_readability: number | null;
  false_negative_rate: number | null;
  coverage_ratio: number | null;
}

export async function sampleSegmentationVideos(
  limit: number,
  exclude?: string[]
): Promise<{ video_ids: string[] }> {
  const excludeParam =
    exclude && exclude.length > 0 ? `&exclude=${exclude.join(",")}` : "";
  const res = await fetch(
    `/api/models/segmentation-eval/sample?limit=${limit}${excludeParam}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function normalizeSegmentEvalResult(raw: any): any {
  const metrics = raw.metrics ?? {};
  const segCons = metrics.segment_consistency ?? 0;
  const gapCons = metrics.gap_consistency ?? 1;
  return {
    video_id: raw.video_id,
    duration: raw.duration_sec ?? 0,
    segments: (raw.segments ?? []).map((s: any) => {
      const frames: any[] = s.frames ?? [];
      const framesCount = frames.length;
      const overlayCount = frames.filter((f) => f.overlay_detected).length;
      const gridCount = frames.filter((f) => f.grid_found).length;
      const piecesCount = frames.filter((f) => f.pieces_readable).length;
      return {
        start: s.start_time,
        end: s.end_time,
        overlay_bbox: s.overlay_bbox,
        thumbnail_b64: s.thumbnail_b64,
        frames,
        validation: {
          frames_sampled: framesCount,
          overlay_detected: overlayCount,
          grid_found: gridCount,
          pieces_readable: piecesCount,
        },
        segment_consistency: framesCount > 0 ? overlayCount / framesCount : 0,
        piece_readability: framesCount > 0 ? piecesCount / framesCount : 0,
      };
    }),
    gaps: (raw.gaps ?? []).map((g: any) => ({
      start: g.start_time,
      end: g.end_time,
      has_overlay: g.has_overlay ?? false,
      frames: g.frames ?? [],
    })),
    // Core metrics
    segment_consistency: segCons,
    fast_check_consistency: metrics.fast_check_consistency ?? segCons,
    gap_consistency: gapCons,
    piece_readability: metrics.piece_readability ?? 0,
    false_negative_count: metrics.false_negative_count ?? 0,
    coverage_ratio: metrics.coverage_ratio ?? 0,
    // Balanced accuracy = (segment_consistency + gap_consistency) / 2
    balanced_accuracy: (segCons + gapCons) / 2,
    // Sub-step breakdown for diagnosing piece readability failures
    overlay_miss_count: metrics.overlay_miss_count ?? 0,
    fast_check_miss_count: metrics.fast_check_miss_count ?? 0,
    grid_miss_count: metrics.grid_miss_count ?? 0,
    fen_miss_count: metrics.fen_miss_count ?? 0,
    // Raw timing
    elapsed_ms: raw.elapsed_ms ?? 0,
  };
}

export async function inspectSegmentation(
  videoId: string
): Promise<any> {
  const res = await fetch("/api/models/segmentation-eval/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return normalizeSegmentEvalResult(await res.json());
}

export async function saveSegmentationEval(body: {
  segment_consistency: number;
  gap_consistency: number;
  piece_readability: number;
  false_negative_rate: number;
  coverage_ratio: number;
  sample_size: number;
  notes?: string | null;
}): Promise<{ id: number; evaluated_at: string }> {
  const res = await fetch("/api/models/segmentation-eval/save-eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createSegmentationEvalSession(body: {
  results: any[];
  segment_consistency?: number | null;
  gap_consistency?: number | null;
  piece_readability?: number | null;
  false_negative_rate?: number | null;
  coverage_ratio?: number | null;
  sample_size?: number;
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}): Promise<{ session_id: string }> {
  const res = await fetch("/api/models/segmentation-eval/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSegmentationEvalSession(
  sessionId: string
): Promise<any> {
  const res = await fetch(
    `/api/models/segmentation-eval/sessions/${sessionId}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listSegmentationEvalSessions(
  limit = 20
): Promise<{ sessions: SegmentEvalSessionSummary[] }> {
  const res = await fetch(
    `/api/models/segmentation-eval/sessions?limit=${limit}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateSegmentationEvalPins(
  sessionId: string,
  pinState: Record<string, boolean>
): Promise<{ pin_state: Record<string, boolean> }> {
  const res = await fetch(
    `/api/models/segmentation-eval/sessions/${sessionId}/pins`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin_state: pinState }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Calibration Evaluation ──────────────────────────────────

export interface CalibrationEvalSessionSummary {
  id: string;
  created_at: string;
  sample_size: number;
  overlay_iou_avg: number | null;
  theme_accuracy: number | null;
  orientation_accuracy: number | null;
  grid_success_rate: number | null;
  fen_validity_rate: number | null;
}

export async function sampleCalibrationClips(
  limit: number,
  exclude?: number[]
): Promise<{ clips: any[] }> {
  const excludeParam =
    exclude && exclude.length > 0
      ? `&exclude=${exclude.join(",")}`
      : "";
  const res = await fetch(
    `/api/models/calibration-eval/sample?limit=${limit}${excludeParam}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function normalizeCalibrationResult(raw: any): any {
  const metrics = raw.metrics ?? {};
  const validation = raw.validation ?? {};
  const frames = (validation.frames ?? []).map((f: any) => ({
    ...f,
    // API returns grid_found; component reads grid_success
    grid_success: f.grid_found ?? f.grid_success,
  }));
  return {
    ...raw,
    // Flatten nested metrics
    overlay_iou: metrics.overlay_iou ?? null,
    grid_success_rate: metrics.grid_success_rate ?? null,
    fen_validity_rate: metrics.fen_validity_rate ?? null,
    theme_accuracy: metrics.theme_accuracy ?? null,
    orientation_accuracy: metrics.orientation_accuracy ?? null,
    camera_iou: metrics.camera_iou ?? validation.camera_iou ?? null,
    // Flatten validation
    frames,
    fresh_camera_bbox: validation.fresh_camera_bbox ?? null,
  };
}

export async function inspectCalibration(
  clipId: number
): Promise<any> {
  const res = await fetch("/api/models/calibration-eval/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip_id: clipId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return normalizeCalibrationResult(await res.json());
}

export async function saveCalibrationEval(body: {
  overlay_iou: number;
  grid_success_rate: number;
  fen_validity_rate: number;
  theme_accuracy: number;
  orientation_accuracy: number;
  camera_iou: number;
  sample_size: number;
  notes?: string | null;
}): Promise<{ id: number; evaluated_at: string }> {
  const res = await fetch("/api/models/calibration-eval/save-eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createCalibrationEvalSession(body: {
  results: any[];
  overlay_iou_avg?: number | null;
  theme_accuracy?: number | null;
  orientation_accuracy?: number | null;
  grid_success_rate?: number | null;
  fen_validity_rate?: number | null;
  sample_size?: number;
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}): Promise<{ session_id: string }> {
  const res = await fetch("/api/models/calibration-eval/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getCalibrationEvalSession(
  sessionId: string
): Promise<any> {
  const res = await fetch(
    `/api/models/calibration-eval/sessions/${sessionId}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listCalibrationEvalSessions(
  limit = 20
): Promise<{ sessions: CalibrationEvalSessionSummary[] }> {
  const res = await fetch(
    `/api/models/calibration-eval/sessions?limit=${limit}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateCalibrationEvalPins(
  sessionId: string,
  pinState: Record<string, boolean>
): Promise<{ pin_state: Record<string, boolean> }> {
  const res = await fetch(
    `/api/models/calibration-eval/sessions/${sessionId}/pins`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin_state: pinState }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Auto-Calibration ────────────────────────────────────────

export async function inspectAutoCalibration(
  videoId: string
): Promise<any> {
  const res = await fetch("/api/models/auto-calibration/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
