// Overlay tester
export interface OverlayTestResponse {
  detected: boolean;
  bbox: [number, number, number, number] | null;
  detection_score: number | null;
  fen: string | null;
  piece_count: number | null;
  board_ascii: string | null;
  annotated_image_b64: string;
  image_width: number;
  image_height: number;
}

export interface SquareAnalysis {
  square: string;
  piece: string | null;
  variance: number;
}

export interface ReaderTestResponse {
  fen: string | null;
  board_ascii: string | null;
  piece_count: number | null;
  squares: SquareAnalysis[];
  annotated_crop_b64: string;
}

// Calibration
export interface CalibrationEntry {
  channel_handle: string;
  overlay: [number, number, number, number];
  camera: [number, number, number, number];
  ref_resolution: [number, number];
  board_flipped: boolean;
  board_theme: string;
}

// Clip inspector
export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
}

export interface DetectedMove {
  frame_index: number;
  uci: string;
  san: string | null;
  detect_value: number | null;
}

export interface ClipInspectResponse {
  file_size_mb: number;
  tensors: TensorInfo[];
  num_frames: number;
  pixel_range: [number, number];
  moves: DetectedMove[];
  total_moves: number;
  no_move_frames: number;
  unknown_frames: number;
  replay_valid: boolean;
  replay_error: string | null;
  final_fen: string | null;
  avg_legal_moves: number | null;
  metadata?: Record<string, string | number | boolean | null>;
}

// Video annotator
export interface VideoSession {
  session_id: string;
  fps: number;
  total_frames: number;
  duration_seconds: number;
  width: number;
  height: number;
  has_calibration: boolean;
  calibration: CalibrationEntry | null;
}

export interface FrameOverlayResponse {
  frame_index: number;
  timestamp_seconds: number;
  fen: string | null;
  board_ascii: string | null;
  read_method: string | null;
  overlay_crop_b64: string;
  camera_crop_b64: string;
}

export interface VideoDetectedMove {
  move_index: number;
  move_uci: string;
  move_san: string;
  frame_idx: number;
  timestamp_seconds: number;
  fen_before: string;
  fen_after: string;
}

export interface GameSegmentResponse {
  game_index: number;
  num_moves: number;
  pgn_moves: string;
  moves: VideoDetectedMove[];
  start_frame: number;
  end_frame: number;
}

export interface VideoMoveDetectionResponse {
  num_frames_sampled: number;
  num_readable: number;
  reader_backend: string;
  segments: GameSegmentResponse[];
}

export interface VideoMoveDetectionJobStatus {
  job_id: string;
  status: "running" | "done" | "failed";
  sample_fps: number;
  clip_id: number | null;
  reader_backend: string;
  error: string | null;
  result: VideoMoveDetectionResponse | null;
  total_samples: number;
  completed_samples: number;
  num_readable: number;
  current_frame_idx: number | null;
}

// Crawl
export interface CrawlChannel {
  channel_id: string;
  channel_handle: string | null;
  channel_name: string;
  tier: number;
  uploads_playlist_id: string | null;
  last_crawled_at: string | null;
  enabled: boolean;
  notes: string | null;
  video_count: number;
}

export interface CrawlChannelDetail extends CrawlChannel {
  status_counts: Record<string, number>;
}

export interface CrawlVideo {
  video_id: string;
  channel_id: string;
  channel_handle: string | null;
  title: string;
  description: string | null;
  published_at: string | null;
  screening_status: string | null;
  screening_confidence: number | null;
  layout_type?: string | null;
  screened_by?: string | null;
  annotations?: VideoAnnotations | null;
}

export interface GameSegment {
  start_time: number;
  end_time: number;
  has_overlay: boolean;
  notes?: string;
}

export interface VideoClip {
  id: number;
  video_id: string;
  clip_index: number;
  label: string | null;
  start_time: number;
  end_time: number | null;
  overlay_bbox: [number, number, number, number];
  camera_bbox: [number, number, number, number];
  ref_resolution: [number, number];
  board_flipped: boolean;
  board_theme: string;
  is_gap: boolean;
}

export interface VideoAnnotations {
  games: GameSegment[] | null;
  notes: string | null;
}

export interface CrawlVideosResponse {
  videos: CrawlVideo[];
  total: number;
  auto_rejected_count?: number;
  auto_rejected_ids?: string[];
}

export interface QuotaStatus {
  daily_usage: number;
  remaining: number;
  daily_limit: number;
}

// Video frame inspection
export interface InspectFrame {
  label: string;
  overlay_found: boolean;
  overlay_score: number;
  overlay_bbox: number[] | null;
  otb_found: boolean;
  otb_confidence: number;
  has_person: boolean;
  person_count: number;
  image_base64: string;
}

export interface InspectResult {
  video_id: string;
  title: string;
  has_overlay: boolean;
  has_otb: boolean;
  has_person: boolean;
  overlay_score: number;
  otb_confidence: number;
  approved: boolean;
  status: string;
  frames: InspectFrame[];
}

export interface InspectJobResult {
  video_id: string;
  approved: boolean;
  has_overlay?: boolean;
  has_otb?: boolean;
  overlay_score?: number;
  otb_confidence?: number;
  status?: string;
  error?: string;
}

export interface InspectJobStatus {
  job_id: string;
  status: "running" | "done";
  total: number;
  completed: number;
  approved: number;
  rejected: number;
  failed: number;
  current_video: string | null;
  results: InspectJobResult[];
}

// Download status
export interface DownloadStatus {
  downloaded: boolean;
  path: string | null;
  file_size_mb: number | null;
  duration_seconds: number | null;
}

// Download result
export interface DownloadResult {
  status: "downloaded" | "already_downloaded";
  path: string;
  file_size_mb: number;
}

// Generate clips
export interface GeneratedClip {
  filepath: string;
  num_frames: number;
  num_moves: number;
  game_index: number;
  pgn_moves: string;
}

export interface GenerateClipsResponse {
  clips: GeneratedClip[];
  total_clips: number;
}

// AI Screening result (lightweight, for screening page)
export interface AiScreenResult {
  video_id: string;
  predicted_class?: string;
  confidence?: number;
  auto_decided?: boolean;
  vertical: boolean;
  max_ovl_score?: number;
  max_otb_score?: number;
  model_version?: string | null;
  error?: string;
}

// Synthetic data monitoring
export interface SyntheticClipFile {
  filename: string;
  size_mb: number;
  modified: string;
}

export interface SyntheticScanResponse {
  directory: string;
  exists: boolean;
  expected_clips: number | null;
  clip_count: number;
  total_size_mb: number;
  clips: SyntheticClipFile[];
}

export interface SyntheticStatsResponse {
  clip_count: number;
  total_frames: number;
  avg_frames_per_clip: number;
  total_moves: number;
  avg_moves_per_clip: number;
  moves_per_clip_distribution: number[];
  avg_file_size_mb: number;
  total_size_mb: number;
  avg_legal_moves: number | null;
  image_size: [number, number] | null;
  clip_length: number | null;
}

export interface GenerationStatus {
  status: "idle" | "running" | "done" | "failed" | "stopped" | "no_job_running";
  job_id?: string;
  num_clips?: number;
  completed?: number;
  output_dir?: string;
  error?: string | null;
}

export interface RealVideoInventoryRow {
  video_id: string;
  video_path: string;
  file_size_mb: number;
  modified_ts: number;
  title: string | null;
  channel_handle: string | null;
  published_at: string | null;
  screening_status: string | null;
  layout_type: string | null;
  existing_clip_count: number;
  db_clip_count: number;
  has_channel_calibration: boolean;
  ready: boolean;
  blocker: string | null;
}

export interface RealDataOverview {
  clips_dir: string;
  clip_stats: SyntheticStatsResponse;
  source_video_count: number;
  local_video_count: number;
  ready_video_count: number;
  processed_video_count: number;
  blocked_video_count: number;
  videos: RealVideoInventoryRow[];
}

export interface RealVideoProcessingResult {
  video_id: string;
  title: string | null;
  status: "generated" | "no_clips" | "failed";
  generated_clip_count: number;
  error: string | null;
}

export interface RealVideoProcessingStatus {
  status: "idle" | "running" | "done" | "failed" | "stopped" | "no_job_running";
  job_id?: string;
  requested_limit?: number;
  completed_videos?: number;
  total_videos?: number;
  generated_clips?: number;
  current_video_id?: string | null;
  current_video_title?: string | null;
  clips_dir?: string;
  results?: RealVideoProcessingResult[];
  error?: string | null;
}

// Auto-segment
export interface SegmentResult {
  start_time: number;
  end_time: number;
  overlay_bbox: [number, number, number, number] | null;
  score: number;
  sample_count: number;
  clip_id: number;
}

export interface SegmentGap {
  start_time: number;
  end_time: number;
}

export interface AutoSegmentResponse {
  segments: SegmentResult[];
  gaps: SegmentGap[];
  video_resolution: [number, number];
  total_frames_sampled: number;
  processing_time_sec: number;
  error?: string;
  existing_clips?: number;
}

// Segmentation evaluation
export interface SegmentFrameValidation {
  time: number;
  overlay_detected: boolean;
  grid_found?: boolean;
  pieces_readable?: boolean;
}

export interface SegmentEvalSegment {
  start_time: number;
  end_time: number;
  overlay_bbox: [number, number, number, number] | null;
  score: number;
  sample_count: number;
  frames: SegmentFrameValidation[];
  thumbnail_b64?: string;
}

export interface SegmentEvalGap {
  start_time: number;
  end_time: number;
  frames: SegmentFrameValidation[];
  has_overlay: boolean;
}

export interface SegmentEvalMetrics {
  segment_consistency: number;
  gap_consistency: number;
  piece_readability: number;
  false_negative_count: number;
  coverage_ratio: number;
}

export interface SegmentEvalResult {
  video_id: string;
  duration_sec: number;
  resolution: string;
  num_segments: number;
  num_gaps: number;
  segments: SegmentEvalSegment[];
  gaps: SegmentEvalGap[];
  metrics: SegmentEvalMetrics;
  elapsed_ms: number;
  error?: string;
}

export interface SegmentEvalSession {
  id: string;
  created_at: string;
  sample_size: number;
  segment_consistency: number | null;
  gap_consistency: number | null;
  piece_readability: number | null;
  false_negative_rate: number | null;
  coverage_ratio: number | null;
  results?: SegmentEvalResult[];
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}

// Calibration evaluation
export interface CalibrationFrameValidation {
  timestamp: number;
  overlay_detected: boolean;
  overlay_iou: number;
  grid_found: boolean;
  fen: string | null;
  fen_valid: boolean;
  detected_theme: string;
  theme_confidence: number;
  theme_match: boolean;
  detected_flipped: boolean;
  orientation_confidence: number;
  orientation_match: boolean;
  frame_b64?: string;
  crop_b64?: string;
}

export interface CalibrationEvalResult {
  clip_id: number;
  video_id: string;
  clip_index: number;
  start_time: number;
  end_time: number | null;
  stored_calibration: {
    overlay_bbox: number[];
    camera_bbox: number[];
    ref_resolution: number[];
    board_theme: string;
    board_flipped: boolean;
  };
  validation: {
    frames: CalibrationFrameValidation[];
    fresh_camera_bbox: number[] | null;
    camera_iou: number;
  };
  metrics: {
    overlay_iou: number;
    grid_success_rate: number;
    fen_validity_rate: number;
    theme_accuracy: number;
    orientation_accuracy: number;
    camera_iou: number;
  };
  elapsed_ms: number;
}

export interface CalibrationEvalSession {
  id: string;
  created_at: string;
  sample_size: number;
  overlay_iou_avg: number | null;
  theme_accuracy: number | null;
  orientation_accuracy: number | null;
  grid_success_rate: number | null;
  fen_validity_rate: number | null;
  results?: CalibrationEvalResult[];
  pin_state?: Record<string, boolean>;
  evaluation_id?: number | null;
}

// Auto-calibrate clip
export interface AutoCalibrateProposal {
  overlay_bbox: [number, number, number, number];
  camera_bbox: [number, number, number, number];
  board_theme: string;
  theme_confidence: number;
  board_flipped: boolean;
  orientation_confidence: number;
  ref_resolution: [number, number];
}

export interface AutoCalibrateResponse {
  clip_id: number;
  proposal: AutoCalibrateProposal | null;
  applied: boolean;
  preview_frame_b64: string | null;
  failure_reason: string | null;
  detected_overlay_bbox: [number, number, number, number] | null;
}
