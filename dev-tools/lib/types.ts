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
  segments: GameSegmentResponse[];
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
  title_score: number;
  title_is_candidate: boolean;
}

export interface CrawlVideosResponse {
  videos: CrawlVideo[];
  total: number;
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
