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
