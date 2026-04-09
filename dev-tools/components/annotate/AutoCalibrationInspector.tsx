"use client";

import { useState } from "react";
import { toast } from "sonner";

interface OverlayDetection {
  found: boolean;
  bbox: number[] | null;
  seed_bbox: number[] | null;
  score: number;
}

interface FrameResult {
  timestamp_sec: number;
  image_base64: string;
  overlay_detection: OverlayDetection;
  used_for_proposal?: boolean;
}

interface Proposal {
  overlay: number[];
  camera: number[];
  theme: string;
  theme_confidence: number;
  board_flipped: boolean;
  orientation_confidence: number;
}

interface CalibrationResult {
  video_id: string;
  source: string;
  frames: FrameResult[];
  proposal: Proposal | null;
  proposal_frame_base64: string | null;
  saved_calibration: Record<string, any> | null;
  error?: string;
}

function BboxLabel({ label, bbox }: { label: string; bbox: number[] }) {
  return (
    <span className="text-xs text-muted-foreground">
      {label}: ({bbox.join(", ")})
    </span>
  );
}

export default function AutoCalibrationInspector() {
  const [videoId, setVideoId] = useState("");
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function inspect() {
    if (!videoId.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/models/auto-calibration/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId.trim() }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.error) toast.error(data.error);
      setResult(data);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Calibration analysis failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          type="text"
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && inspect()}
          placeholder="Video ID (must be downloaded locally)"
          className="flex-1 px-3 py-1.5 border rounded text-sm"
        />
        <button
          onClick={inspect}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Inspect"}
        </button>
      </div>


      {result && !result.error && (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Source: <b>{result.source}</b>
          </p>

          {/* Frames with overlay detection */}
          <div>
            <h3 className="text-sm font-medium mb-2">
              Overlay Detection at Each Timestamp
            </h3>
            <p className="text-xs text-muted-foreground mb-2">
              Green = expanded bbox, Yellow = seed bbox (initial detection before expansion)
            </p>
            <div className="grid grid-cols-3 gap-3">
              {result.frames.map((frame) => (
                <div key={frame.timestamp_sec} className="space-y-1">
                  <div className="relative">
                    <img
                      src={`data:image/jpeg;base64,${frame.image_base64}`}
                      alt={`${frame.timestamp_sec}s`}
                      className={`w-full rounded border ${frame.used_for_proposal ? "ring-2 ring-green-500" : ""}`}
                    />
                    {frame.used_for_proposal && (
                      <span className="absolute top-1 right-1 bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">
                        Best
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <b>{frame.timestamp_sec}s</b>
                    {frame.overlay_detection.found ? (
                      <span className="text-green-600 ml-2">
                        Detected (score={frame.overlay_detection.score})
                      </span>
                    ) : (
                      <span className="text-red-600 ml-2">Not found</span>
                    )}
                  </div>
                  {frame.overlay_detection.bbox && (
                    <BboxLabel label="Expanded" bbox={frame.overlay_detection.bbox} />
                  )}
                  {frame.overlay_detection.seed_bbox && (
                    <div>
                      <BboxLabel label="Seed" bbox={frame.overlay_detection.seed_bbox} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Proposed calibration visualization */}
          {result.proposal_frame_base64 && (
            <div>
              <h3 className="text-sm font-medium mb-2">
                Proposed Calibration (visual)
              </h3>
              <p className="text-xs text-muted-foreground mb-1">
                Green = Overlay region, Red = OTB board region
              </p>
              <img
                src={`data:image/jpeg;base64,${result.proposal_frame_base64}`}
                alt="Proposal visualization"
                className="w-full max-w-xl rounded border"
              />
            </div>
          )}

          {/* Proposal vs Saved */}
          <div className="grid grid-cols-2 gap-4">
            {result.proposal && (
              <div className="border rounded p-3 space-y-2">
                <h3 className="text-sm font-medium">Proposed Calibration</h3>
                <div className="text-xs space-y-1">
                  <p>Overlay: ({result.proposal.overlay.join(", ")})</p>
                  <p>OTB Board: ({result.proposal.camera.join(", ")})</p>
                  <p>
                    Theme: <b>{result.proposal.theme}</b>{" "}
                    <span className="text-muted-foreground">
                      ({(result.proposal.theme_confidence * 100).toFixed(0)}%)
                    </span>
                  </p>
                  <p>
                    Flipped: <b>{result.proposal.board_flipped ? "Yes" : "No"}</b>{" "}
                    <span className="text-muted-foreground">
                      ({(result.proposal.orientation_confidence * 100).toFixed(0)}%)
                    </span>
                  </p>
                </div>
              </div>
            )}
            {result.saved_calibration ? (
              <div className="border rounded p-3 space-y-2">
                <h3 className="text-sm font-medium">Saved Calibration</h3>
                <div className="text-xs space-y-1">
                  <p>Overlay: ({result.saved_calibration.overlay.join(", ")})</p>
                  <p>OTB Board: ({result.saved_calibration.camera.join(", ")})</p>
                  <p>Theme: <b>{result.saved_calibration.board_theme}</b></p>
                  <p>Flipped: <b>{result.saved_calibration.board_flipped ? "Yes" : "No"}</b></p>
                </div>
              </div>
            ) : (
              <div className="border rounded p-3 border-dashed">
                <p className="text-sm text-muted-foreground">No saved calibration</p>
              </div>
            )}
          </div>

        </div>
      )}
    </div>
  );
}
