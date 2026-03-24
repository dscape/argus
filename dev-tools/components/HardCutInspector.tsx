"use client";

import { useState } from "react";

interface Move {
  move_san: string;
  move_uci: string;
  frame_idx: number;
  timestamp_sec: number;
  confidence: number;
  fen_before: string;
  fen_after: string;
  squares_changed: number;
}

interface Segment {
  game_index: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  num_moves: number;
  pgn_moves: string;
  moves: Move[];
}

interface HardCutResult {
  video_id: string;
  total_frames_sampled: number;
  readable_fens: number;
  segments: Segment[];
  total_segments: number;
  total_moves: number;
  avg_confidence: number;
  error?: string;
}

function confidenceColor(conf: number) {
  if (conf >= 0.8) return "text-green-600";
  if (conf >= 0.5) return "text-yellow-600";
  return "text-red-600";
}

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function HardCutInspector() {
  const [videoId, setVideoId] = useState("");
  const [sampleFps, setSampleFps] = useState(2.0);
  const [result, setResult] = useState<HardCutResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedSeg, setExpandedSeg] = useState<number | null>(null);

  async function inspect() {
    if (!videoId.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/models/hard-cuts/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId.trim(), sample_fps: sampleFps }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-center">
        <input
          type="text"
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && inspect()}
          placeholder="Video ID (must be downloaded + calibrated)"
          className="flex-1 px-3 py-1.5 border rounded text-sm"
        />
        <label className="text-xs text-muted-foreground">
          FPS:
          <input
            type="number"
            value={sampleFps}
            onChange={(e) => setSampleFps(Number(e.target.value))}
            step={0.5}
            min={0.5}
            max={10}
            className="w-14 ml-1 px-1 py-1.5 border rounded text-sm"
          />
        </label>
        <button
          onClick={inspect}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Inspect"}
        </button>
      </div>

      {result?.error && (
        <div className="p-3 bg-red-50 text-red-700 rounded text-sm">{result.error}</div>
      )}

      {result && !result.error && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-4 gap-3">
            <div className="border rounded p-3 text-center">
              <p className="text-2xl font-bold">{result.total_segments}</p>
              <p className="text-xs text-muted-foreground">Segments</p>
            </div>
            <div className="border rounded p-3 text-center">
              <p className="text-2xl font-bold">{result.total_moves}</p>
              <p className="text-xs text-muted-foreground">Moves</p>
            </div>
            <div className="border rounded p-3 text-center">
              <p className={`text-2xl font-bold ${confidenceColor(result.avg_confidence)}`}>
                {(result.avg_confidence * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-muted-foreground">Avg Confidence</p>
            </div>
            <div className="border rounded p-3 text-center">
              <p className="text-2xl font-bold">
                {result.readable_fens}/{result.total_frames_sampled}
              </p>
              <p className="text-xs text-muted-foreground">Readable FENs</p>
            </div>
          </div>

          {/* Segment timeline */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Game Segments</h3>
            {result.segments.map((seg) => (
              <div key={seg.game_index} className="border rounded overflow-hidden">
                <button
                  onClick={() =>
                    setExpandedSeg(expandedSeg === seg.game_index ? null : seg.game_index)
                  }
                  className="w-full flex items-center justify-between p-3 hover:bg-muted/30 text-left"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">
                      Game {seg.game_index + 1}
                    </span>
                    <span className="text-sm">
                      {formatTime(seg.start_time)} - {formatTime(seg.end_time)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {seg.num_moves} moves
                    </span>
                  </div>
                  <span className="text-xs">{expandedSeg === seg.game_index ? "collapse" : "expand"}</span>
                </button>

                {expandedSeg === seg.game_index && (
                  <div className="border-t p-3 space-y-2">
                    <p className="text-xs font-mono text-muted-foreground break-all">
                      {seg.pgn_moves}
                    </p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b text-left text-muted-foreground">
                            <th className="p-1">#</th>
                            <th className="p-1">Move</th>
                            <th className="p-1">Time</th>
                            <th className="p-1">Confidence</th>
                            <th className="p-1">Squares</th>
                          </tr>
                        </thead>
                        <tbody>
                          {seg.moves.map((m, i) => (
                            <tr key={i} className="border-b border-muted/30">
                              <td className="p-1 text-muted-foreground">{i + 1}</td>
                              <td className="p-1 font-mono font-medium">{m.move_san}</td>
                              <td className="p-1">{formatTime(m.timestamp_sec)}</td>
                              <td className={`p-1 font-mono ${confidenceColor(m.confidence)}`}>
                                {(m.confidence * 100).toFixed(0)}%
                              </td>
                              <td className="p-1">
                                <span
                                  className={
                                    m.squares_changed > 4
                                      ? "text-red-600 font-bold"
                                      : ""
                                  }
                                >
                                  {m.squares_changed}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
