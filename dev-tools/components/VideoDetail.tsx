"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BboxDrawer, type Bbox } from "@/components/BboxDrawer";
import { ChessBoard } from "@/components/ChessBoard";
import {
  listCalibrations,
  saveCalibration,
  testOverlayUrl,
  getVideoAnnotations,
  saveVideoAnnotations,
} from "@/lib/api";
import type {
  CrawlVideo,
  CrawlChannel,
  CalibrationEntry,
  OverlayTestResponse,
  GameSegment,
} from "@/lib/types";

interface VideoDetailProps {
  video: CrawlVideo;
  channels: CrawlChannel[];
}

function youtubeThumb(videoId: string, index: number): string {
  return `https://img.youtube.com/vi/${videoId}/${index}.jpg`;
}

export default function VideoDetail({ video, channels }: VideoDetailProps) {
  return (
    <div className="border-t bg-muted/10">
      <Tabs defaultValue="calibrate" className="w-full">
        <TabsList className="mx-3 mt-2">
          <TabsTrigger value="calibrate">Calibrate</TabsTrigger>
          <TabsTrigger value="overlay">Overlay Test</TabsTrigger>
          <TabsTrigger value="annotate">Annotate</TabsTrigger>
        </TabsList>

        <TabsContent value="calibrate" className="px-3 pb-3">
          <CalibrateTab video={video} channels={channels} />
        </TabsContent>

        <TabsContent value="overlay" className="px-3 pb-3">
          <OverlayTab video={video} />
        </TabsContent>

        <TabsContent value="annotate" className="px-3 pb-3">
          <AnnotateTab video={video} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ── Calibrate Tab ──────────────────────────────────────────

function CalibrateTab({
  video,
  channels,
}: {
  video: CrawlVideo;
  channels: CrawlChannel[];
}) {
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [overlayBbox, setOverlayBbox] = useState<Bbox | null>(null);
  const [cameraBbox, setCameraBbox] = useState<Bbox | null>(null);
  const [drawingMode, setDrawingMode] = useState<"overlay" | "camera">(
    "overlay"
  );
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [boardTheme, setBoardTheme] = useState("lichess_default");
  const [saving, setSaving] = useState(false);
  const [existingCal, setExistingCal] = useState<CalibrationEntry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const channelHandle = video.channel_handle;

  // Load existing calibration for this channel
  useEffect(() => {
    if (!channelHandle) return;
    listCalibrations()
      .then((cals) => {
        const existing = cals.find(
          (c) => c.channel_handle === channelHandle
        );
        if (existing) {
          setExistingCal(existing);
          setOverlayBbox({
            x: existing.overlay[0],
            y: existing.overlay[1],
            w: existing.overlay[2],
            h: existing.overlay[3],
          });
          setCameraBbox({
            x: existing.camera[0],
            y: existing.camera[1],
            w: existing.camera[2],
            h: existing.camera[3],
          });
          setBoardFlipped(existing.board_flipped);
          setBoardTheme(existing.board_theme);
        }
      })
      .catch(() => {});
  }, [channelHandle]);

  const handleBboxChange = (bbox: Bbox | null) => {
    if (drawingMode === "overlay") {
      setOverlayBbox(bbox);
    } else {
      setCameraBbox(bbox);
    }
  };

  const handleSave = async () => {
    if (!channelHandle || !overlayBbox || !cameraBbox) return;
    setSaving(true);
    setError(null);
    try {
      await saveCalibration(channelHandle, {
        overlay: [overlayBbox.x, overlayBbox.y, overlayBbox.w, overlayBbox.h],
        camera: [cameraBbox.x, cameraBbox.y, cameraBbox.w, cameraBbox.h],
        ref_resolution: [1920, 1080],
        board_flipped: boardFlipped,
        board_theme: boardTheme,
      });
      setSuccess("Calibration saved");
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3 pt-2">
      {!channelHandle && (
        <p className="text-sm text-muted-foreground">
          No channel handle available for this video.
        </p>
      )}

      {existingCal && (
        <div className="text-xs text-muted-foreground">
          Existing calibration for <strong>{channelHandle}</strong> — editing
          will apply to all videos from this channel.
        </div>
      )}

      {/* Frame selector */}
      <div className="flex gap-2">
        {[0, 1, 2, 3].map((i) => (
          <button
            key={i}
            onClick={() => setSelectedFrame(i)}
            className={`relative rounded overflow-hidden border-2 transition-colors ${
              selectedFrame === i
                ? "border-primary"
                : "border-transparent hover:border-muted-foreground/30"
            }`}
          >
            <img
              src={youtubeThumb(video.video_id, i)}
              alt={`Frame ${i}`}
              className="w-24 aspect-video object-cover"
            />
            {selectedFrame === i && (
              <div className="absolute inset-0 bg-primary/10" />
            )}
          </button>
        ))}
      </div>

      {/* Drawing controls */}
      <div className="flex items-center gap-2 flex-wrap">
        <Button
          variant={drawingMode === "overlay" ? "default" : "outline"}
          size="sm"
          onClick={() => setDrawingMode("overlay")}
        >
          Draw Overlay (green)
        </Button>
        <Button
          variant={drawingMode === "camera" ? "default" : "outline"}
          size="sm"
          onClick={() => setDrawingMode("camera")}
        >
          Draw Camera (blue)
        </Button>
        {overlayBbox && <Badge variant="outline">Overlay: set</Badge>}
        {cameraBbox && <Badge variant="outline">Camera: set</Badge>}
      </div>

      {/* Bbox drawer */}
      <BboxDrawer
        imageSrc={youtubeThumb(video.video_id, selectedFrame)}
        onBboxChange={handleBboxChange}
        existingBbox={drawingMode === "overlay" ? overlayBbox : cameraBbox}
        secondBbox={drawingMode === "overlay" ? cameraBbox : overlayBbox}
      />

      {/* Theme / flipped */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={boardFlipped}
            onChange={(e) => setBoardFlipped(e.target.checked)}
            className="rounded"
          />
          Board flipped
        </label>
        <div className="flex items-center gap-2">
          <label className="text-sm text-muted-foreground">Theme:</label>
          <select
            value={boardTheme}
            onChange={(e) => setBoardTheme(e.target.value)}
            className="h-8 rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="lichess_default">Lichess Default</option>
            <option value="chess_com">Chess.com</option>
          </select>
        </div>
      </div>

      {/* Save */}
      <div className="flex items-center gap-2">
        <Button
          onClick={handleSave}
          disabled={saving || !channelHandle || !overlayBbox || !cameraBbox}
          size="sm"
        >
          {saving ? "Saving..." : "Save Calibration"}
        </Button>
        {error && <span className="text-xs text-destructive">{error}</span>}
        {success && (
          <span className="text-xs text-green-600">{success}</span>
        )}
      </div>
    </div>
  );
}

// ── Overlay Test Tab ──────────────────────────────────────

function OverlayTab({ video }: { video: CrawlVideo }) {
  const [results, setResults] = useState<Map<number, OverlayTestResponse>>(
    new Map()
  );
  const [testing, setTesting] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const runTest = async (frameIndex: number) => {
    setTesting((prev) => new Set(prev).add(frameIndex));
    setError(null);
    try {
      const url = youtubeThumb(video.video_id, frameIndex);
      const result = await testOverlayUrl(url);
      setResults((prev) => new Map(prev).set(frameIndex, result));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Test failed");
    } finally {
      setTesting((prev) => {
        const next = new Set(prev);
        next.delete(frameIndex);
        return next;
      });
    }
  };

  const runAll = async () => {
    for (const i of [0, 1, 2, 3]) {
      await runTest(i);
    }
  };

  return (
    <div className="space-y-3 pt-2">
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={runAll} disabled={testing.size > 0}>
          {testing.size > 0 ? "Testing..." : "Test All Frames"}
        </Button>
        {error && <span className="text-xs text-destructive">{error}</span>}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {[0, 1, 2, 3].map((i) => {
          const result = results.get(i);
          return (
            <div key={i} className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium">Frame {i}</span>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs"
                  onClick={() => runTest(i)}
                  disabled={testing.has(i)}
                >
                  {testing.has(i) ? (
                    <span className="h-3 w-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    "Test"
                  )}
                </Button>
                {result && (
                  <Badge
                    variant={result.detected ? "default" : "destructive"}
                    className="text-[10px]"
                  >
                    {result.detected
                      ? `detected (${result.detection_score?.toFixed(2)})`
                      : "not detected"}
                  </Badge>
                )}
              </div>

              {result?.annotated_image_b64 ? (
                <img
                  src={`data:image/png;base64,${result.annotated_image_b64}`}
                  alt={`Annotated frame ${i}`}
                  className="w-full rounded border"
                />
              ) : (
                <img
                  src={youtubeThumb(video.video_id, i)}
                  alt={`Frame ${i}`}
                  className="w-full rounded border opacity-50"
                  loading="lazy"
                />
              )}

              {result?.fen && (
                <div className="flex items-center gap-2">
                  <ChessBoard fen={result.fen} size={120} />
                  <code className="text-[10px] bg-muted px-1 py-0.5 rounded break-all flex-1">
                    {result.fen}
                  </code>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Annotate Tab ──────────────────────────────────────────

function AnnotateTab({ video }: { video: CrawlVideo }) {
  const [games, setGames] = useState<GameSegment[]>([]);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Load existing annotations
  useEffect(() => {
    getVideoAnnotations(video.video_id)
      .then((data) => {
        if (data?.annotations) {
          setGames(data.annotations.games ?? []);
          setNotes(data.annotations.notes ?? "");
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [video.video_id]);

  const addGame = () => {
    setGames((prev) => [
      ...prev,
      { start_time: 0, end_time: 0, has_overlay: true, notes: "" },
    ]);
  };

  const removeGame = (index: number) => {
    setGames((prev) => prev.filter((_, i) => i !== index));
  };

  const updateGame = (index: number, field: keyof GameSegment, value: unknown) => {
    setGames((prev) =>
      prev.map((g, i) => (i === index ? { ...g, [field]: value } : g))
    );
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveVideoAnnotations(video.video_id, {
        games: games.length > 0 ? games : null,
        notes: notes.trim() || null,
      });
      setSuccess("Annotations saved");
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="text-sm text-muted-foreground py-4">
        Loading annotations...
      </div>
    );
  }

  return (
    <div className="space-y-3 pt-2">
      {/* Game segments */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-medium">Game Segments</h4>
          <Button size="sm" variant="outline" className="h-6 text-xs" onClick={addGame}>
            + Add Game
          </Button>
        </div>

        {games.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No game segments defined. Add one to mark where games start and end in this video.
          </p>
        )}

        {games.map((game, i) => (
          <div
            key={i}
            className="flex items-center gap-2 p-2 rounded border bg-muted/20 text-sm"
          >
            <span className="text-xs font-medium text-muted-foreground w-16">
              Game {i + 1}
            </span>
            <label className="flex items-center gap-1 text-xs">
              Start:
              <input
                type="number"
                value={game.start_time}
                onChange={(e) =>
                  updateGame(i, "start_time", parseFloat(e.target.value) || 0)
                }
                className="h-7 w-20 rounded border bg-background px-2 text-xs"
                step="0.1"
                min="0"
              />
              s
            </label>
            <label className="flex items-center gap-1 text-xs">
              End:
              <input
                type="number"
                value={game.end_time}
                onChange={(e) =>
                  updateGame(i, "end_time", parseFloat(e.target.value) || 0)
                }
                className="h-7 w-20 rounded border bg-background px-2 text-xs"
                step="0.1"
                min="0"
              />
              s
            </label>
            <label className="flex items-center gap-1 text-xs">
              <input
                type="checkbox"
                checked={game.has_overlay}
                onChange={(e) =>
                  updateGame(i, "has_overlay", e.target.checked)
                }
                className="rounded"
              />
              Has overlay
            </label>
            <input
              type="text"
              value={game.notes ?? ""}
              onChange={(e) => updateGame(i, "notes", e.target.value)}
              placeholder="Notes..."
              className="h-7 flex-1 rounded border bg-background px-2 text-xs"
            />
            <button
              onClick={() => removeGame(i)}
              className="text-xs text-destructive hover:text-destructive/80"
            >
              &times;
            </button>
          </div>
        ))}
      </div>

      {/* Notes */}
      <div>
        <label className="text-sm font-medium block mb-1">Notes</label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="w-full h-20 rounded-md border bg-background px-3 py-2 text-sm resize-y"
          placeholder="Any additional notes about this video..."
        />
      </div>

      {/* Save */}
      <div className="flex items-center gap-2">
        <Button onClick={handleSave} disabled={saving} size="sm">
          {saving ? "Saving..." : "Save Annotations"}
        </Button>
        {error && <span className="text-xs text-destructive">{error}</span>}
        {success && (
          <span className="text-xs text-green-600">{success}</span>
        )}
      </div>
    </div>
  );
}
