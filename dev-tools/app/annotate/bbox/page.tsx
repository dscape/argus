"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, X, Trash2 } from "lucide-react";

interface FrameEntry {
  key: string;
  video_id: string;
  label: string;
  frame_width: number;
  frame_height: number;
  annotated: boolean;
  has_overlay: boolean | null;
  bbox: number[] | null;
  is_target?: boolean;
  target_issue?: string | null;
}

interface RefineResult {
  bbox: number[];
  score: number;
  has_pattern: boolean;
  original: number[];
}

interface AutoDetectResult {
  detected: boolean;
  bbox: number[] | null;
  score: number;
  grid_score?: number;
  has_pattern?: boolean;
}

type AutoDetectStatus = "idle" | "running" | "found" | "not_found" | "error";

export default function OverlayBboxPage() {
  const [frames, setFrames] = useState<FrameEntry[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [currentBbox, setCurrentBbox] = useState<number[] | null>(null);
  const [refinedScore, setRefinedScore] = useState<number | null>(null);
  const [hasPattern, setHasPattern] = useState<boolean | null>(null);
  const [autoDetectStatus, setAutoDetectStatus] =
    useState<AutoDetectStatus>("idle");
  const [autoDetectScore, setAutoDetectScore] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(
    null
  );
  const [drawCurrent, setDrawCurrent] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [refining, setRefining] = useState(false);
  const [saving, setSaving] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const autoDetectRequestRef = useRef(0);
  const [imgLoaded, setImgLoaded] = useState(false);

  const selected = frames.find((f) => f.key === selectedKey) ?? null;

  const fetchFrames = useCallback(async (): Promise<FrameEntry[]> => {
    const res = await fetch("/api/overlay-bbox/frames");
    const data = await res.json();
    setFrames(data.frames);
    return data.frames as FrameEntry[];
  }, []);

  useEffect(() => {
    fetchFrames();
  }, [fetchFrames]);

  useEffect(() => {
    if (frames.length === 0) {
      if (selectedKey !== null) setSelectedKey(null);
      return;
    }
    if (!selectedKey || !frames.some((frame) => frame.key === selectedKey)) {
      setSelectedKey(frames[0].key);
    }
  }, [frames, selectedKey]);

  // Load image when selection changes
  useEffect(() => {
    if (!selectedKey) return;
    let cancelled = false;
    autoDetectRequestRef.current += 1;
    const requestId = autoDetectRequestRef.current;
    setImgLoaded(false);
    setCurrentBbox(null);
    setRefinedScore(null);
    setHasPattern(null);
    setAutoDetectStatus("idle");
    setAutoDetectScore(null);

    const frame = frames.find((entry) => entry.key === selectedKey);
    const [videoId, label] = selectedKey.split("/");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = async () => {
      if (cancelled) return;
      imgRef.current = img;
      setImgLoaded(true);

      // If already annotated, load the saved bbox
      if (frame?.bbox) {
        setCurrentBbox(frame.bbox);
        return;
      }
      if (frame?.annotated) return;

      setAutoDetectStatus("running");
      try {
        const res = await fetch("/api/overlay-bbox/auto-detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame_key: selectedKey }),
        });
        if (!res.ok) {
          throw new Error(`Auto-detect failed with status ${res.status}`);
        }

        const data: AutoDetectResult = await res.json();
        if (cancelled || requestId !== autoDetectRequestRef.current) return;

        if (data.detected && data.bbox) {
          setCurrentBbox(data.bbox);
          setAutoDetectStatus("found");
          setAutoDetectScore(data.score);
          setRefinedScore(data.grid_score ?? null);
          setHasPattern(data.has_pattern ?? null);
          return;
        }

        setAutoDetectStatus("not_found");
      } catch (err) {
        if (cancelled) return;
        console.error("Auto-detect failed:", err);
        setAutoDetectStatus("error");
      }
    };
    img.onerror = () => {
      if (cancelled) return;
      setImgLoaded(false);
    };
    img.src = `/api/overlay-bbox/frame-image/${videoId}/${label}`;
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [selectedKey, frames]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgLoaded) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas to image dimensions
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);

    // Draw current bbox
    if (currentBbox) {
      const [x, y, w, h] = currentBbox;
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(34, 197, 94, 0.1)";
      ctx.fillRect(x, y, w, h);
    }

    // Draw in-progress rectangle
    if (isDrawing && drawStart && drawCurrent) {
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(239, 68, 68, 0.1)";
      ctx.fillRect(x, y, w, h);
    }
  }, [imgLoaded, currentBbox, isDrawing, drawStart, drawCurrent]);

  const canvasToImage = (
    clientX: number,
    clientY: number
  ): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((clientX - rect.left) * scaleX),
      y: Math.round((clientY - rect.top) * scaleY),
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const pt = canvasToImage(e.clientX, e.clientY);
    if (!pt) return;
    autoDetectRequestRef.current += 1;
    setIsDrawing(true);
    setDrawStart(pt);
    setDrawCurrent(pt);
    setCurrentBbox(null);
    setRefinedScore(null);
    setHasPattern(null);
    setAutoDetectStatus("idle");
    setAutoDetectScore(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    const pt = canvasToImage(e.clientX, e.clientY);
    if (pt) setDrawCurrent(pt);
  };

  const handleMouseUp = async () => {
    if (!isDrawing || !drawStart || !drawCurrent || !selectedKey) {
      setIsDrawing(false);
      return;
    }
    setIsDrawing(false);

    const x = Math.min(drawStart.x, drawCurrent.x);
    const y = Math.min(drawStart.y, drawCurrent.y);
    const w = Math.abs(drawCurrent.x - drawStart.x);
    const h = Math.abs(drawCurrent.y - drawStart.y);

    if (w < 30 || h < 30) {
      setDrawStart(null);
      setDrawCurrent(null);
      return;
    }

    // Refine
    setRefining(true);
    try {
      const res = await fetch("/api/overlay-bbox/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame_key: selectedKey,
          rough_bbox: [x, y, w, h],
        }),
      });
      const data: RefineResult = await res.json();
      setCurrentBbox(data.bbox);
      setRefinedScore(data.score);
      setHasPattern(data.has_pattern);
    } catch (err) {
      console.error("Refine failed:", err);
      setCurrentBbox([x, y, w, h]);
    } finally {
      setRefining(false);
      setDrawStart(null);
      setDrawCurrent(null);
    }
  };

  const selectNextWorkItem = useCallback(
    (updatedFrames: FrameEntry[], previousKey: string | null) => {
      const nextUnannotated = updatedFrames.find((frame) => !frame.annotated);
      if (nextUnannotated) {
        setSelectedKey(nextUnannotated.key);
        return;
      }

      if (previousKey && updatedFrames.some((frame) => frame.key === previousKey)) {
        setSelectedKey(previousKey);
        return;
      }

      setSelectedKey(updatedFrames[0]?.key ?? null);
    },
    []
  );

  const handleSave = async () => {
    if (!selectedKey || !currentBbox) return;
    const frameKey = selectedKey;
    setSaving(true);
    try {
      await fetch("/api/overlay-bbox/annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame_key: selectedKey,
          has_overlay: true,
          bbox: currentBbox,
        }),
      });
      const updatedFrames = await fetchFrames();
      selectNextWorkItem(updatedFrames, frameKey);
    } finally {
      setSaving(false);
    }
  };

  const handleNoOverlay = async () => {
    if (!selectedKey) return;
    const frameKey = selectedKey;
    setSaving(true);
    try {
      await fetch("/api/overlay-bbox/annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame_key: selectedKey,
          has_overlay: false,
          bbox: null,
        }),
      });
      const updatedFrames = await fetchFrames();
      selectNextWorkItem(updatedFrames, frameKey);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedKey) return;
    autoDetectRequestRef.current += 1;
    await fetch(`/api/overlay-bbox/annotate/${selectedKey}`, {
      method: "DELETE",
    });
    setCurrentBbox(null);
    setRefinedScore(null);
    setHasPattern(null);
    await fetchFrames();
  };

  const handleClear = () => {
    autoDetectRequestRef.current += 1;
    setCurrentBbox(null);
    setRefinedScore(null);
    setHasPattern(null);
    setAutoDetectStatus("idle");
    setAutoDetectScore(null);
  };

  const annotatedCount = frames.filter((f) => f.annotated).length;
  const remainingTargetCount = frames.filter(
    (f) => f.is_target && !f.annotated,
  ).length;

  return (
    <div className="flex gap-4 h-[calc(100vh-180px)]">
      {/* Sidebar */}
      <div className="w-64 shrink-0 border rounded-lg overflow-y-auto">
        <div className="p-3 border-b bg-muted/50 space-y-1">
          <div className="text-sm font-medium">
            Frames ({annotatedCount}/{frames.length} annotated)
          </div>
          {remainingTargetCount > 0 && (
            <div className="text-xs text-amber-700">
              {remainingTargetCount} fixture targets remaining
            </div>
          )}
        </div>
        {frames.map((f) => (
          <button
            key={f.key}
            onClick={() => setSelectedKey(f.key)}
            className={`w-full text-left px-3 py-2 text-sm border-b flex items-center gap-2 hover:bg-muted/50 transition-colors ${
              f.key === selectedKey ? "bg-muted" : ""
            } ${f.is_target ? "border-l-2 border-l-amber-500" : ""}`}
          >
            <span className="flex-1 truncate font-mono text-xs">{f.key}</span>
            {f.is_target && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-amber-100 text-amber-700 shrink-0">
                target
              </span>
            )}
            {f.annotated &&
              (f.has_overlay ? (
                <Check className="w-4 h-4 text-green-500 shrink-0" />
              ) : (
                <X className="w-4 h-4 text-red-500 shrink-0" />
              ))}
          </button>
        ))}
      </div>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="mb-3 text-xs px-3 py-2 rounded border border-blue-200 bg-blue-50 text-blue-900">
          Training labels only. Runtime overlay localization uses the committed
          YOLO detector; this page exists to create and fix detector ground
          truth.
        </div>
        {selectedKey && imgLoaded ? (
          <>
            {/* Canvas */}
            <div className="flex-1 overflow-auto border rounded-lg bg-black/5">
              <canvas
                ref={canvasRef}
                className="max-w-full max-h-full cursor-crosshair"
                style={{ imageRendering: "auto" }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => {
                  if (isDrawing) handleMouseUp();
                }}
              />
            </div>

            {/* Controls */}
            <div className="mt-3 space-y-3">
              {selected?.is_target && selected.target_issue && (
                <div className="text-xs px-2.5 py-2 rounded border border-amber-200 bg-amber-50 text-amber-800">
                  Fixture target: {selected.target_issue}
                </div>
              )}
              <div className="flex items-center gap-3 flex-wrap">
              {currentBbox && (
                <div className="text-xs font-mono text-muted-foreground">
                  bbox: [{currentBbox.join(", ")}]
                  {refinedScore !== null && (
                    <span className="ml-2">
                      score: {refinedScore}
                      {hasPattern !== null && (
                        <span
                          className={
                            hasPattern ? "text-green-500" : "text-red-500"
                          }
                        >
                          {" "}
                          {hasPattern ? "pattern" : "no pattern"}
                        </span>
                      )}
                    </span>
                  )}
                </div>
              )}
              {autoDetectStatus === "running" && (
                <div className="text-xs text-muted-foreground">
                  Detecting overlay...
                </div>
              )}
              {autoDetectStatus === "found" && autoDetectScore !== null && (
                <div className="text-xs text-muted-foreground">
                  Auto-detected candidate (YOLO score: {autoDetectScore.toFixed(4)})
                </div>
              )}
              {autoDetectStatus === "not_found" && (
                <div className="text-xs text-amber-700">
                  Auto-detect found nothing. Draw a bbox if this frame still has an overlay.
                </div>
              )}
              {autoDetectStatus === "error" && (
                <div className="text-xs text-red-600">
                  Auto-detect failed. Draw a bbox manually.
                </div>
              )}
              {refining && (
                <div className="text-xs text-muted-foreground">Refining...</div>
              )}
              <div className="flex-1" />
              <button
                onClick={handleClear}
                className="px-3 py-1.5 text-xs border rounded hover:bg-muted transition-colors"
              >
                Clear
              </button>
              <button
                onClick={handleNoOverlay}
                disabled={saving}
                className="px-3 py-1.5 text-xs border rounded hover:bg-red-50 text-red-600 border-red-200 transition-colors"
              >
                <X className="w-3 h-3 inline mr-1" />
                No Overlay
              </button>
              {selected?.annotated && (
                <button
                  onClick={handleDelete}
                  className="px-3 py-1.5 text-xs border rounded hover:bg-red-50 text-red-600 border-red-200 transition-colors"
                >
                  <Trash2 className="w-3 h-3 inline mr-1" />
                  Remove
                </button>
              )}
              <button
                onClick={handleSave}
                disabled={!currentBbox || saving}
                className="px-3 py-1.5 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                <Check className="w-3 h-3 inline mr-1" />
                Save
              </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            {selectedKey
              ? "Loading frame..."
              : "Select a frame from the sidebar to annotate"}
          </div>
        )}
      </div>
    </div>
  );
}
