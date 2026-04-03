"use client";

import { useRef, useState } from "react";
import { ChessBoard } from "@/components/ChessBoard";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { updateVideoStatus } from "@/lib/api";
import {
  Check,
  X,
  Ban,
  Clipboard,
  ClipboardCheck,
  Pencil,
} from "lucide-react";

interface ExtractionResult {
  frame_key?: string;
  video_id: string;
  frame_name?: string;
  status: "ok" | "warning" | "no_overlay" | "detected";
  warning?: string;
  image_b64?: string;
  predicted_fen?: string;
  fen_loading?: boolean;
}

function resultKey(r: ExtractionResult): string {
  return r.frame_key ?? r.video_id;
}

function youtubeUrl(videoId: string): string {
  return `https://www.youtube.com/watch?v=${videoId}`;
}

export default function ExtractOverlaysPage() {
  const [results, setResults] = useState<ExtractionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [videoIdsInput, setVideoIdsInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  // Track user decisions: confirmed FEN edits, rejected, or saved
  const [editedFens, setEditedFens] = useState<Record<string, string>>({});
  const [rejected, setRejected] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState<Set<string>>(new Set());
  const [rejectedVideos, setRejectedVideos] = useState<Set<string>>(new Set());
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  async function runExtraction() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setResults([]);
    setEditedFens({});
    setRejected(new Set());
    setSaved(new Set());
    setRejectedVideos(new Set());
    setProgress(null);

    try {
      // Step 1: get candidate video IDs (fast)
      const params = new URLSearchParams();
      const trimmed = videoIdsInput.trim();
      if (trimmed) params.set("video_ids", trimmed);
      const candidateUrl = `/api/models/overlay-test/extract-candidates${params.toString() ? `?${params}` : ""}`;
      const candRes = await fetch(candidateUrl, {
        signal: controller.signal,
      });
      if (!candRes.ok) throw new Error(await candRes.text());
      const { video_ids } = await candRes.json();

      setProgress({ current: 0, total: video_ids.length });

      // Step 2: process each video — fast detect, then async FEN
      for (let i = 0; i < video_ids.length; i++) {
        if (controller.signal.aborted) break;

        // Fast phase: detect overlay + grid, get crop image
        const res = await fetch("/api/models/overlay-test/extract-detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: video_ids[i] }),
          signal: controller.signal,
        });

        if (res.ok) {
          const result: ExtractionResult = await res.json();

          if (result.status === "detected" && result.frame_name) {
            // Mark as loading FEN, append immediately
            result.fen_loading = true;
            setResults((prev) => [...prev, result]);

            // Slow phase: classify FEN asynchronously (non-blocking)
            const frameKey = resultKey(result);
            const frameName = result.frame_name;
            const videoId = result.video_id;
            fetch("/api/models/overlay-test/extract-fen", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                video_id: videoId,
                frame_name: frameName,
              }),
            })
              .then((fenRes) => fenRes.json())
              .then((fenData) => {
                // If classification failed (error/no pieces), demote to no_overlay
                if (fenData.status === "error" || !fenData.predicted_fen) {
                  setResults((prev) =>
                    prev.map((r) =>
                      resultKey(r) === frameKey
                        ? {
                            ...r,
                            status: "no_overlay" as const,
                            warning: fenData.warning,
                            fen_loading: false,
                          }
                        : r,
                    ),
                  );
                  return;
                }
                setResults((prev) =>
                  prev.map((r) =>
                    resultKey(r) === frameKey
                      ? {
                          ...r,
                          predicted_fen: fenData.predicted_fen,
                          status: fenData.status,
                          warning: fenData.warning,
                          fen_loading: false,
                        }
                      : r,
                  ),
                );
                if (fenData.predicted_fen) {
                  setEditedFens((prev) => ({
                    ...prev,
                    [frameKey]: fenData.predicted_fen,
                  }));
                }
              })
              .catch(() => {
                setResults((prev) =>
                  prev.map((r) =>
                    resultKey(r) === frameKey
                      ? {
                          ...r,
                          status: "no_overlay" as const,
                          warning: "FEN classification failed",
                          fen_loading: false,
                        }
                      : r,
                  ),
                );
              });
          } else {
            // no_overlay or other non-detected status
            setResults((prev) => [...prev, result]);
          }
        }

        setProgress({ current: i + 1, total: video_ids.length });
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      alert(e instanceof Error ? e.message : "Failed to extract");
    } finally {
      setLoading(false);
    }
  }

  function stopExtraction() {
    abortRef.current?.abort();
  }

  async function saveOne(r: ExtractionResult) {
    const key = resultKey(r);
    const fen = editedFens[key];
    if (!fen) return;

    setSavingId(key);
    try {
      const res = await fetch("/api/models/overlay-test/extract-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmations: [
            {
              video_id: r.video_id,
              frame_name: r.frame_name,
              fen,
            },
          ],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.errors?.length > 0) {
        alert(data.errors.join("\n"));
      } else {
        setSaved((prev) => new Set(prev).add(key));
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSavingId(null);
    }
  }

  async function rejectVideo(videoId: string) {
    try {
      await updateVideoStatus(videoId, "rejected");
      setRejectedVideos((prev) => new Set(prev).add(videoId));
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to reject video");
    }
  }

  function copyFen(key: string, fen: string) {
    navigator.clipboard.writeText(fen).then(() => {
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2000);
    });
  }

  const reviewableResults = results.filter(
    (r) =>
      r.status === "ok" ||
      r.status === "warning" ||
      r.status === "detected",
  );
  const noOverlayResults = results.filter((r) => r.status === "no_overlay");

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-lg font-semibold">Extract Overlay Annotations</h2>
        {loading ? (
          <button
            onClick={stopExtraction}
            className="px-4 py-1.5 bg-red-600 text-white rounded text-sm"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={runExtraction}
            className="px-4 py-1.5 bg-foreground text-background rounded text-sm"
          >
            Extract
          </button>
        )}
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm text-muted-foreground whitespace-nowrap">
          Video IDs:
        </label>
        <input
          type="text"
          value={videoIdsInput}
          onChange={(e) => setVideoIdsInput(e.target.value)}
          placeholder="e.g. Unu6antTBGs, ycitHs8_NY4 (leave empty for all)"
          className="flex-1 px-2 py-1.5 border rounded text-sm font-mono"
          disabled={loading}
        />
      </div>

      <p className="text-sm text-muted-foreground">
        Extracts overlay crops from cached screening frames, auto-labels with
        FEN, and lets you review before saving. Saved images go to{" "}
        <code className="text-xs">data/overlay/val_real/</code>.
      </p>

      {/* Progress bar */}
      {progress && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Processing {progress.current} / {progress.total} frames
            </span>
            <span>
              {reviewableResults.length} found
              {noOverlayResults.length > 0 &&
                ` · ${noOverlayResults.length} no overlay`}
              {saved.size > 0 && ` · ${saved.size} saved`}
            </span>
          </div>
          <div className="w-full bg-muted rounded-full h-1.5">
            <div
              className="bg-foreground h-1.5 rounded-full transition-all duration-300"
              style={{
                width: `${(progress.current / progress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Summary (after completion) */}
      {!loading && results.length > 0 && (
        <div className="flex items-center gap-4 text-sm">
          <span className="text-green-600 font-medium">
            {reviewableResults.length} extracted
          </span>
          {noOverlayResults.length > 0 && (
            <span className="text-yellow-600 font-medium">
              {noOverlayResults.length} no overlay
            </span>
          )}
          {saved.size > 0 && (
            <span className="text-muted-foreground">{saved.size} saved</span>
          )}
        </div>
      )}

      {/* No overlay summary */}
      {noOverlayResults.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-yellow-600">
            {noOverlayResults.length} frames with no overlay detected
          </summary>
          <div className="mt-2 space-y-2">
            {noOverlayResults.map((r) => {
              const key = resultKey(r);
              const isVideoRejected = rejectedVideos.has(r.video_id);
              return (
                <div
                  key={key}
                  className={`flex items-center gap-3 text-xs ${isVideoRejected ? "opacity-40" : ""}`}
                >
                  <a
                    href={youtubeUrl(r.video_id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline font-mono"
                  >
                    {r.video_id}
                  </a>
                  <button
                    onClick={() => rejectVideo(r.video_id)}
                    disabled={isVideoRejected}
                    className={`px-2 py-0.5 rounded border text-xs ${
                      isVideoRejected
                        ? "bg-red-100 text-red-700 border-red-300"
                        : "text-red-600 border-red-300 hover:bg-red-50"
                    } disabled:opacity-40`}
                  >
                    {isVideoRejected ? "Video Rejected" : "Reject Video"}
                  </button>
                </div>
              );
            })}
          </div>
        </details>
      )}

      {/* Reviewable results — compact card grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
        {reviewableResults.map((r) => {
          const key = resultKey(r);
          const isRejected = rejected.has(key);
          const isVideoRejected = rejectedVideos.has(r.video_id);
          const isSaved = saved.has(key);
          const isSaving = savingId === key;
          const currentFen = editedFens[key] ?? r.predicted_fen ?? "";
          const isWarning = r.status === "warning";
          const locked = isRejected || isSaved || isVideoRejected;
          const fenLoading = r.fen_loading ?? false;
          const isCopied = copiedKey === key;

          return (
            <div
              key={key}
              className={`border rounded-lg p-3 space-y-2 ${isRejected || isVideoRejected ? "opacity-40" : ""} ${isSaved ? "border-green-400" : ""} ${isWarning && !isSaved ? "border-yellow-400" : ""}`}
            >
              {/* Header */}
              <div className="flex items-center justify-between gap-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-xs font-mono font-medium truncate">
                    {r.frame_name ?? "frame"}
                  </span>
                  <a
                    href={youtubeUrl(r.video_id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] text-blue-600 hover:underline font-mono truncate"
                  >
                    {r.video_id}
                  </a>
                  {isWarning && !isSaved && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-yellow-100 text-yellow-700 font-medium whitespace-nowrap">
                      {r.warning}
                    </span>
                  )}
                  {isSaved && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-green-100 text-green-700 font-medium">
                      Saved
                    </span>
                  )}
                  {isVideoRejected && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-red-100 text-red-700 font-medium">
                      Rejected
                    </span>
                  )}
                </div>
              </div>

              {/* Images: overlay crop + board side by side */}
              <div className="flex gap-2">
                {/* Source crop */}
                <div className="flex-1 min-w-0">
                  <label className="text-[10px] text-muted-foreground block mb-0.5">
                    Overlay
                  </label>
                  {r.image_b64 && (
                    <img
                      src={`data:image/jpeg;base64,${r.image_b64}`}
                      alt={key}
                      className="w-full max-w-[140px] rounded border"
                    />
                  )}
                </div>

                {/* Predicted board / skeleton */}
                <div className="flex-1 min-w-0">
                  <label className="text-[10px] text-muted-foreground block mb-0.5">
                    Predicted
                  </label>
                  {fenLoading ? (
                    <Skeleton className="w-[120px] h-[120px] rounded" />
                  ) : currentFen ? (
                    <ChessBoard fen={currentFen} size={120} />
                  ) : (
                    <div className="w-[120px] h-[120px] rounded border border-dashed flex items-center justify-center text-[10px] text-muted-foreground">
                      No FEN
                    </div>
                  )}
                </div>
              </div>

              {/* Actions row */}
              <div className="flex items-center gap-1.5">
                {/* Copy FEN */}
                <button
                  onClick={() => currentFen && copyFen(key, currentFen)}
                  disabled={fenLoading || !currentFen}
                  className="flex items-center gap-1 text-xs px-2 py-1 rounded border hover:bg-muted disabled:opacity-40 transition-colors"
                  title="Copy FEN"
                >
                  {isCopied ? (
                    <ClipboardCheck className="h-3 w-3 text-green-600" />
                  ) : (
                    <Clipboard className="h-3 w-3" />
                  )}
                  <span>{isCopied ? "Copied" : "Copy FEN"}</span>
                </button>

                {/* Edit FEN dialog */}
                <Dialog>
                  <DialogTrigger asChild>
                    <button
                      disabled={locked || fenLoading}
                      className="p-1 rounded border hover:bg-muted disabled:opacity-40"
                      title="Edit FEN"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Edit FEN — {r.frame_name}</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-3">
                      <textarea
                        value={currentFen}
                        onChange={(e) =>
                          setEditedFens((prev) => ({
                            ...prev,
                            [key]: e.target.value,
                          }))
                        }
                        rows={3}
                        className="w-full text-sm font-mono px-3 py-2 border rounded resize-none"
                      />
                      {currentFen && <ChessBoard fen={currentFen} size={280} />}
                      {currentFen !== r.predicted_fen && (
                        <p className="text-xs text-blue-600">edited</p>
                      )}
                    </div>
                  </DialogContent>
                </Dialog>

                <div className="flex-1" />

                {/* Save */}
                <button
                  onClick={() => saveOne(r)}
                  disabled={locked || isSaving || fenLoading || !currentFen}
                  className="p-1 rounded border bg-green-600 text-white border-green-600 disabled:opacity-40"
                  title="Save"
                >
                  <Check className="h-3.5 w-3.5" />
                </button>

                {/* Reject */}
                <button
                  onClick={() =>
                    setRejected((prev) => {
                      const next = new Set(prev);
                      if (next.has(key)) next.delete(key);
                      else next.add(key);
                      return next;
                    })
                  }
                  disabled={isSaved || isVideoRejected}
                  className={`p-1 rounded border ${
                    isRejected
                      ? "bg-red-100 text-red-700 border-red-300"
                      : "text-muted-foreground hover:text-foreground"
                  } disabled:opacity-40`}
                  title={isRejected ? "Undo reject" : "Reject"}
                >
                  <X className="h-3.5 w-3.5" />
                </button>

                {/* Reject Video */}
                <button
                  onClick={() => rejectVideo(r.video_id)}
                  disabled={isVideoRejected || isSaved}
                  className={`p-1 rounded border ${
                    isVideoRejected
                      ? "bg-red-100 text-red-700 border-red-300"
                      : "text-red-600 border-red-300 hover:bg-red-50"
                  } disabled:opacity-40`}
                  title={isVideoRejected ? "Video rejected" : "Reject video"}
                >
                  <Ban className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
