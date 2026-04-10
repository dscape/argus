"use client";

import { useState } from "react";
import type { OverlayEvalResult } from "@/lib/api";
import { ChessBoard } from "@/components/ChessBoard";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Check,
  X,
  Clipboard,
  ClipboardCheck,
  Pencil,
} from "lucide-react";

interface OverlayEvalCardProps {
  result: OverlayEvalResult;
  pinned?: boolean;
  onPin?: () => void;
  editedFen?: string;
  onEditFen?: (fen: string) => void;
  onSave?: () => void;
  onReject?: () => void;
  isSaved?: boolean;
  isRejected?: boolean;
  isSaving?: boolean;
}

export default function OverlayEvalCard({
  result,
  pinned,
  onPin,
  editedFen,
  onEditFen,
  onSave,
  onReject,
  isSaved,
  isRejected,
  isSaving,
}: OverlayEvalCardProps) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const currentFen = editedFen ?? result.predicted_fen ?? "";
  const fenLoading = result.fen_loading ?? false;
  const isWarning = result.status === "warning";
  const isNoOverlay = result.status === "no_overlay";
  const detectorFound = result.detector_found !== false;
  const isDetectorMiss = !detectorFound && !!result.image_b64;
  const savedState = result.already_saved || isSaved;
  const locked = savedState || isRejected;

  function copyFen() {
    if (!currentFen) return;
    navigator.clipboard.writeText(currentFen).then(() => {
      setCopiedKey(result.frame_key);
      setTimeout(() => setCopiedKey(null), 2000);
    });
  }

  const statusLabel = fenLoading
    ? "\u2026 Extracting FEN"
    : isNoOverlay
      ? "\u2717 No Overlay"
      : isDetectorMiss
        ? "\u26A0 Detector Miss"
        : isWarning
          ? "\u26A0 Review"
          : result.predicted_fen
            ? "\u2713 FEN Extracted"
            : "\u2713 Overlay Detected";

  const statusColor = isNoOverlay
    ? "text-red-600"
    : isDetectorMiss || isWarning
      ? "text-yellow-600"
      : "text-green-600";

  return (
    <div
      className={`border rounded-lg p-3 space-y-3 ${isRejected ? "opacity-40" : ""} ${savedState ? "border-green-400" : ""} ${(isWarning || isDetectorMiss) && !savedState ? "border-yellow-400" : ""}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            {onPin && (
              <button
                onClick={onPin}
                title={pinned ? "Unpin from top" : "Pin to top"}
                className={`flex-shrink-0 w-5 h-5 flex items-center justify-center rounded transition-colors ${
                  pinned
                    ? "text-foreground"
                    : "text-muted-foreground/40 hover:text-foreground"
                }`}
              >
                <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                  <path d="M9.828.722a.5.5 0 0 1 .354.146l4.95 4.95a.5.5 0 0 1-.707.707l-.71-.71-3.18 3.18a5.5 5.5 0 0 1-1.32 4.988.5.5 0 0 1-.707 0L5.464 10.94l-3.89 3.89a.5.5 0 0 1-.707-.708l3.89-3.889L1.714 7.19a.5.5 0 0 1 0-.707 5.5 5.5 0 0 1 4.988-1.32L9.88 1.985l-.71-.71a.5.5 0 0 1 .5-.853z" />
                </svg>
              </button>
            )}
            <p className="text-sm font-medium font-mono truncate">
              {result.frame_name ?? "frame"}
            </p>
            <a
              href={`/videos/${result.video_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] text-blue-600 hover:underline font-mono truncate"
            >
              {result.video_id}
            </a>
            {savedState && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-green-100 text-green-700 font-medium">
                Saved
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
            {result.overlay_detect_ms != null && (
              <span>
                overlay: {result.overlay_detect_ms}ms | grid: {result.grid_detect_ms}ms | classify: {result.piece_classify_ms}ms
              </span>
            )}
            {result.elapsed_ms != null && result.overlay_detect_ms == null && (
              <span>{result.elapsed_ms}ms</span>
            )}
          </div>
        </div>
        <div className="text-right shrink-0">
          <span className={`text-sm font-bold ${statusColor}`}>
            {statusLabel}
          </span>
        </div>
      </div>

      {/* Side-by-side: source image + predicted board */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-muted-foreground block mb-1">Source Image</label>
          {result.image_b64 ? (
            <img
              src={`data:image/jpeg;base64,${result.image_b64}`}
              alt={result.frame_key}
              className="w-full max-w-[200px] rounded border"
              loading="lazy"
            />
          ) : (
            <div className="w-[200px] h-[200px] rounded border border-dashed flex items-center justify-center text-xs text-muted-foreground">
              {isNoOverlay ? "No overlay detected" : "No image"}
            </div>
          )}
        </div>
        <div>
          <label className="text-xs text-muted-foreground block mb-1">
            Predicted {currentFen ? "" : fenLoading ? "(loading)" : "(none)"}
          </label>
          {fenLoading ? (
            <Skeleton className="w-[200px] h-[200px] rounded" />
          ) : currentFen ? (
            <ChessBoard fen={currentFen} size={200} />
          ) : (
            <div className="w-[200px] h-[200px] rounded border border-dashed flex items-center justify-center text-xs text-muted-foreground">
              No FEN
            </div>
          )}
        </div>
      </div>

      {/* Warning / FEN display */}
      {result.warning && (
        <div className="text-xs px-2 py-1 rounded bg-yellow-50 text-yellow-800 border border-yellow-200">
          {result.warning}
        </div>
      )}
      {currentFen && (
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`text-xs px-2 py-0.5 rounded ${
              isNoOverlay
                ? "bg-red-100 text-red-700"
                : isDetectorMiss || isWarning
                  ? "bg-yellow-100 text-yellow-700"
                  : "bg-green-100 text-green-700"
            }`}
          >
            FEN: {currentFen}
          </span>
        </div>
      )}

      {/* Actions row */}
      <div className="flex items-center gap-1.5">
        {/* Copy FEN */}
        <button
          onClick={copyFen}
          disabled={fenLoading || !currentFen}
          className="flex items-center gap-1 text-xs px-2 py-1 rounded border hover:bg-muted disabled:opacity-40 transition-colors"
          title="Copy FEN"
        >
          {copiedKey === result.frame_key ? (
            <ClipboardCheck className="h-3 w-3 text-green-600" />
          ) : (
            <Clipboard className="h-3 w-3" />
          )}
          <span>{copiedKey === result.frame_key ? "Copied" : "Copy FEN"}</span>
        </button>

        {/* Edit FEN dialog */}
        {onEditFen && (
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
                <DialogTitle>Edit FEN — {result.frame_name}</DialogTitle>
              </DialogHeader>
              <div className="space-y-3">
                <textarea
                  value={currentFen}
                  onChange={(e) => onEditFen(e.target.value)}
                  rows={3}
                  className="w-full text-sm font-mono px-3 py-2 border rounded resize-none"
                />
                {currentFen && <ChessBoard fen={currentFen} size={280} />}
                {currentFen !== result.predicted_fen && (
                  <p className="text-xs text-blue-600">edited</p>
                )}
              </div>
            </DialogContent>
          </Dialog>
        )}

        <div className="flex-1" />

        {/* Save */}
        {onSave && (
          <button
            onClick={onSave}
            disabled={locked || isSaving || fenLoading || !currentFen}
            className="p-1 rounded border bg-green-600 text-white border-green-600 disabled:opacity-40"
            title="Save"
          >
            <Check className="h-3.5 w-3.5" />
          </button>
        )}

        {/* Reject */}
        {onReject && (
          <button
            onClick={onReject}
            disabled={savedState}
            className={`p-1 rounded border ${
              isRejected
                ? "bg-red-100 text-red-700 border-red-300"
                : "text-muted-foreground hover:text-foreground"
            } disabled:opacity-40`}
            title={isRejected ? "Undo reject" : "Reject"}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
