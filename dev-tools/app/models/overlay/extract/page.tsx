"use client";

import { useState } from "react";
import { ChessBoard } from "@/components/ChessBoard";

interface ExtractionResult {
  clip_id: number;
  video_id: string;
  status: "ok" | "warning" | "error" | "pending";
  error?: string;
  warning?: string;
  image_b64?: string;
  predicted_fen?: string;
}

export default function ExtractOverlaysPage() {
  const [results, setResults] = useState<ExtractionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState<number | null>(null);
  const [videoIdsInput, setVideoIdsInput] = useState("");

  // Track user decisions: confirmed FEN edits, rejected, or saved
  const [editedFens, setEditedFens] = useState<Record<number, string>>({});
  const [rejected, setRejected] = useState<Set<number>>(new Set());
  const [saved, setSaved] = useState<Set<number>>(new Set());

  async function runExtraction() {
    setLoading(true);
    setResults([]);
    setEditedFens({});
    setRejected(new Set());
    setSaved(new Set());
    try {
      const params = new URLSearchParams();
      const trimmed = videoIdsInput.trim();
      if (trimmed) {
        params.set("video_ids", trimmed);
      }
      const url = `/api/models/overlay-test/extract-preview${params.toString() ? `?${params}` : ""}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResults(data.results);
      // Pre-populate editedFens with predicted FENs for ok + warning results
      const initial: Record<number, string> = {};
      for (const r of data.results) {
        if ((r.status === "ok" || r.status === "warning") && r.predicted_fen) {
          initial[r.clip_id] = r.predicted_fen;
        }
      }
      setEditedFens(initial);
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to extract");
    } finally {
      setLoading(false);
    }
  }

  async function saveOne(r: ExtractionResult) {
    const fen = editedFens[r.clip_id];
    if (!fen) return;

    setSavingId(r.clip_id);
    try {
      const res = await fetch("/api/models/overlay-test/extract-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmations: [{ clip_id: r.clip_id, video_id: r.video_id, fen }],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.errors?.length > 0) {
        alert(data.errors.join("\n"));
      } else {
        setSaved((prev) => new Set(prev).add(r.clip_id));
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSavingId(null);
    }
  }

  const reviewableResults = results.filter(
    (r) => r.status === "ok" || r.status === "warning",
  );
  const errorResults = results.filter((r) => r.status === "error");
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-lg font-semibold">Extract Overlay Annotations</h2>
        <button
          onClick={runExtraction}
          disabled={loading}
          className="px-4 py-1.5 bg-foreground text-background rounded text-sm disabled:opacity-50"
        >
          {loading ? "Extracting..." : "Extract from Video Clips"}
        </button>
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
        />
      </div>

      <p className="text-sm text-muted-foreground">
        Extracts overlay crops from calibrated video clips, auto-labels with
        FEN, and lets you review before saving. Saved images go to{" "}
        <code className="text-xs">data/chess_positions/test_real/</code>.
      </p>

      {results.length > 0 && (
        <div className="flex items-center gap-4 text-sm">
          <span className="text-green-600 font-medium">
            {reviewableResults.length} extracted
          </span>
          {errorResults.length > 0 && (
            <span className="text-red-600 font-medium">
              {errorResults.length} errors
            </span>
          )}
          {saved.size > 0 && (
            <span className="text-muted-foreground">
              {saved.size} saved
            </span>
          )}
        </div>
      )}

      {/* Error summary */}
      {errorResults.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground">
            {errorResults.length} clips with errors
          </summary>
          <div className="mt-2 space-y-1">
            {errorResults.map((r) => (
              <div
                key={r.clip_id}
                className="flex items-center gap-2 text-xs text-red-600"
              >
                <span className="font-mono">
                  clip {r.clip_id} ({r.video_id})
                </span>
                <span>{r.error}</span>
                {r.image_b64 && (
                  <img
                    src={`data:image/jpeg;base64,${r.image_b64}`}
                    alt={`clip ${r.clip_id}`}
                    className="w-16 h-16 rounded border object-cover"
                  />
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Reviewable results (ok + warning) */}
      <div className="space-y-4">
        {reviewableResults.map((r) => {
          const isRejected = rejected.has(r.clip_id);
          const isSaved = saved.has(r.clip_id);
          const isSaving = savingId === r.clip_id;
          const currentFen = editedFens[r.clip_id] ?? r.predicted_fen ?? "";
          const isWarning = r.status === "warning";
          const locked = isRejected || isSaved;

          return (
            <div
              key={r.clip_id}
              className={`border rounded-lg p-3 space-y-3 ${isRejected ? "opacity-40" : ""} ${isSaved ? "border-green-400" : ""} ${isWarning && !isSaved ? "border-yellow-400" : ""}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-mono font-medium">
                    clip {r.clip_id}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {r.video_id}
                  </span>
                  {isWarning && !isSaved && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 font-medium">
                      {r.warning}
                    </span>
                  )}
                  {isSaved && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-medium">
                      Saved
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => saveOne(r)}
                    disabled={locked || isSaving || !currentFen}
                    className="text-xs px-2 py-1 rounded border bg-green-600 text-white border-green-600 disabled:opacity-40"
                  >
                    {isSaving ? "Saving..." : "Save"}
                  </button>
                  <button
                    onClick={() =>
                      setRejected((prev) => {
                        const next = new Set(prev);
                        if (next.has(r.clip_id)) next.delete(r.clip_id);
                        else next.add(r.clip_id);
                        return next;
                      })
                    }
                    disabled={isSaved}
                    className={`text-xs px-2 py-1 rounded border ${
                      isRejected
                        ? "bg-red-100 text-red-700 border-red-300"
                        : "text-muted-foreground hover:text-foreground"
                    } disabled:opacity-40`}
                  >
                    {isRejected ? "Rejected" : "Reject"}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                {/* Source crop */}
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">
                    Overlay Crop
                  </label>
                  {r.image_b64 && (
                    <img
                      src={`data:image/jpeg;base64,${r.image_b64}`}
                      alt={`clip ${r.clip_id}`}
                      className="w-full max-w-[200px] rounded border"
                    />
                  )}
                </div>

                {/* Predicted board visualization */}
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">
                    Predicted Board
                  </label>
                  {currentFen && <ChessBoard fen={currentFen} size={200} />}
                </div>

                {/* FEN editor */}
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">
                    FEN (editable)
                  </label>
                  <textarea
                    value={currentFen}
                    onChange={(e) =>
                      setEditedFens((prev) => ({
                        ...prev,
                        [r.clip_id]: e.target.value,
                      }))
                    }
                    rows={3}
                    className="w-full text-xs font-mono px-2 py-1 border rounded resize-none"
                    disabled={locked}
                  />
                  {currentFen !== r.predicted_fen && (
                    <p className="text-[10px] text-blue-600 mt-0.5">
                      edited
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
