"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FileUpload } from "@/components/FileUpload";
import { ChessBoard } from "@/components/ChessBoard";
import { BboxDrawer, type Bbox } from "@/components/BboxDrawer";
import { testOverlayImage } from "@/lib/api";
import type { OverlayTestResponse } from "@/lib/types";

export default function OverlayTesterPage() {
  const [result, setResult] = useState<OverlayTestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flipped, setFlipped] = useState(false);
  const [theme, setTheme] = useState("lichess_default");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedUrl, setUploadedUrl] = useState<string | null>(null);
  const [manualBbox, setManualBbox] = useState<Bbox | null>(null);
  const [useManualBbox, setUseManualBbox] = useState(false);

  const handleFile = async (file: File) => {
    setUploadedFile(file);
    setUploadedUrl(URL.createObjectURL(file));
    setManualBbox(null);
    setUseManualBbox(false);
    await runTest(file);
  };

  const runTest = async (file?: File) => {
    const f = file || uploadedFile;
    if (!f) return;

    setLoading(true);
    setError(null);
    try {
      const bboxStr =
        useManualBbox && manualBbox
          ? `${manualBbox.x},${manualBbox.y},${manualBbox.w},${manualBbox.h}`
          : undefined;
      const res = await testOverlayImage(f, {
        flipped,
        theme,
        overlay_bbox: bboxStr,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Test failed");
    }
    setLoading(false);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Overlay Tester</h2>
      <p className="text-muted-foreground mb-6">
        Upload a screenshot to detect and read 2D chess board overlays.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Upload + controls */}
        <div className="space-y-4">
          <FileUpload accept="image/*" onFile={handleFile} label="Drop a screenshot or click to browse" />

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={flipped}
                onChange={(e) => setFlipped(e.target.checked)}
                className="rounded"
              />
              Board flipped
            </label>
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground">Theme:</label>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
                className="h-8 rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="lichess_default">Lichess Default</option>
                <option value="chess_com">Chess.com</option>
              </select>
            </div>
          </div>

          {/* Manual bbox mode */}
          {uploadedUrl && (
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={useManualBbox}
                  onChange={(e) => setUseManualBbox(e.target.checked)}
                  className="rounded"
                />
                Draw manual bounding box
              </label>
              {useManualBbox && (
                <>
                  <BboxDrawer
                    imageSrc={uploadedUrl}
                    onBboxChange={setManualBbox}
                    existingBbox={manualBbox}
                  />
                  <Button
                    onClick={() => runTest()}
                    disabled={loading || !manualBbox}
                    size="sm"
                  >
                    Re-test with bbox
                  </Button>
                </>
              )}
            </div>
          )}

          {uploadedFile && !useManualBbox && (
            <Button onClick={() => runTest()} disabled={loading}>
              {loading ? "Testing..." : "Re-test"}
            </Button>
          )}

          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        {/* Right: Results */}
        <div>
          {loading && (
            <p className="text-sm text-muted-foreground">Analyzing image...</p>
          )}
          {result && (
            <div className="space-y-4">
              {/* Annotated image */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Result
                    <Badge variant={result.detected ? "default" : "destructive"}>
                      {result.detected ? "detected" : "not detected"}
                    </Badge>
                    {result.detection_score !== null && (
                      <Badge variant="outline">
                        score: {result.detection_score.toFixed(2)}
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={`data:image/png;base64,${result.annotated_image_b64}`}
                    alt="Annotated"
                    className="w-full rounded border"
                  />
                </CardContent>
              </Card>

              {/* FEN + Board */}
              {result.fen && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      Board Reading
                      <Badge variant="outline">
                        {result.piece_count} pieces
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <code className="text-xs bg-muted px-2 py-1 rounded block break-all">
                      {result.fen}
                    </code>
                    <ChessBoard fen={result.fen} size={280} flipped={flipped} />
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
