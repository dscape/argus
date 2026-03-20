"use client";

import { useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChessBoard } from "@/components/ChessBoard";
import { MoveList } from "@/components/MoveList";
import {
  openVideo,
  videoFrameUrl,
  readOverlayFrame,
  detectVideoMoves,
  deleteVideoSession,
} from "@/lib/api";
import type {
  VideoSession,
  FrameOverlayResponse,
  VideoMoveDetectionResponse,
} from "@/lib/types";

export default function VideoAnnotatorPage() {
  const [videoPath, setVideoPath] = useState("");
  const [channelHandle, setChannelHandle] = useState("");
  const [session, setSession] = useState<VideoSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Frame viewer
  const [frameIndex, setFrameIndex] = useState(0);
  const [overlayResult, setOverlayResult] = useState<FrameOverlayResponse | null>(null);
  const [overlayLoading, setOverlayLoading] = useState(false);

  // Move detection
  const [detection, setDetection] = useState<VideoMoveDetectionResponse | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [sampleFps, setSampleFps] = useState(2.0);

  const sessionIdRef = useRef<string | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionIdRef.current) {
        deleteVideoSession(sessionIdRef.current).catch(() => {});
      }
    };
  }, []);

  const handleOpen = async () => {
    if (!videoPath) return;

    // Clean up previous session
    if (sessionIdRef.current) {
      deleteVideoSession(sessionIdRef.current).catch(() => {});
    }

    setLoading(true);
    setError(null);
    setSession(null);
    setDetection(null);
    setOverlayResult(null);

    try {
      const s = await openVideo(videoPath, channelHandle || undefined);
      setSession(s);
      sessionIdRef.current = s.session_id;
      setFrameIndex(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to open video");
    }
    setLoading(false);
  };

  const handleReadOverlay = async () => {
    if (!session) return;
    setOverlayLoading(true);
    try {
      const res = await readOverlayFrame(session.session_id, frameIndex);
      setOverlayResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Overlay read failed");
    }
    setOverlayLoading(false);
  };

  const handleDetectMoves = async () => {
    if (!session) return;
    setDetecting(true);
    try {
      const res = await detectVideoMoves(session.session_id, sampleFps);
      setDetection(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Move detection failed");
    }
    setDetecting(false);
  };

  const jumpToFrame = (idx: number) => {
    setFrameIndex(idx);
    setOverlayResult(null);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Video Annotator</h2>
      <p className="text-muted-foreground mb-6">
        Open a video file, step through frames, read overlays, and detect moves.
      </p>

      {/* Open form */}
      {!session && (
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1">
                Video path (on server filesystem)
              </label>
              <input
                type="text"
                value={videoPath}
                onChange={(e) => setVideoPath(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                placeholder="/path/to/video.mp4"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1">
                Channel handle (optional, for calibration)
              </label>
              <input
                type="text"
                value={channelHandle}
                onChange={(e) => setChannelHandle(e.target.value)}
                className="h-9 w-64 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="@ChessChannel"
              />
            </div>
            <Button onClick={handleOpen} disabled={loading || !videoPath}>
              {loading ? "Opening..." : "Open Video"}
            </Button>
          </CardContent>
        </Card>
      )}

      {error && <p className="text-sm text-destructive mt-4">{error}</p>}

      {/* Video session */}
      {session && (
        <div className="space-y-4">
          {/* Metadata */}
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline">{session.fps.toFixed(1)} fps</Badge>
            <Badge variant="outline">{session.total_frames} frames</Badge>
            <Badge variant="outline">{session.duration_seconds.toFixed(1)}s</Badge>
            <Badge variant="outline">{session.width}x{session.height}</Badge>
            <Badge variant={session.has_calibration ? "default" : "destructive"}>
              {session.has_calibration ? "calibrated" : "no calibration"}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (sessionIdRef.current) deleteVideoSession(sessionIdRef.current).catch(() => {});
                setSession(null);
                sessionIdRef.current = null;
                setDetection(null);
                setOverlayResult(null);
              }}
            >
              Close
            </Button>
          </div>

          <Tabs defaultValue="frames">
            <TabsList>
              <TabsTrigger value="frames">Frames</TabsTrigger>
              <TabsTrigger value="detect">Detect Moves</TabsTrigger>
            </TabsList>

            {/* Frames tab */}
            <TabsContent value="frames" className="space-y-4">
              {/* Frame scrubber */}
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium w-20">
                    Frame {frameIndex}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={session.total_frames - 1}
                    value={frameIndex}
                    onChange={(e) => {
                      setFrameIndex(parseInt(e.target.value, 10));
                      setOverlayResult(null);
                    }}
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground">
                    {(frameIndex / session.fps).toFixed(2)}s
                  </span>
                </div>

                {/* Frame image */}
                <img
                  src={videoFrameUrl(session.session_id, frameIndex)}
                  alt={`Frame ${frameIndex}`}
                  className="max-w-full rounded border"
                />
              </div>

              {/* Overlay read */}
              {session.has_calibration && (
                <div className="space-y-2">
                  <Button onClick={handleReadOverlay} disabled={overlayLoading} size="sm">
                    {overlayLoading ? "Reading..." : "Read Overlay"}
                  </Button>

                  {overlayResult && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Overlay Crop</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <img
                            src={`data:image/jpeg;base64,${overlayResult.overlay_crop_b64}`}
                            alt="Overlay"
                            className="w-full rounded border"
                          />
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Camera Crop</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <img
                            src={`data:image/jpeg;base64,${overlayResult.camera_crop_b64}`}
                            alt="Camera"
                            className="w-full rounded border"
                          />
                        </CardContent>
                      </Card>
                      {overlayResult.fen && (
                        <Card>
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">Board Reading</CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-2">
                            <ChessBoard fen={overlayResult.fen} size={200} />
                            <code className="text-xs bg-muted px-2 py-1 rounded block break-all">
                              {overlayResult.fen}
                            </code>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  )}
                </div>
              )}
            </TabsContent>

            {/* Detect moves tab */}
            <TabsContent value="detect" className="space-y-4">
              <div className="flex items-center gap-3">
                <div>
                  <label className="text-xs font-medium text-muted-foreground block mb-1">
                    Sample FPS
                  </label>
                  <input
                    type="number"
                    value={sampleFps}
                    onChange={(e) => setSampleFps(parseFloat(e.target.value) || 2.0)}
                    className="h-9 w-24 rounded-md border border-input bg-background px-3 text-sm"
                    step="0.5"
                    min="0.5"
                  />
                </div>
                <Button
                  onClick={handleDetectMoves}
                  disabled={detecting || !session.has_calibration}
                  className="mt-4"
                >
                  {detecting ? "Detecting..." : "Detect All Moves"}
                </Button>
              </div>

              {!session.has_calibration && (
                <p className="text-sm text-muted-foreground">
                  Calibration required for move detection. Add one in the Calibration Editor first.
                </p>
              )}

              {detection && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">
                      {detection.num_frames_sampled} frames sampled
                    </Badge>
                    <Badge variant="outline">
                      {detection.num_readable} readable
                    </Badge>
                    <Badge variant="outline">
                      {detection.segments.length} game{detection.segments.length !== 1 ? "s" : ""}
                    </Badge>
                  </div>

                  {detection.segments.map((seg) => (
                    <Card key={seg.game_index}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm flex items-center gap-2">
                          Game {seg.game_index + 1}
                          <Badge variant="outline">{seg.num_moves} moves</Badge>
                          <span className="text-xs text-muted-foreground font-normal">
                            frames {seg.start_frame} – {seg.end_frame}
                          </span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <code className="text-xs bg-muted px-2 py-1 rounded block break-all">
                          {seg.pgn_moves}
                        </code>
                        <MoveList
                          moves={seg.moves.map((m) => ({
                            frame_index: m.frame_idx,
                            uci: m.move_uci,
                            san: m.move_san,
                          }))}
                          onMoveClick={jumpToFrame}
                          activeFrame={frameIndex}
                        />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}
