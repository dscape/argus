"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FileUpload } from "@/components/FileUpload";
import { BboxDrawer, type Bbox } from "@/components/BboxDrawer";
import { listCalibrations, saveCalibration, deleteCalibration } from "@/lib/api";
import type { CalibrationEntry } from "@/lib/types";

export default function CalibrationPage() {
  const [calibrations, setCalibrations] = useState<CalibrationEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);

  // Form state
  const [channelHandle, setChannelHandle] = useState("");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [overlayBbox, setOverlayBbox] = useState<Bbox | null>(null);
  const [cameraBbox, setCameraBbox] = useState<Bbox | null>(null);
  const [drawingMode, setDrawingMode] = useState<"overlay" | "camera">("overlay");
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [boardTheme, setBoardTheme] = useState("lichess_default");

  const fetchCalibrations = useCallback(async () => {
    try {
      const cals = await listCalibrations();
      setCalibrations(cals);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load calibrations");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchCalibrations();
  }, [fetchCalibrations]);

  const handleImageFile = (file: File) => {
    setImageSrc(URL.createObjectURL(file));
    setOverlayBbox(null);
    setCameraBbox(null);
  };

  const handleBboxChange = (bbox: Bbox | null) => {
    if (drawingMode === "overlay") {
      setOverlayBbox(bbox);
    } else {
      setCameraBbox(bbox);
    }
  };

  const handleSave = async () => {
    if (!channelHandle || !overlayBbox || !cameraBbox) return;
    try {
      await saveCalibration(channelHandle, {
        overlay: [overlayBbox.x, overlayBbox.y, overlayBbox.w, overlayBbox.h],
        camera: [cameraBbox.x, cameraBbox.y, cameraBbox.w, cameraBbox.h],
        ref_resolution: [1920, 1080],
        board_flipped: boardFlipped,
        board_theme: boardTheme,
      });
      setEditing(false);
      resetForm();
      fetchCalibrations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const handleDelete = async (channel: string) => {
    try {
      await deleteCalibration(channel);
      fetchCalibrations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const handleEdit = (cal: CalibrationEntry) => {
    setChannelHandle(cal.channel_handle);
    setOverlayBbox({ x: cal.overlay[0], y: cal.overlay[1], w: cal.overlay[2], h: cal.overlay[3] });
    setCameraBbox({ x: cal.camera[0], y: cal.camera[1], w: cal.camera[2], h: cal.camera[3] });
    setBoardFlipped(cal.board_flipped);
    setBoardTheme(cal.board_theme);
    setEditing(true);
  };

  const resetForm = () => {
    setChannelHandle("");
    setImageSrc(null);
    setOverlayBbox(null);
    setCameraBbox(null);
    setBoardFlipped(false);
    setBoardTheme("lichess_default");
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Calibration Editor</h2>
      <p className="text-muted-foreground mb-6">
        Configure overlay and camera crop regions for YouTube channels.
      </p>

      {error && <p className="text-sm text-destructive mb-4">{error}</p>}

      {/* Calibration list */}
      {!editing && (
        <>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">
              Saved Calibrations ({calibrations.length})
            </h3>
            <Button onClick={() => setEditing(true)}>Add New</Button>
          </div>

          {loading ? (
            <p className="text-sm text-muted-foreground">Loading...</p>
          ) : calibrations.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No calibrations saved yet. Click &quot;Add New&quot; to create one.
            </p>
          ) : (
            <div className="border rounded-md overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left px-3 py-2 font-medium">Channel</th>
                    <th className="text-left px-3 py-2 font-medium">Overlay bbox</th>
                    <th className="text-left px-3 py-2 font-medium">Camera bbox</th>
                    <th className="text-left px-3 py-2 font-medium">Theme</th>
                    <th className="text-right px-3 py-2 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {calibrations.map((cal) => (
                    <tr key={cal.channel_handle} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-3 py-2 font-mono text-xs">{cal.channel_handle}</td>
                      <td className="px-3 py-2 text-xs text-muted-foreground">
                        [{cal.overlay.join(", ")}]
                      </td>
                      <td className="px-3 py-2 text-xs text-muted-foreground">
                        [{cal.camera.join(", ")}]
                      </td>
                      <td className="px-3 py-2 text-xs">
                        {cal.board_theme}
                        {cal.board_flipped && (
                          <Badge variant="outline" className="ml-1 text-[10px]">flipped</Badge>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right space-x-1">
                        <Button variant="outline" size="sm" onClick={() => handleEdit(cal)}>
                          Edit
                        </Button>
                        <Button variant="destructive" size="sm" onClick={() => handleDelete(cal.channel_handle)}>
                          Delete
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* Edit/Add form */}
      {editing && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              {channelHandle ? `Edit: ${channelHandle}` : "New Calibration"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1">
                Channel handle
              </label>
              <input
                type="text"
                value={channelHandle}
                onChange={(e) => setChannelHandle(e.target.value)}
                className="h-9 w-64 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="@ChessChannel"
              />
            </div>

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

            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1">
                Reference frame image
              </label>
              <FileUpload accept="image/*" onFile={handleImageFile} label="Drop a reference frame image" />
            </div>

            {imageSrc && (
              <>
                <div className="flex items-center gap-2">
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
                <BboxDrawer
                  imageSrc={imageSrc}
                  onBboxChange={handleBboxChange}
                  existingBbox={drawingMode === "overlay" ? overlayBbox : cameraBbox}
                  secondBbox={drawingMode === "overlay" ? cameraBbox : overlayBbox}
                />
              </>
            )}

            <div className="flex gap-2">
              <Button
                onClick={handleSave}
                disabled={!channelHandle || !overlayBbox || !cameraBbox}
              >
                Save
              </Button>
              <Button variant="outline" onClick={() => { setEditing(false); resetForm(); }}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
