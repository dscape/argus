"use client";

import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { clipFrameUrl } from "@/lib/api";
import type { ClipInspectResponse } from "@/lib/types";

interface ClipInspectorProps {
  open: boolean;
  onClose: () => void;
  filename: string;
  sessionId: string;
  clipInfo: ClipInspectResponse;
}

export function ClipInspector({
  open,
  onClose,
  filename,
  sessionId,
  clipInfo,
}: ClipInspectorProps) {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent fullscreen>
        <DialogHeader>
          <div className="flex items-center gap-3">
            <DialogTitle>{filename}</DialogTitle>
            <Badge variant="outline">{clipInfo.file_size_mb} MB</Badge>
            <Badge variant={clipInfo.replay_valid ? "default" : "destructive"}>
              {clipInfo.replay_valid ? "valid" : "invalid"}
            </Badge>
          </div>
        </DialogHeader>

        <div className="mb-4">
          <h4 className="mb-2 text-sm font-medium">Synthetic</h4>
          <div className="overflow-hidden rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-3 py-1 text-left font-medium">Name</th>
                  <th className="px-3 py-1 text-left font-medium">Shape</th>
                  <th className="px-3 py-1 text-left font-medium">Dtype</th>
                </tr>
              </thead>
              <tbody>
                {clipInfo.tensors.map((tensor) => (
                  <tr key={tensor.name} className="border-b last:border-0">
                    <td className="px-3 py-1 font-mono text-xs">{tensor.name}</td>
                    <td className="px-3 py-1 font-mono text-xs">[{tensor.shape.join(", ")}]</td>
                    <td className="px-3 py-1 text-xs text-muted-foreground">{tensor.dtype}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {clipInfo.moves.length > 0 && (
          <div className="mb-4">
            <h4 className="mb-2 text-sm font-medium">Moves ({clipInfo.total_moves})</h4>
            <div className="flex flex-wrap gap-1">
              {clipInfo.moves.map((move, index) => (
                <span
                  key={index}
                  className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono"
                  title={`Frame ${move.frame_index}`}
                >
                  {move.san || move.uci}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="mb-4">
          <h4 className="mb-2 text-sm font-medium">Frames ({clipInfo.num_frames})</h4>
          <div className="grid grid-cols-4 gap-2 md:grid-cols-8">
            {Array.from({ length: clipInfo.num_frames }, (_, index) => (
              <div key={index} className="relative">
                <img
                  src={clipFrameUrl(sessionId, index)}
                  alt={`Frame ${index}`}
                  className="w-full rounded border"
                />
                <span className="absolute bottom-0 right-0 rounded-tl bg-black/60 px-1 text-[10px] text-white">
                  {index}
                </span>
              </div>
            ))}
          </div>
        </div>

        {clipInfo.replay_error && (
          <p className="text-sm text-destructive">{clipInfo.replay_error}</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
