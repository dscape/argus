"use client";

import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
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

        {/* Tensors */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Tensors</h4>
          <div className="border rounded-md overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left px-3 py-1 font-medium">Name</th>
                  <th className="text-left px-3 py-1 font-medium">Shape</th>
                  <th className="text-left px-3 py-1 font-medium">Dtype</th>
                </tr>
              </thead>
              <tbody>
                {clipInfo.tensors.map((t) => (
                  <tr key={t.name} className="border-b last:border-0">
                    <td className="px-3 py-1 font-mono text-xs">{t.name}</td>
                    <td className="px-3 py-1 font-mono text-xs">
                      [{t.shape.join(", ")}]
                    </td>
                    <td className="px-3 py-1 text-xs text-muted-foreground">
                      {t.dtype}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Moves */}
        {clipInfo.moves.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-medium mb-2">
              Moves ({clipInfo.total_moves})
            </h4>
            <div className="flex flex-wrap gap-1">
              {clipInfo.moves.map((m, i) => (
                <span
                  key={i}
                  className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded"
                  title={`Frame ${m.frame_index}`}
                >
                  {m.san || m.uci}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Frames grid */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">
            Frames ({clipInfo.num_frames})
          </h4>
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {Array.from({ length: clipInfo.num_frames }, (_, i) => (
              <div key={i} className="relative">
                <img
                  src={clipFrameUrl(sessionId, i)}
                  alt={`Frame ${i}`}
                  className="w-full rounded border"
                />
                <span className="absolute bottom-0 right-0 text-[10px] bg-black/60 text-white px-1 rounded-tl">
                  {i}
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
