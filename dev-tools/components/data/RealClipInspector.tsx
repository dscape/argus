"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { clipFrameUrl } from "@/lib/api";
import type { ClipInspectResponse } from "@/lib/types";

interface RealClipInspectorProps {
  open: boolean;
  onClose: () => void;
  filename: string;
  sessionId: string;
  clipInfo: ClipInspectResponse;
}

export function RealClipInspector({
  open,
  onClose,
  filename,
  sessionId,
  clipInfo,
}: RealClipInspectorProps) {
  useEffect(() => {
    if (open && clipInfo.replay_error) toast.error(clipInfo.replay_error);
  }, [open, clipInfo.replay_error]);

  const metadata = clipInfo.metadata ?? {};

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent fullscreen>
        <DialogHeader>
          <div className="flex items-center gap-3 flex-wrap">
            <DialogTitle>{filename}</DialogTitle>
            <Badge variant="outline">{clipInfo.file_size_mb} MB</Badge>
            <Badge variant={clipInfo.replay_valid ? "default" : "destructive"}>
              {clipInfo.replay_valid ? "valid" : "invalid"}
            </Badge>
          </div>
        </DialogHeader>

        <div className="grid gap-4 lg:grid-cols-2 mb-4">
          <MetadataCard title="Real-footage metadata">
            <MetadataRow label="Source video" value={metadata.source_video_id} />
            <MetadataRow label="Channel" value={metadata.source_channel_handle} />
            <MetadataRow label="Initial FEN" value={metadata.initial_board_fen} mono />
            <MetadataRow label="PGN moves" value={metadata.pgn_moves} mono multiline />
            <MetadataRow
              label="Segment start"
              value={formatSeconds(metadata.segment_start_time_seconds)}
            />
            <MetadataRow
              label="Segment end"
              value={formatSeconds(metadata.segment_end_time_seconds)}
            />
            <MetadataRow label="Sampled fps" value={formatNumber(metadata.sampled_video_fps)} />
            <MetadataRow label="Stored num moves" value={formatNumber(metadata.num_moves)} />
            <MetadataRow label="Replay final FEN" value={clipInfo.final_fen} mono />
          </MetadataCard>

          <MetadataCard title="Inspection summary">
            <MetadataRow label="Frames" value={String(clipInfo.num_frames)} />
            <MetadataRow label="Detected moves" value={String(clipInfo.total_moves)} />
            <MetadataRow label="No-move frames" value={String(clipInfo.no_move_frames)} />
            <MetadataRow label="Unknown frames" value={String(clipInfo.unknown_frames)} />
            <MetadataRow
              label="Average legal moves"
              value={formatNumber(clipInfo.avg_legal_moves)}
            />
            <MetadataRow
              label="Pixel range"
              value={`${clipInfo.pixel_range[0]} - ${clipInfo.pixel_range[1]}`}
            />
          </MetadataCard>
        </div>

        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Tensor payload</h4>
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
            <h4 className="text-sm font-medium mb-2">Moves ({clipInfo.total_moves})</h4>
            <div className="flex flex-wrap gap-1">
              {clipInfo.moves.map((move, index) => (
                <span
                  key={index}
                  className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded"
                  title={`Frame ${move.frame_index}`}
                >
                  {move.san || move.uci}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Frames ({clipInfo.num_frames})</h4>
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {Array.from({ length: clipInfo.num_frames }, (_, index) => (
              <div key={index} className="relative">
                <img
                  src={clipFrameUrl(sessionId, index)}
                  alt={`Frame ${index}`}
                  className="w-full rounded border"
                />
                <span className="absolute bottom-0 right-0 text-[10px] bg-black/60 text-white px-1 rounded-tl">
                  {index}
                </span>
              </div>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function MetadataCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border rounded-md p-3 space-y-2">
      <h4 className="text-sm font-medium">{title}</h4>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function MetadataRow({
  label,
  value,
  mono = false,
  multiline = false,
}: {
  label: string;
  value: string | number | boolean | null | undefined;
  mono?: boolean;
  multiline?: boolean;
}) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={[
          mono ? "font-mono text-xs" : "",
          multiline ? "whitespace-pre-wrap break-words" : "truncate",
        ].join(" ")}
        title={typeof value === "string" ? value : undefined}
      >
        {String(value)}
      </span>
    </div>
  );
}

function formatNumber(value: string | number | boolean | null | undefined): string | null {
  if (typeof value !== "number") return null;
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function formatSeconds(value: string | number | boolean | null | undefined): string | null {
  if (typeof value !== "number") return null;
  return `${value.toFixed(1)}s`;
}
