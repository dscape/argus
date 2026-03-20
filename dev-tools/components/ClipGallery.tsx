"use client";

import { useEffect, useRef } from "react";
import type { ClipInspectResponse, SyntheticClipFile } from "@/lib/types";
import { deleteClipSession } from "@/lib/api";
import { ClipCard } from "@/components/ClipCard";

interface ClipGalleryProps {
  clips: SyntheticClipFile[];
  directory: string;
  onClipClick: (
    clip: SyntheticClipFile,
    sessionId: string,
    clipInfo: ClipInspectResponse
  ) => void;
  onClipInspected?: (clipInfo: ClipInspectResponse) => void;
}

export function ClipGallery({ clips, directory, onClipClick, onClipInspected }: ClipGalleryProps) {
  const cache = useRef<
    Map<string, { sessionId: string; clipInfo: ClipInspectResponse }>
  >(new Map());

  // Cleanup sessions on unmount
  useEffect(() => {
    const currentCache = cache.current;
    return () => {
      currentCache.forEach((entry) => {
        deleteClipSession(entry.sessionId).catch(() => {});
      });
    };
  }, []);

  if (clips.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-8 text-center">
        No .pt files found.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {clips.map((clip) => (
        <ClipCard
          key={clip.filename}
          clip={clip}
          directory={directory}
          cache={cache}
          onClick={(sessionId, clipInfo) =>
            onClipClick(clip, sessionId, clipInfo)
          }
          onInspected={onClipInspected}
        />
      ))}
    </div>
  );
}
