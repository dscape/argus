"use client";

import { useEffect, useRef, useState } from "react";
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
  const knownFilenames = useRef<Set<string>>(new Set());
  const [newFilenames, setNewFilenames] = useState<Set<string>>(new Set());

  // Track new clips: any filename not previously known is "new"
  useEffect(() => {
    const currentNames = new Set(clips.map((c) => c.filename));
    const fresh = new Set<string>();
    for (const name of currentNames) {
      if (!knownFilenames.current.has(name)) {
        fresh.add(name);
      }
    }
    knownFilenames.current = currentNames;

    if (fresh.size > 0) {
      setNewFilenames(fresh);
      // Clear the "new" state after the animation completes
      const timer = setTimeout(() => setNewFilenames(new Set()), 600);
      return () => clearTimeout(timer);
    }
  }, [clips]);

  // Cleanup sessions on unmount
  useEffect(() => {
    const currentCache = cache.current;
    return () => {
      currentCache.forEach((entry) => {
        deleteClipSession(entry.sessionId).catch(() => {});
      });
    };
  }, []);

  // Clean up orphaned sessions when clips are removed from the list
  useEffect(() => {
    const currentFilenames = new Set(clips.map((c) => c.filename));
    const currentCache = cache.current;
    for (const [filename, entry] of currentCache) {
      if (!currentFilenames.has(filename)) {
        deleteClipSession(entry.sessionId).catch(() => {});
        currentCache.delete(filename);
      }
    }
  }, [clips]);

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
          isNew={newFilenames.has(clip.filename)}
          onClick={(sessionId, clipInfo) =>
            onClipClick(clip, sessionId, clipInfo)
          }
          onInspected={onClipInspected}
        />
      ))}
    </div>
  );
}
