"use client";

import { useEffect, useRef, useState } from "react";
import { ClipCard } from "@/components/ClipCard";
import { deleteClipSession } from "@/lib/api";
import type { ClipInspectResponse, SyntheticClipFile } from "@/lib/types";

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

export function ClipGallery({
  clips,
  directory,
  onClipClick,
  onClipInspected,
}: ClipGalleryProps) {
  const cache = useRef<Map<string, { sessionId: string; clipInfo: ClipInspectResponse }>>(
    new Map()
  );
  const knownFilenames = useRef<Set<string>>(new Set());
  const [newFilenames, setNewFilenames] = useState<Set<string>>(new Set());

  useEffect(() => {
    const currentNames = new Set(clips.map((clip) => clip.filename));
    const fresh = new Set<string>();

    for (const name of currentNames) {
      if (!knownFilenames.current.has(name)) {
        fresh.add(name);
      }
    }

    knownFilenames.current = currentNames;

    if (fresh.size > 0) {
      setNewFilenames(fresh);
      const timer = setTimeout(() => setNewFilenames(new Set()), 600);
      return () => clearTimeout(timer);
    }
  }, [clips]);

  useEffect(() => {
    const currentCache = cache.current;
    return () => {
      currentCache.forEach((entry) => {
        deleteClipSession(entry.sessionId).catch(() => {});
      });
    };
  }, []);

  useEffect(() => {
    const currentFilenames = new Set(clips.map((clip) => clip.filename));
    const currentCache = cache.current;

    for (const [filename, entry] of currentCache) {
      if (!currentFilenames.has(filename)) {
        deleteClipSession(entry.sessionId).catch(() => {});
        currentCache.delete(filename);
      }
    }
  }, [clips]);

  if (clips.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No .pt files found.</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {clips.map((clip) => (
        <ClipCard
          key={clip.filename}
          clip={clip}
          directory={directory}
          cache={cache}
          isNew={newFilenames.has(clip.filename)}
          onClick={(sessionId, clipInfo) => onClipClick(clip, sessionId, clipInfo)}
          onInspected={onClipInspected}
        />
      ))}
    </div>
  );
}
