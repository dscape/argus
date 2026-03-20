"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  inspectSyntheticClip,
  getClipInfo,
  clipFrameUrl,
} from "@/lib/api";
import type { ClipInspectResponse, SyntheticClipFile } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface ClipCardProps {
  clip: SyntheticClipFile;
  directory: string;
  cache: React.MutableRefObject<
    Map<string, { sessionId: string; clipInfo: ClipInspectResponse }>
  >;
  isNew?: boolean;
  onClick: (sessionId: string, clipInfo: ClipInspectResponse) => void;
  onInspected?: (clipInfo: ClipInspectResponse) => void;
}

export function ClipCard({ clip, directory, cache, isNew, onClick, onInspected }: ClipCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clipInfo, setClipInfo] = useState<ClipInspectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [framesReady, setFramesReady] = useState(false);
  const animIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameIndicesRef = useRef<number[]>([]);
  const inspectedRef = useRef(false);

  const inspect = useCallback(async () => {
    if (inspectedRef.current || loading) return;
    inspectedRef.current = true;

    // Check cache first
    const cached = cache.current.get(clip.filename);
    if (cached) {
      setSessionId(cached.sessionId);
      setClipInfo(cached.clipInfo);
      computeFrameIndices(cached.clipInfo.num_frames);
      preloadFrames(cached.sessionId, cached.clipInfo.num_frames);
      onInspected?.(cached.clipInfo);
      return;
    }

    setLoading(true);
    try {
      const filepath = `${directory}/${clip.filename}`;
      const { session_id } = await inspectSyntheticClip(filepath);
      const info = await getClipInfo(session_id);
      setSessionId(session_id);
      setClipInfo(info);
      cache.current.set(clip.filename, { sessionId: session_id, clipInfo: info });
      computeFrameIndices(info.num_frames);
      preloadFrames(session_id, info.num_frames);
      onInspected?.(info);
    } catch {
      // Silently fail for thumbnail loading
      inspectedRef.current = false;
    }
    setLoading(false);
  }, [clip.filename, directory, cache, loading, onInspected]);

  function computeFrameIndices(numFrames: number) {
    if (numFrames <= 1) {
      frameIndicesRef.current = [0];
      return;
    }
    const count = Math.min(6, numFrames);
    frameIndicesRef.current = Array.from({ length: count }, (_, i) =>
      Math.floor((i * (numFrames - 1)) / (count - 1))
    );
  }

  function preloadFrames(sid: string, numFrames: number) {
    const indices = frameIndicesRef.current;
    if (indices.length <= 1) {
      setFramesReady(true);
      return;
    }
    let loaded = 0;
    for (const idx of indices) {
      const img = new Image();
      img.onload = () => {
        loaded++;
        if (loaded >= indices.length) setFramesReady(true);
      };
      img.onerror = () => {
        loaded++;
        if (loaded >= indices.length) setFramesReady(true);
      };
      img.src = clipFrameUrl(sid, idx);
    }
  }

  // Lazy load via IntersectionObserver
  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          inspect();
          observer.disconnect();
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [inspect]);

  const handleMouseEnter = () => {
    if (!framesReady || !sessionId || frameIndicesRef.current.length <= 1) return;
    let frameIdx = 0;
    animIntervalRef.current = setInterval(() => {
      frameIdx = (frameIdx + 1) % frameIndicesRef.current.length;
      setCurrentFrame(frameIndicesRef.current[frameIdx]);
    }, 400);
  };

  const handleMouseLeave = () => {
    if (animIntervalRef.current) {
      clearInterval(animIntervalRef.current);
      animIntervalRef.current = null;
    }
    setCurrentFrame(0);
  };

  useEffect(() => {
    return () => {
      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
    };
  }, []);

  const movesText =
    clipInfo && clipInfo.moves.length > 0
      ? clipInfo.moves
          .map((m, i) => {
            const moveNum = Math.floor(i / 2) + 1;
            const isWhite = i % 2 === 0;
            const san = m.san || m.uci;
            return isWhite ? `${moveNum}.${san}` : san;
          })
          .join(" ")
      : null;

  return (
    <div
      ref={cardRef}
      className={`group cursor-pointer rounded-lg border bg-card overflow-hidden hover:border-primary/50 transition-colors ${
        isNew ? "animate-clip-appear" : ""
      }`}
      onClick={() => {
        if (sessionId && clipInfo) onClick(sessionId, clipInfo);
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="relative aspect-square bg-muted">
        {sessionId ? (
          <img
            src={clipFrameUrl(sessionId, currentFrame)}
            alt={clip.filename}
            className="w-full h-full object-cover"
          />
        ) : (
          <Skeleton className="w-full h-full rounded-none" />
        )}
        {clipInfo && !clipInfo.replay_valid && (
          <Badge
            variant="destructive"
            className="absolute top-1.5 right-1.5 text-[10px] px-1.5 py-0"
          >
            invalid
          </Badge>
        )}
      </div>
      <div className="p-2 space-y-1">
        {movesText ? (
          <p className="text-xs font-mono truncate text-foreground" title={movesText}>
            {movesText}
          </p>
        ) : clipInfo ? (
          <p className="text-xs text-muted-foreground">No moves</p>
        ) : (
          <Skeleton className="h-3 w-3/4" />
        )}
        <p className="text-[11px] text-muted-foreground truncate">
          {clip.filename} &middot; {clip.size_mb} MB
        </p>
      </div>
    </div>
  );
}
