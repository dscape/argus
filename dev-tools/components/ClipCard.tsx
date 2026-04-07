"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { clipFrameUrl, getClipInfo, inspectSyntheticClip } from "@/lib/api";
import type { ClipInspectResponse, SyntheticClipFile } from "@/lib/types";

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

export function ClipCard({
  clip,
  directory,
  cache,
  isNew,
  onClick,
  onInspected,
}: ClipCardProps) {
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

    const cached = cache.current.get(clip.filename);
    if (cached) {
      setSessionId(cached.sessionId);
      setClipInfo(cached.clipInfo);
      computeFrameIndices(cached.clipInfo.num_frames);
      preloadFrames(cached.sessionId);
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
      preloadFrames(session_id);
      onInspected?.(info);
    } catch {
      inspectedRef.current = false;
    } finally {
      setLoading(false);
    }
  }, [cache, clip.filename, directory, loading, onInspected]);

  function computeFrameIndices(numFrames: number) {
    if (numFrames <= 1) {
      frameIndicesRef.current = [0];
      return;
    }

    const count = Math.min(6, numFrames);
    frameIndicesRef.current = Array.from({ length: count }, (_, index) =>
      Math.floor((index * (numFrames - 1)) / (count - 1))
    );
  }

  function preloadFrames(clipSessionId: string) {
    const indices = frameIndicesRef.current;
    if (indices.length <= 1) {
      setFramesReady(true);
      return;
    }

    let loaded = 0;

    for (const index of indices) {
      const image = new Image();
      image.onload = () => {
        loaded += 1;
        if (loaded >= indices.length) setFramesReady(true);
      };
      image.onerror = () => {
        loaded += 1;
        if (loaded >= indices.length) setFramesReady(true);
      };
      image.src = clipFrameUrl(clipSessionId, index);
    }
  }

  useEffect(() => {
    const element = cardRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          inspect();
          observer.disconnect();
        }
      },
      { rootMargin: "200px" }
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [inspect]);

  const handleMouseEnter = () => {
    if (!framesReady || !sessionId || frameIndicesRef.current.length <= 1) return;

    let frameIndex = 0;
    animIntervalRef.current = setInterval(() => {
      frameIndex = (frameIndex + 1) % frameIndicesRef.current.length;
      setCurrentFrame(frameIndicesRef.current[frameIndex] ?? 0);
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
          .map((move, index) => {
            const moveNumber = Math.floor(index / 2) + 1;
            const isWhite = index % 2 === 0;
            const san = move.san || move.uci;
            return isWhite ? `${moveNumber}.${san}` : san;
          })
          .join(" ")
      : null;

  return (
    <div
      ref={cardRef}
      className={`group cursor-pointer overflow-hidden rounded-lg border bg-card transition-colors hover:border-primary/50 ${
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
            className="h-full w-full object-cover"
          />
        ) : (
          <Skeleton className="h-full w-full rounded-none" />
        )}
        {clipInfo && !clipInfo.replay_valid && (
          <Badge
            variant="destructive"
            className="absolute right-1.5 top-1.5 px-1.5 py-0 text-[10px]"
          >
            invalid
          </Badge>
        )}
      </div>
      <div className="space-y-1 p-2">
        {movesText ? (
          <p className="truncate text-xs font-mono text-foreground" title={movesText}>
            {movesText}
          </p>
        ) : clipInfo ? (
          <p className="text-xs text-muted-foreground">No moves</p>
        ) : (
          <Skeleton className="h-3 w-3/4" />
        )}
        <p className="truncate text-[11px] text-muted-foreground">
          {clip.filename} &middot; {clip.size_mb} MB
        </p>
      </div>
    </div>
  );
}
