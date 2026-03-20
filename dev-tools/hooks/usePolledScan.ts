"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { scanSyntheticDir } from "@/lib/api";
import type { SyntheticScanResponse, SyntheticClipFile } from "@/lib/types";

interface UsePolledScanOptions {
  directory: string;
  intervalMs?: number;
  enabled?: boolean;
  onClipCountChange?: (count: number) => void;
}

function hasChanged(
  prev: SyntheticScanResponse | null,
  next: SyntheticScanResponse
): boolean {
  if (!prev) return true;
  if (prev.clip_count !== next.clip_count) return true;
  for (let i = 0; i < next.clips.length; i++) {
    if (prev.clips[i]?.filename !== next.clips[i]?.filename) return true;
  }
  return false;
}

/** Sort clips newest-first by modified timestamp. */
function sortNewestFirst(clips: SyntheticClipFile[]): SyntheticClipFile[] {
  return [...clips].sort(
    (a, b) => new Date(b.modified).getTime() - new Date(a.modified).getTime()
  );
}

export function usePolledScan({
  directory,
  intervalMs = 2000,
  enabled = true,
  onClipCountChange,
}: UsePolledScanOptions) {
  const [scan, setScan] = useState<SyntheticScanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(enabled);
  const scanRef = useRef<SyntheticScanResponse | null>(null);
  const onClipCountChangeRef = useRef(onClipCountChange);
  onClipCountChangeRef.current = onClipCountChange;

  const doFetch = useCallback(() => {
    scanSyntheticDir(directory)
      .then((result) => {
        if (hasChanged(scanRef.current, result)) {
          const prevCount = scanRef.current?.clip_count ?? 0;
          const sorted: SyntheticScanResponse = {
            ...result,
            clips: sortNewestFirst(result.clips),
          };
          scanRef.current = sorted;
          setScan(sorted);
          if (result.clip_count !== prevCount) {
            onClipCountChangeRef.current?.(result.clip_count);
          }
        }
        setError(null);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Scan failed");
      });
  }, [directory]);

  // Initial fetch
  useEffect(() => {
    doFetch();
  }, [doFetch]);

  // Polling interval
  useEffect(() => {
    if (!isPolling) return;
    const id = setInterval(doFetch, intervalMs);
    return () => clearInterval(id);
  }, [doFetch, intervalMs, isPolling]);

  return { scan, error, isPolling, setPolling: setIsPolling };
}
