"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import SegmentationEvalInspector from "@/components/SegmentationEvalInspector";
import { getSegmentationEvalSession } from "@/lib/api";

export default function SegmentationSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: any[];
    segment_consistency: number | null;
    gap_consistency: number | null;
    piece_readability: number | null;
    false_negative_rate: number | null;
    coverage_ratio: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getSegmentationEvalSession(sessionId);
        setSession({
          id: data.id,
          results: data.results ?? [],
          segment_consistency: data.segment_consistency,
          gap_consistency: data.gap_consistency,
          piece_readability: data.piece_readability,
          false_negative_rate: data.false_negative_rate,
          coverage_ratio: data.coverage_ratio,
          pin_state: data.pin_state ?? {},
          created_at: data.created_at,
        });
      } catch (e) {
        const raw = e instanceof Error ? e.message : "Failed to load session";
        try {
          const parsed = JSON.parse(raw);
          setError(parsed.detail ?? raw);
        } catch {
          setError(raw);
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [sessionId]);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading session...</p>;
  }

  if (error || !session) {
    return <p className="text-sm text-red-600">{error ?? "Session not found"}</p>;
  }

  return <SegmentationEvalInspector initialSession={session} />;
}
