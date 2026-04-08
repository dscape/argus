"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import OverlayEvalInspector from "@/components/evaluate/OverlayEvalInspector";
import { getOverlayEvalSession, type OverlayEvalResult } from "@/lib/api";

export default function OverlayEvalSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: OverlayEvalResult[];
    detection_rate: number | null;
    fen_success_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getOverlayEvalSession(sessionId);
        setSession({
          id: data.id,
          results: (data.results ?? []) as OverlayEvalResult[],
          detection_rate: data.detection_rate,
          fen_success_rate: data.fen_success_rate,
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

  return <OverlayEvalInspector initialSession={session} />;
}
