"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import AiScreeningInspector from "@/components/evaluate/AiScreeningInspector";
import { getScreeningSession } from "@/lib/api";
import type { InspectResult } from "@/components/evaluate/VideoCard";

export default function ScreeningSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: InspectResult[];
    accuracy: number | null;
    per_class: Record<string, { correct: number; total: number }> | null;
    pin_state: Record<string, boolean>;
    model_version: string | null;
    created_at: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getScreeningSession(sessionId);
        setSession({
          id: data.id,
          results: (data.results ?? []) as InspectResult[],
          accuracy: data.accuracy,
          per_class: data.per_class ?? null,
          pin_state: data.pin_state ?? {},
          model_version: data.model_version,
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

  return <AiScreeningInspector initialSession={session} />;
}
