"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import AiScreeningInspector from "@/components/AiScreeningInspector";
import { getScreeningSession } from "@/lib/api";
import type { InspectResult } from "@/components/VideoCard";

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
        // Parse FastAPI JSON error if present
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
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Models</h1>
        <p className="text-sm text-muted-foreground">Loading session...</p>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Models</h1>
        <p className="text-sm text-red-600">{error ?? "Session not found"}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Models</h1>
      <div className="flex gap-1 border-b">
        <span className="px-4 py-2 text-sm font-medium border-b-2 border-foreground text-foreground">
          Screening
        </span>
      </div>
      <AiScreeningInspector initialSession={session} />
    </div>
  );
}
