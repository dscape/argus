"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import FenInspector from "@/components/evaluate/FenInspector";
import { getOverlayTestSession, type OverlayTestResult } from "@/lib/api";

export default function FenSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: OverlayTestResult[];
    accuracy: number | null;
    piece_accuracy: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getOverlayTestSession(sessionId);
        setSession({
          id: data.id,
          results: (data.results ?? []) as OverlayTestResult[],
          accuracy: data.accuracy,
          piece_accuracy: data.piece_accuracy,
          pin_state: data.pin_state ?? {},
          created_at: data.created_at,
        });
      } catch (e) {
        const raw = e instanceof Error ? e.message : "Failed to load session";
        try {
          const parsed = JSON.parse(raw);
          toast.error(parsed.detail ?? raw);
        } catch {
          toast.error(raw);
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

  if (!session) {
    return <p className="text-sm text-muted-foreground">Session not found</p>;
  }

  return <FenInspector initialSession={session} />;
}
