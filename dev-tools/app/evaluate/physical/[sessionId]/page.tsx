"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import PhysicalRuntimeInspector from "@/components/evaluate/PhysicalRuntimeInspector";
import {
  getPhysicalRuntimeSession,
  type PhysicalRuntimeEvalResult,
} from "@/lib/api";

export default function PhysicalRuntimeSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: PhysicalRuntimeEvalResult[];
    square_accuracy: number | null;
    non_empty_accuracy: number | null;
    exact_match_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
    model_label?: string | null;
    model_path?: string | null;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getPhysicalRuntimeSession(sessionId);
        setSession({
          id: data.id,
          results: (data.results ?? []) as PhysicalRuntimeEvalResult[],
          square_accuracy: data.square_accuracy,
          non_empty_accuracy: data.non_empty_accuracy,
          exact_match_rate: data.exact_match_rate,
          pin_state: data.pin_state ?? {},
          created_at: data.created_at,
          model_label: data.model_label ?? null,
          model_path: data.model_path ?? null,
        });
      } catch (error) {
        const raw = error instanceof Error ? error.message : "Failed to load session";
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

  return <PhysicalRuntimeInspector initialSession={session} />;
}
