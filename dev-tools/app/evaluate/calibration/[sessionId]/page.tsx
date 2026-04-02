"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import CalibrationEvalInspector from "@/components/evaluate/CalibrationEvalInspector";
import { getCalibrationEvalSession } from "@/lib/api";

export default function CalibrationEvalSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<{
    id: string;
    results: any[];
    overlay_iou_avg: number | null;
    theme_accuracy: number | null;
    orientation_accuracy: number | null;
    grid_success_rate: number | null;
    fen_validity_rate: number | null;
    pin_state: Record<string, boolean>;
    created_at: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getCalibrationEvalSession(sessionId);
        setSession({
          id: data.id,
          results: data.results ?? [],
          overlay_iou_avg: data.overlay_iou_avg,
          theme_accuracy: data.theme_accuracy,
          orientation_accuracy: data.orientation_accuracy,
          grid_success_rate: data.grid_success_rate,
          fen_validity_rate: data.fen_validity_rate,
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

  return <CalibrationEvalInspector initialSession={session} />;
}
