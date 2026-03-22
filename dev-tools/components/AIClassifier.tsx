"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  listCrawlVideos,
  classifyTitles,
  autoClassifyTitles,
  getVideoCounts,
} from "@/lib/api";

export default function AIClassifier() {
  const [approvedCount, setApprovedCount] = useState<number | null>(null);
  const [unscreenedCount, setUnscreenedCount] = useState<number | null>(null);
  const [approvedIds, setApprovedIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [autoClassifying, setAutoClassifying] = useState(false);
  const [generatedPrompt, setGeneratedPrompt] = useState<string | null>(null);
  const [autoResult, setAutoResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const loadApproved = useCallback(async () => {
    try {
      const data = await listCrawlVideos({
        status: "approved",
        limit: 200,
        offset: 0,
      });
      setApprovedCount(data.total);
      setApprovedIds(data.videos.map((v) => v.video_id));
    } catch {
      setApprovedCount(0);
      setApprovedIds([]);
    }
  }, []);

  const loadCounts = useCallback(async () => {
    try {
      const counts = await getVideoCounts();
      setUnscreenedCount(counts.unscreened ?? 0);
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    loadApproved();
    loadCounts();
  }, [loadApproved, loadCounts]);

  const handleClassify = async () => {
    if (approvedIds.length === 0) return;
    setLoading(true);
    setError(null);
    setGeneratedPrompt(null);
    try {
      const result = await classifyTitles(approvedIds);
      setGeneratedPrompt(result.prompt);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Classification failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAutoClassify = async () => {
    setAutoClassifying(true);
    setError(null);
    setAutoResult(null);
    try {
      const result = await autoClassifyTitles();
      setAutoResult(
        `Classified ${result.classified} videos: ${result.candidates} candidates, ${result.rejected} rejected`
      );
      loadCounts();
      loadApproved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-classification failed");
    } finally {
      setAutoClassifying(false);
    }
  };

  const handleCopy = async () => {
    if (!generatedPrompt) return;
    try {
      await navigator.clipboard.writeText(generatedPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: select the textarea content
    }
  };

  return (
    <div className="space-y-4 max-w-3xl">
      {/* Auto-Classify Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Auto-Classify Videos</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Use Claude to automatically classify unscreened video titles as
            candidates or rejected, based on your approved examples. This
            replaces manual review for most videos.
          </p>

          <div className="flex items-center gap-3">
            <div className="text-sm space-x-4">
              <span>
                <span className="text-muted-foreground">Approved examples: </span>
                <span className="font-medium tabular-nums">
                  {approvedCount === null ? "..." : approvedCount}
                </span>
              </span>
              <span>
                <span className="text-muted-foreground">Unscreened: </span>
                <span className="font-medium tabular-nums">
                  {unscreenedCount === null ? "..." : unscreenedCount}
                </span>
              </span>
            </div>
            <Button
              onClick={handleAutoClassify}
              disabled={
                autoClassifying ||
                approvedCount === null ||
                approvedCount === 0 ||
                unscreenedCount === 0
              }
            >
              {autoClassifying ? "Classifying..." : "Auto-Classify Unscreened"}
            </Button>
          </div>

          {approvedCount === 0 && (
            <p className="text-sm text-muted-foreground">
              No approved videos yet. Go to the Videos tab and approve some
              videos first to use as examples.
            </p>
          )}

          {autoResult && (
            <div className="rounded-md bg-primary/10 border border-primary/20 p-3 text-sm">
              {autoResult}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Generate Classification Prompt Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Generate Classification Rules</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Analyze all manually approved videos and generate classification
            rules that can help automatically identify training-worthy titles
            in the future. Uses Claude to find patterns in your positive and
            negative examples.
          </p>

          <div className="flex items-center gap-3">
            <Button
              onClick={handleClassify}
              disabled={
                loading || approvedCount === null || approvedCount === 0
              }
              variant="outline"
            >
              {loading
                ? "Generating..."
                : "Generate Classification Prompt"}
            </Button>
          </div>

          {error && (
            <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
              {error}
            </div>
          )}

          {generatedPrompt && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  Generated Classification Prompt
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={handleCopy}
                >
                  {copied ? "Copied!" : "Copy"}
                </Button>
              </div>
              <pre className="rounded-md bg-muted p-4 text-sm overflow-auto max-h-[500px] whitespace-pre-wrap font-mono">
                {generatedPrompt}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
