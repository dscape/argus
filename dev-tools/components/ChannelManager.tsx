"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  listCrawlChannels,
  addCrawlChannel,
  toggleCrawlChannel,
  crawlChannel,
  crawlAllChannels,
  getQuotaStatus,
} from "@/lib/api";
import type { CrawlChannel, QuotaStatus } from "@/lib/types";

interface ChannelManagerProps {
  onSelectChannel: (channelId: string) => void;
}

export default function ChannelManager({
  onSelectChannel,
}: ChannelManagerProps) {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newHandle, setNewHandle] = useState("");
  const [adding, setAdding] = useState(false);
  const [crawlingId, setCrawlingId] = useState<string | null>(null);
  const [crawlingAll, setCrawlingAll] = useState(false);
  const [quota, setQuota] = useState<QuotaStatus | null>(null);
  const [crawlResult, setCrawlResult] = useState<string | null>(null);

  const loadChannels = useCallback(async () => {
    try {
      const data = await listCrawlChannels();
      setChannels(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load channels");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadQuota = useCallback(async () => {
    try {
      setQuota(await getQuotaStatus());
    } catch {
      // Quota fetch is best-effort
    }
  }, []);

  useEffect(() => {
    loadChannels();
    loadQuota();
  }, [loadChannels, loadQuota]);

  const handleAdd = async () => {
    if (!newHandle.trim()) return;
    setAdding(true);
    setCrawlResult(null);
    try {
      await addCrawlChannel(newHandle.trim());
      setNewHandle("");
      await loadChannels();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add channel");
    } finally {
      setAdding(false);
    }
  };

  const handleToggle = async (channelId: string, enabled: boolean) => {
    try {
      await toggleCrawlChannel(channelId, enabled);
      setChannels((prev) =>
        prev.map((ch) =>
          ch.channel_id === channelId ? { ...ch, enabled } : ch
        )
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to toggle channel");
    }
  };

  const handleCrawl = async (channelId: string) => {
    setCrawlingId(channelId);
    setCrawlResult(null);
    try {
      const result = await crawlChannel(channelId);
      setCrawlResult(`Crawled: ${result.new_videos} new videos`);
      await loadChannels();
      await loadQuota();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Crawl failed");
    } finally {
      setCrawlingId(null);
    }
  };

  const handleCrawlAll = async () => {
    setCrawlingAll(true);
    setCrawlResult(null);
    try {
      const result = await crawlAllChannels();
      setCrawlResult(
        `Crawled ${result.channels_crawled} channels: ${result.total_new_videos} new videos`
      );
      await loadChannels();
      await loadQuota();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Crawl all failed");
    } finally {
      setCrawlingAll(false);
    }
  };

  const formatDate = (d: string | null) => {
    if (!d) return "Never";
    return new Date(d).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="space-y-4">
      {/* Header: Add channel + Crawl All + Quota */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 flex-1 min-w-[250px]">
          <input
            type="text"
            placeholder="@handle or channel handle"
            value={newHandle}
            onChange={(e) => setNewHandle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            className="flex-1 rounded-md border bg-background px-3 py-2 text-sm"
          />
          <Button onClick={handleAdd} disabled={adding || !newHandle.trim()} size="sm">
            {adding ? "Adding..." : "Add Channel"}
          </Button>
        </div>
        <Button
          onClick={handleCrawlAll}
          disabled={crawlingAll || crawlingId !== null}
          variant="outline"
          size="sm"
        >
          {crawlingAll ? "Crawling All..." : "Crawl All"}
        </Button>
        {quota && (
          <Badge variant="outline" className="text-xs">
            Quota: {quota.remaining.toLocaleString()} / {quota.daily_limit.toLocaleString()}
          </Badge>
        )}
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-2 underline text-xs"
          >
            dismiss
          </button>
        </div>
      )}

      {crawlResult && (
        <div className="rounded-md bg-primary/10 border border-primary/20 p-3 text-sm">
          {crawlResult}
        </div>
      )}

      {/* Channels table */}
      {loading ? (
        <div className="text-sm text-muted-foreground">Loading channels...</div>
      ) : channels.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No channels yet. Add one above or run{" "}
            <code className="text-xs bg-muted px-1 rounded">make seed-channels</code>.
          </CardContent>
        </Card>
      ) : (
        <div className="border rounded-md overflow-x-auto">
          <table className="w-full text-sm min-w-[700px]">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left px-3 py-2 font-medium">Channel</th>
                <th className="text-left px-3 py-2 font-medium w-16">Tier</th>
                <th className="text-right px-3 py-2 font-medium w-20">Videos</th>
                <th className="text-left px-3 py-2 font-medium w-40">Last Crawled</th>
                <th className="text-center px-3 py-2 font-medium w-20">Enabled</th>
                <th className="text-right px-3 py-2 font-medium w-24">Actions</th>
              </tr>
            </thead>
            <tbody>
              {channels.map((ch) => (
                <tr
                  key={ch.channel_id}
                  className="border-b last:border-b-0 hover:bg-muted/30"
                >
                  <td className="px-3 py-2">
                    <button
                      onClick={() => onSelectChannel(ch.channel_id)}
                      className="text-left hover:text-primary transition-colors"
                    >
                      <div className="font-medium">{ch.channel_name}</div>
                      {ch.channel_handle && (
                        <div className="text-xs text-muted-foreground">
                          {ch.channel_handle}
                        </div>
                      )}
                    </button>
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant="outline" className="text-xs">
                      T{ch.tier}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {ch.video_count.toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {formatDate(ch.last_crawled_at)}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <button
                      onClick={() => handleToggle(ch.channel_id, !ch.enabled)}
                      className={`w-8 h-5 rounded-full relative transition-colors ${
                        ch.enabled ? "bg-primary" : "bg-muted-foreground/30"
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                          ch.enabled ? "left-3.5" : "left-0.5"
                        }`}
                      />
                    </button>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleCrawl(ch.channel_id)}
                      disabled={
                        crawlingId !== null || crawlingAll || !ch.enabled
                      }
                      className="text-xs h-7"
                    >
                      {crawlingId === ch.channel_id ? "Crawling..." : "Crawl"}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
