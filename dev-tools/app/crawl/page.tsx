"use client";

import { useState, useEffect, useCallback } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ChannelManager from "@/components/ChannelManager";
import VideoInspector from "@/components/VideoInspector";
import AIClassifier from "@/components/AIClassifier";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function CrawlPage() {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);
  const [selectedChannelId, setSelectedChannelId] = useState<string | null>(
    null
  );
  const [activeTab, setActiveTab] = useState("channels");

  const loadChannels = useCallback(async () => {
    try {
      setChannels(await listCrawlChannels());
    } catch {
      // Will be shown by ChannelManager
    }
  }, []);

  useEffect(() => {
    loadChannels();
  }, [loadChannels]);

  const handleSelectChannel = (channelId: string) => {
    setSelectedChannelId(channelId);
    setActiveTab("videos");
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Crawl</h2>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="channels">Channels</TabsTrigger>
          <TabsTrigger value="videos">Videos</TabsTrigger>
          <TabsTrigger value="classify">AI Classify</TabsTrigger>
        </TabsList>

        <TabsContent value="channels">
          <ChannelManager onSelectChannel={handleSelectChannel} />
        </TabsContent>

        <TabsContent value="videos">
          <VideoInspector
            channels={channels}
            initialChannelId={selectedChannelId}
          />
        </TabsContent>

        <TabsContent value="classify">
          <AIClassifier />
        </TabsContent>
      </Tabs>
    </div>
  );
}
