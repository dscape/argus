"use client";

import { useSearchParams } from "next/navigation";
import { useState, useEffect } from "react";
import VideoScreener from "@/components/VideoScreener";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function VideosPage() {
  const searchParams = useSearchParams();
  const initialChannelId = searchParams.get("channel");

  const [channels, setChannels] = useState<CrawlChannel[]>([]);

  useEffect(() => {
    listCrawlChannels().then(setChannels).catch(() => {});
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Videos</h2>
      <VideoScreener channels={channels} initialChannelId={initialChannelId} />
    </div>
  );
}
