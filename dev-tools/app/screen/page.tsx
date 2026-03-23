"use client";

import { useState, useEffect } from "react";
import VideoScreenerPage from "@/components/VideoScreenerPage";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function ScreenPage() {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);

  useEffect(() => {
    listCrawlChannels().then(setChannels).catch(() => {});
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Screen</h2>
      <VideoScreenerPage channels={channels} />
    </div>
  );
}
