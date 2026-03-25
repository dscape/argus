"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import VideoScreenerPage from "@/components/VideoScreenerPage";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function ScreenPage() {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);
  const searchParams = useSearchParams();
  const videosParam = searchParams.get("videos");
  const initialVideoIds = videosParam ? videosParam.split(",") : undefined;

  useEffect(() => {
    listCrawlChannels().then(setChannels).catch(() => {});
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Screen</h2>
      <VideoScreenerPage channels={channels} initialVideoIds={initialVideoIds} />
    </div>
  );
}
