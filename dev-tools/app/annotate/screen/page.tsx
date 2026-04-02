"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import VideoScreenerPage from "@/components/annotate/VideoScreenerPage";
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

  return <VideoScreenerPage channels={channels} initialVideoIds={initialVideoIds} />;
}
