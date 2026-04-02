"use client";

import { useState, useEffect } from "react";
import VideoBrowser from "@/components/videos/VideoBrowser";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function BrowsePage() {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);

  useEffect(() => {
    listCrawlChannels({ screenedOnly: true }).then(setChannels).catch(() => {});
  }, []);

  return <VideoBrowser channels={channels} />;
}
