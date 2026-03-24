"use client";

import { useState, useEffect } from "react";
import VideoBrowser from "@/components/VideoBrowser";
import { listCrawlChannels } from "@/lib/api";
import type { CrawlChannel } from "@/lib/types";

export default function VideosPage() {
  const [channels, setChannels] = useState<CrawlChannel[]>([]);

  useEffect(() => {
    listCrawlChannels({ screenedOnly: true }).then(setChannels).catch(() => {});
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Videos</h2>
      <VideoBrowser channels={channels} />
    </div>
  );
}
