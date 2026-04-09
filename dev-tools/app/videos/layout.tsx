"use client";

import { usePathname } from "next/navigation";
import { NavTabs } from "@/components/ui/nav-tabs";

const TABS = [
  { id: "channels", label: "Channels", href: "/videos/channels" },
  { id: "browse", label: "Browse", href: "/videos/browse" },
] as const;

export default function VideosLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isTabRoute = TABS.some(tab => pathname.startsWith(tab.href));

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Videos</h1>
      {isTabRoute && <NavTabs tabs={TABS} />}
      <div>{children}</div>
    </div>
  );
}
