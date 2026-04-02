"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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

      {isTabRoute && (
        <div className="flex gap-1 border-b">
          {TABS.map((tab) => {
            const active = pathname.startsWith(tab.href);
            return (
              <Link
                key={tab.id}
                href={tab.href}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  active
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab.label}
              </Link>
            );
          })}
        </div>
      )}

      <div>{children}</div>
    </div>
  );
}
