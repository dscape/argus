"use client";

import { NavTabs } from "@/components/ui/nav-tabs";

const TABS = [
  { id: "real", label: "Real footage", href: "/data/real" },
  { id: "synthetic", label: "Synthetic", href: "/data/synthetic" },
] as const;

export default function DataLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-bold">Data</h2>
          <p className="text-sm text-muted-foreground">
            Inspect clip datasets and unblock real-footage processing.
          </p>
        </div>
      </div>
      <NavTabs tabs={TABS} />
      <div>{children}</div>
    </div>
  );
}
