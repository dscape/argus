"use client";

import { NavTabs } from "@/components/ui/nav-tabs";

const TABS = [
  { id: "screen", label: "Screen", href: "/annotate/screen" },
  { id: "bbox", label: "BBox", href: "/annotate/bbox" },
] as const;

export default function AnnotateLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Annotate</h1>
      <NavTabs tabs={TABS} />
      <div>{children}</div>
    </div>
  );
}
