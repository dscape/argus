"use client";

import { NavTabs } from "@/components/ui/nav-tabs";

const TABS = [
  { id: "screen", label: "Screen", href: "/annotate/screen" },
  { id: "bbox", label: "BBox", href: "/annotate/bbox" },
  { id: "physical", label: "Physical eval", href: "/annotate/physical" },
  { id: "physical-train", label: "Physical train", href: "/annotate/physical-train" },
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
