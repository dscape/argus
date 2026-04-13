"use client";

import { NavTabs } from "@/components/ui/nav-tabs";

const TABS = [
  { id: "screening", label: "Screening", href: "/evaluate/screening" },
  { id: "overlay", label: "Overlay", href: "/evaluate/overlay" },
  { id: "fen", label: "FEN", href: "/evaluate/fen" },
  { id: "physical", label: "Physical", href: "/evaluate/physical" },
  { id: "segmentation", label: "Segmentation", href: "/evaluate/segmentation" },
  { id: "calibration", label: "Calibration", href: "/evaluate/calibration" },
] as const;

export default function EvaluateLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Evaluate</h1>
      <NavTabs tabs={TABS} />
      <div>{children}</div>
    </div>
  );
}
