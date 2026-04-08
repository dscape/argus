"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { id: "screening", label: "Screening", href: "/evaluate/screening" },
  { id: "overlay", label: "Overlay", href: "/evaluate/overlay" },
  { id: "fen", label: "FEN", href: "/evaluate/fen" },
  { id: "segmentation", label: "Segmentation", href: "/evaluate/segmentation" },
  { id: "calibration", label: "Calibration", href: "/evaluate/calibration" },
] as const;

export default function EvaluateLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Evaluate</h1>

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

      <div>{children}</div>
    </div>
  );
}
