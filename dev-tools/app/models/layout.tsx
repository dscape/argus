"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { id: "screening", label: "Screening", href: "/models/screening" },
  { id: "overlay", label: "Overlay", href: "/models/overlay" },
  { id: "calibration", label: "Auto-Calibration", href: "/models/calibration" },
  { id: "hardcuts", label: "Hard Cuts", href: "/models/hardcuts" },
] as const;

export default function ModelsLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Models</h1>

      {/* Tab bar */}
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

      {/* Tab content */}
      <div>{children}</div>
    </div>
  );
}
