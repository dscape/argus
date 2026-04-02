"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { getVideoCounts } from "@/lib/api";

const navItems = [
  {
    href: "/videos",
    label: "Videos",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="w-5 h-5"
      >
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <line x1="2" y1="8" x2="22" y2="8" />
        <line x1="8" y1="4" x2="8" y2="8" />
        <polygon points="10,12 10,18 16,15" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
  {
    href: "/annotate",
    label: "Annotate",
    hasBadge: true,
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="w-5 h-5"
      >
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
      </svg>
    ),
  },
  {
    href: "/data",
    label: "Data",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="w-5 h-5"
      >
        <path d="M12 3l7 4v6.5a2 2 0 01-1 1.73L12 19l-6-3.77A2 2 0 015 13.5V7l7-4z" />
        <path d="M12 12L5 7.5" />
        <path d="M12 12l7-4.5" />
        <path d="M12 12v7.5" />
      </svg>
    ),
  },
  {
    href: "/evaluate",
    label: "Evaluate",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="w-5 h-5"
      >
        <path d="M9.5 2A2.5 2.5 0 0112 4.5V6a2 2 0 002 2h1.5A2.5 2.5 0 0118 10.5v0a2.5 2.5 0 01-2.5 2.5H14a2 2 0 00-2 2v1.5a2.5 2.5 0 01-5 0V15a2 2 0 00-2-2H3.5A2.5 2.5 0 011 10.5v0A2.5 2.5 0 013.5 8H5a2 2 0 002-2V4.5A2.5 2.5 0 019.5 2z" />
        <circle cx="17.5" cy="17.5" r="3.5" />
        <path d="M22 22l-1.5-1.5" />
      </svg>
    ),
  },
];

export default function IconSidebar() {
  const pathname = usePathname();
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [unscreenedCount, setUnscreenedCount] = useState(0);

  useEffect(() => {
    const fetchCount = () => {
      getVideoCounts()
        .then((counts) => setUnscreenedCount(counts.unscreened ?? 0))
        .catch(() => {});
    };
    fetchCount();
    const interval = setInterval(fetchCount, 30_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="flex flex-col items-center py-4 px-1.5">
      <nav className="flex flex-col items-center gap-1 rounded-2xl border bg-muted/30 p-1.5">
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <div key={item.href} className="relative">
              <Link
                href={item.href}
                onMouseEnter={() => setHoveredItem(item.href)}
                onMouseLeave={() => setHoveredItem(null)}
                className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-150 ${
                  isActive
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground/50 hover:text-foreground hover:bg-background/60"
                }`}
              >
                {item.icon}
              </Link>
              {/* Red notification badge for unscreened count */}
              {"hasBadge" in item && item.hasBadge && unscreenedCount > 0 && (
                <span className="absolute -top-1 -right-1 min-w-[16px] h-4 px-1 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center pointer-events-none">
                  {unscreenedCount > 99 ? "99+" : unscreenedCount}
                </span>
              )}
              {hoveredItem === item.href && (
                <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-md bg-foreground text-background text-xs font-medium whitespace-nowrap z-50 pointer-events-none">
                  {item.label}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
