"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const navItems = [
  {
    href: "/synthetic",
    label: "Synthetic",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <path d="M12 3l7 4v6.5a2 2 0 01-1 1.73L12 19l-6-3.77A2 2 0 015 13.5V7l7-4z" />
        <path d="M12 12L5 7.5" />
        <path d="M12 12l7-4.5" />
        <path d="M12 12v7.5" />
      </svg>
    ),
  },
  {
    href: "/crawl",
    label: "Crawl",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 9V4" />
        <path d="M12 15v5" />
        <path d="M9 12H4" />
        <path d="M15 12h5" />
        <path d="M9.5 9.5L6 6" />
        <path d="M14.5 14.5L18 18" />
        <path d="M9.5 14.5L6 18" />
        <path d="M14.5 9.5L18 6" />
      </svg>
    ),
  },
  {
    href: "/overlay-tester",
    label: "Overlay Tester",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <rect x="3" y="3" width="8" height="8" />
        <rect x="13" y="3" width="8" height="8" />
        <rect x="3" y="13" width="8" height="8" />
        <rect x="13" y="13" width="8" height="8" />
      </svg>
    ),
  },
  {
    href: "/calibration",
    label: "Calibration",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <circle cx="12" cy="12" r="8" />
        <line x1="12" y1="2" x2="12" y2="6" />
        <line x1="12" y1="18" x2="12" y2="22" />
        <line x1="2" y1="12" x2="6" y2="12" />
        <line x1="18" y1="12" x2="22" y2="12" />
        <circle cx="12" cy="12" r="2" />
      </svg>
    ),
  },
  {
    href: "/video-annotator",
    label: "Video Annotator",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <line x1="2" y1="8" x2="22" y2="8" />
        <line x1="8" y1="4" x2="8" y2="8" />
        <polygon points="10,12 10,18 16,15" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
];

export default function IconSidebar() {
  const pathname = usePathname();
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

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
