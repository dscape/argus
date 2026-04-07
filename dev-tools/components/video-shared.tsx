"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import type { CrawlVideo, AiScreenResult } from "@/lib/types";

// ── Types ───────────────────────────────────────────────────

export interface Toast {
  id: string;
  message: string;
  type: "success" | "warning" | "error";
  undoAction?: () => Promise<void>;
}

export interface VideoWithReason extends CrawlVideo {
  inspect_reason?: string;
  ai_result?: AiScreenResult;
}

// ── Icons ───────────────────────────────────────────────────

export function SortIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18" />
      <path d="M3 12h12" />
      <path d="M3 18h6" />
    </svg>
  );
}

export function FilterIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46" />
    </svg>
  );
}

export function CheckIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

export function RobotIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect x="4" y="8" width="16" height="12" rx="2" />
      <path d="M12 8V4" />
      <circle cx="12" cy="3" r="1" />
      <circle cx="9" cy="14" r="1.5" fill="currentColor" />
      <circle cx="15" cy="14" r="1.5" fill="currentColor" />
      <path d="M9 18h6" />
      <path d="M2 14h2" />
      <path d="M20 14h2" />
    </svg>
  );
}

export function TrashIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" />
      <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

export function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className={className}>
      <path d="M12 2a10 10 0 0110 10" />
    </svg>
  );
}

// ── AI Info Icon ────────────────────────────────────────────

export function AiInfoIcon({ result }: { result: AiScreenResult }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const recommendation = result.error
    ? "error"
    : result.vertical
    ? "reject"
    : result.auto_decided
    ? result.predicted_class === "reject"
      ? "reject"
      : "approve"
    : "defer";

  const recColor =
    recommendation === "approve"
      ? "text-green-600"
      : recommendation === "reject"
      ? "text-red-600"
      : recommendation === "defer"
      ? "text-amber-600"
      : "text-muted-foreground";

  const iconColor = result.error
    ? "bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-800/60"
    : recommendation === "defer"
    ? "bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400 hover:bg-amber-200 dark:hover:bg-amber-800/60"
    : "bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800/60";

  return (
    <div className="relative inline-flex" ref={ref}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        className={`w-3.5 h-3.5 rounded-full ${iconColor} text-[9px] font-bold flex items-center justify-center transition-colors flex-shrink-0`}
      >
        {result.error ? "!" : "i"}
      </button>
      {open && (
        <div
          className="absolute left-0 top-full mt-1 z-[60] rounded-lg border bg-background shadow-lg p-2 min-w-[160px] text-[11px] space-y-0.5"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between">
            <span className="text-muted-foreground">vertical</span>
            <span className={result.vertical ? "text-red-600 font-medium" : ""}>{result.vertical ? "yes" : "no"}</span>
          </div>
          {result.max_ovl_score != null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">OVL</span>
              <span>{Math.round(result.max_ovl_score * 100)}</span>
            </div>
          )}
          {result.max_otb_score != null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">OTB</span>
              <span>{Math.round(result.max_otb_score * 100)}</span>
            </div>
          )}
          {result.confidence != null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">{result.predicted_class ?? "?"}</span>
              <span>{Math.round(result.confidence * 100)}</span>
            </div>
          )}
          <div className="flex justify-between border-t pt-0.5 mt-0.5">
            <span className="text-muted-foreground">rec</span>
            <span className={`font-medium ${recColor}`}>{recommendation}</span>
          </div>
          {result.error && (
            <div className="text-red-600 text-[10px]">{result.error}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────

export function statusBadge(status: string | null, layoutType?: string | null) {
  switch (status) {
    case "approved":
      return (
        <Badge className="bg-green-600 text-white text-xs">
          {layoutType === "otb_only" ? "OTB only" : "approved"}
        </Badge>
      );
    case "rejected":
      return <Badge variant="destructive" className="text-xs">rejected</Badge>;
    default:
      return <Badge variant="outline" className="text-xs">unscreened</Badge>;
  }
}

const THUMB_URLS = ["0.jpg", "hq1.jpg", "hq2.jpg", "hq3.jpg"];

export function youtubeThumb(videoId: string, index: number): string {
  const file = THUMB_URLS[index] ?? `${index}.jpg`;
  return `https://img.youtube.com/vi/${videoId}/${file}`;
}

export function cardTintClass(status: string | null, layoutType: string | null, inUnscreenedFilter: boolean): string {
  if (!inUnscreenedFilter || status === null) return "";
  if (status === "rejected") return "bg-red-50 dark:bg-red-950/30 opacity-60";
  if (status === "approved") {
    return "bg-green-50 dark:bg-green-950/30 opacity-75";
  }
  return "opacity-60";
}

export function computePageSize(): number {
  if (typeof window === "undefined") return 6;
  const vh = window.innerHeight;
  const headerHeight = 130;
  const cardHeight = 308;
  const cols = window.innerWidth >= 1280 ? 3 : 2;
  const rows = Math.max(1, Math.floor((vh - headerHeight) / cardHeight));
  return rows * cols;
}

// ── Toast Hook ──────────────────────────────────────────────

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const addToast = useCallback((toast: Omit<Toast, "id">) => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { ...toast, id }]);
    const timer = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
      toastTimers.current.delete(id);
    }, 6000);
    toastTimers.current.set(id, timer);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = toastTimers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      toastTimers.current.delete(id);
    }
  }, []);

  return { toasts, addToast, removeToast };
}

// ── Toast Container ─────────────────────────────────────────

export function ToastContainer({
  toasts,
  removeToast,
}: {
  toasts: Toast[];
  removeToast: (id: string) => void;
}) {
  if (toasts.length === 0) return null;

  // Show max 3 toasts stacked; newest on top
  const visible = toasts.slice(-3).reverse();

  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 max-w-sm">
      <div className="relative" style={{ height: 48 }}>
        {visible.map((toast, i) => {
          const bgClass =
            toast.type === "error"
              ? "bg-destructive/10 border-destructive/30 text-destructive"
              : toast.type === "warning"
              ? "bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-400"
              : "bg-background border-border";

          return (
            <div
              key={toast.id}
              className={`absolute left-1/2 rounded-lg border shadow-lg p-3 text-sm flex items-center gap-3 transition-all duration-200 ${bgClass}`}
              style={{
                transform: `translateX(-50%) scale(${1 - i * 0.04})`,
                top: i * 6,
                zIndex: 50 - i,
                opacity: i === 0 ? 1 : 0.5,
                width: "max-content",
                maxWidth: "24rem",
                pointerEvents: i === 0 ? "auto" : "none",
              }}
            >
              <span className="flex-1 truncate">{toast.message}</span>
              {i === 0 && toast.undoAction && (
                <button
                  onClick={async () => {
                    await toast.undoAction!();
                    removeToast(toast.id);
                  }}
                  className="text-xs font-medium text-primary underline whitespace-nowrap"
                >
                  Undo
                </button>
              )}
              {i === 0 && (
                <button
                  onClick={() => removeToast(toast.id)}
                  className="text-muted-foreground hover:text-foreground text-xs"
                >
                  &times;
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Toolbar Components ──────────────────────────────────────

export function ToolbarIconButton({
  children,
  active,
  label,
  className,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  active?: boolean;
  label: string;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={onClick}
        disabled={disabled}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={`flex items-center justify-center w-8 h-8 rounded-xl transition-all duration-150 disabled:opacity-50 disabled:pointer-events-none ${
          active
            ? "bg-background text-foreground shadow-sm"
            : "text-muted-foreground/50 hover:text-foreground hover:bg-background/60"
        } ${className ?? ""}`}
      >
        {children}
      </button>
      {hovered && !disabled && (
        <div className="absolute top-full mt-1.5 left-1/2 -translate-x-1/2 px-2 py-1 rounded-md bg-foreground text-background text-xs font-medium whitespace-nowrap z-[60] pointer-events-none">
          {label}
        </div>
      )}
    </div>
  );
}

export function ToolbarDropdown({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={ref}
      className="absolute top-full mt-1 left-0 z-[60] rounded-lg border bg-background shadow-lg p-2 min-w-[200px]"
    >
      {children}
    </div>
  );
}

// ── Unscreened Actions ──────────────────────────────────────

export function UnscreenedActions({
  videoId,
  onStatusChange,
}: {
  videoId: string;
  onStatusChange: (videoId: string, status: string | null, layoutType?: string) => Promise<void>;
}) {
  return (
    <div className="flex items-center gap-1 flex-shrink-0">
      <button
        onClick={() => onStatusChange(videoId, "approved", "otb_only")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-amber-600 hover:bg-amber-700 text-white transition-colors"
      >
        OTB Only
      </button>
      <button
        onClick={() => onStatusChange(videoId, "approved")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-green-600 hover:bg-green-700 text-white transition-colors"
      >
        Approve
      </button>
      <button
        onClick={() => onStatusChange(videoId, "rejected")}
        className="px-2 py-1 rounded-md text-xs font-medium bg-destructive hover:bg-destructive/90 text-destructive-foreground transition-colors"
      >
        Reject
      </button>
    </div>
  );
}

// ── Status Dropdown ─────────────────────────────────────────

export function StatusDropdown({
  video,
  onStatusChange,
}: {
  video: VideoWithReason;
  onStatusChange: (videoId: string, status: string | null, layoutType?: string) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const statusOptions = [
    {
      label: "Approve",
      status: "approved" as const,
      layoutType: undefined,
      disabled: video.screening_status === "approved" && video.layout_type !== "otb_only",
    },
    {
      label: "OTB Only",
      status: "approved" as const,
      layoutType: "otb_only",
      disabled: video.screening_status === "approved" && video.layout_type === "otb_only",
    },
    {
      label: "Reject",
      status: "rejected" as const,
      layoutType: undefined,
      disabled: video.screening_status === "rejected",
    },
    {
      label: "Unscreen",
      status: null,
      layoutType: undefined,
      disabled: video.screening_status === null,
    },
  ];

  return (
    <div className="relative flex-shrink-0" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 cursor-pointer"
      >
        {statusBadge(video.screening_status, video.layout_type)}
        {video.inspect_reason && (
          <span className="text-[10px] text-muted-foreground">{video.inspect_reason}</span>
        )}
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`w-3 h-3 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-[60] rounded-lg border bg-background shadow-lg p-1 min-w-[130px]">
          {statusOptions.map((opt) => (
            <button
              key={opt.label}
              disabled={opt.disabled}
              onClick={() => {
                onStatusChange(video.video_id, opt.status, opt.layoutType);
                setOpen(false);
              }}
              className="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-muted transition-colors disabled:opacity-40 disabled:pointer-events-none"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
