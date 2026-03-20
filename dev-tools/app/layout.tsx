import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Argus Dev Tools",
  description: "Developer tools for the Argus chess video pipeline",
};

const navGroups = [
  {
    label: "Synthetic",
    items: [
      { href: "/synthetic", label: "Synthetic Monitor" },
    ],
  },
  {
    label: "Video",
    items: [
      { href: "/overlay-tester", label: "Overlay Tester" },
      { href: "/calibration", label: "Calibration" },
      { href: "/video-annotator", label: "Video Annotator" },
    ],
  },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="flex min-h-screen">
          <aside className="w-56 border-r bg-muted/40 p-4">
            <div className="mb-6">
              <Link href="/" className="block">
                <h1 className="text-lg font-bold text-foreground">
                  Argus Dev Tools
                </h1>
              </Link>
              <p className="text-xs text-muted-foreground">
                Pipeline diagnostics
              </p>
            </div>
            <nav className="space-y-4">
              {navGroups.map((group) => (
                <div key={group.label}>
                  <p className="px-3 mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {group.label}
                  </p>
                  <div className="space-y-1">
                    {group.items.map((item) => (
                      <Link
                        key={item.href}
                        href={item.href}
                        className="block rounded-md px-3 py-2 text-sm text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                      >
                        {item.label}
                      </Link>
                    ))}
                  </div>
                </div>
              ))}
            </nav>
          </aside>
          <main className="flex-1 p-6 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
