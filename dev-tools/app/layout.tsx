import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Argus Dev Tools",
  description: "Developer tools for the Argus chess video pipeline",
};

const navItems = [
  { href: "/synthetic", label: "Tensors" },
  { href: "/overlay-tester", label: "Overlay Tester" },
  { href: "/calibration", label: "Calibration" },
  { href: "/video-annotator", label: "Video Annotator" },
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
            </div>
            <nav className="space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="block rounded-md px-3 py-2 text-sm text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </aside>
          <main className="flex-1 p-6 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
