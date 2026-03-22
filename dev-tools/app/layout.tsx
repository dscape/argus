import type { Metadata } from "next";
import IconSidebar from "@/components/IconSidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Argus",
  description: "Developer tools for the Argus chess video pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="flex min-h-screen">
          <IconSidebar />
          <main className="flex-1 p-6 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
