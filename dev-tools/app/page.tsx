import Link from "next/link";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";

const toolGroups = [
  {
    label: "Synthetic",
    tools: [
      {
        href: "/synthetic",
        title: "Synthetic Monitor",
        description:
          "Watch synthetic data generation in real time. Browse generated clips, inspect frames and tensors, and view aggregated pipeline stats.",
      },
    ],
  },
  {
    label: "Video",
    tools: [
      {
        href: "/overlay-tester",
        title: "Overlay Tester",
        description:
          "Upload a screenshot to detect and read 2D chess board overlays. Auto-detect or manually draw a bounding box.",
      },
      {
        href: "/calibration",
        title: "Calibration Editor",
        description:
          "Configure overlay and camera crop regions for YouTube channels. Draw bboxes interactively and preview results.",
      },
      {
        href: "/video-annotator",
        title: "Video Annotator",
        description:
          "Step through video frames, read overlay positions, detect moves, and generate annotated game data.",
      },
    ],
  },
];

export default function Home() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Developer Tools</h2>
      <p className="text-muted-foreground mb-6">
        Visual tools for the Argus pipeline. Select a tool to get started.
      </p>
      {toolGroups.map((group) => (
        <div key={group.label} className="mb-8">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            {group.label}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {group.tools.map((tool) => (
              <Link key={tool.href} href={tool.href}>
                <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
                  <CardHeader>
                    <CardTitle className="text-lg">{tool.title}</CardTitle>
                    <CardDescription>{tool.description}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
