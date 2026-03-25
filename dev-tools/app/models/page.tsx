"use client";

import { useState } from "react";
import AiScreeningInspector from "@/components/AiScreeningInspector";
import AutoCalibrationInspector from "@/components/AutoCalibrationInspector";
import HardCutInspector from "@/components/HardCutInspector";

const TABS = [
  { id: "screening", label: "Screening" },
  { id: "calibration", label: "Auto-Calibration" },
  { id: "hardcuts", label: "Hard Cuts" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function ModelsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("screening");

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Models</h1>

      {/* Tab bar */}
      <div className="flex gap-1 border-b">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "screening" && <AiScreeningInspector />}
        {activeTab === "calibration" && <AutoCalibrationInspector />}
        {activeTab === "hardcuts" && <HardCutInspector />}
      </div>
    </div>
  );
}
