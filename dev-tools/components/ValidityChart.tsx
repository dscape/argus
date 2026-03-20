"use client";

import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

const COLORS = {
  valid: "hsl(142, 71%, 45%)",
  invalid: "hsl(0, 84%, 60%)",
};

interface ValidityChartProps {
  valid: number;
  invalid: number;
}

export function ValidityChart({ valid, invalid }: ValidityChartProps) {
  const total = valid + invalid;
  if (total === 0) return null;

  const data = [
    { name: "Valid", value: valid },
    { name: "Invalid", value: invalid },
  ];

  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={40}
            outerRadius={65}
            dataKey="value"
            strokeWidth={2}
            stroke="hsl(var(--background))"
          >
            <Cell fill={COLORS.valid} />
            <Cell fill={COLORS.invalid} />
          </Pie>
          <Tooltip
            formatter={(value, name) => [
              `${value} (${Math.round((Number(value) / total) * 100)}%)`,
              name,
            ]}
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: "12px",
            }}
          />
          <Legend
            formatter={(value: string) => (
              <span className="text-xs text-foreground">{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
