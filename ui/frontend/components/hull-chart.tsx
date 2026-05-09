"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Line,
  ComposedChart,
} from "recharts";

interface HullPoint {
  x_B: number;
  energy: number;
  formula: string;
  stable: boolean;
}

interface Props {
  data: HullPoint[];
  elementA: string;
  elementB: string;
  title?: string;
}

export function HullChart({ data, elementA, elementB, title }: Props) {
  const onHull = data.filter((d) => d.stable);
  const aboveHull = data.filter((d) => !d.stable);

  const hullLine = [
    { x_B: 0, energy: 0 },
    ...onHull.sort((a, b) => a.x_B - b.x_B),
    { x_B: 1, energy: 0 },
  ];

  return (
    <div className="w-full">
      {title && (
        <h3 className="mb-3 text-sm font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
          <XAxis
            dataKey="x_B"
            type="number"
            domain={[-0.05, 1.05]}
            label={{
              value: `x(${elementB})`,
              position: "insideBottom",
              offset: -15,
              style: { fontSize: 12, fill: "#64748b" },
            }}
            tick={{ fontSize: 11, fill: "#94a3b8" }}
          />
          <YAxis
            dataKey="energy"
            type="number"
            label={{
              value: "Formation energy (eV/atom)",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              style: { fontSize: 12, fill: "#64748b" },
            }}
            tick={{ fontSize: 11, fill: "#94a3b8" }}
          />
          <Tooltip
            content={({ payload }) => {
              if (!payload || payload.length === 0) return null;
              const d = payload[0]?.payload;
              if (!d) return null;
              return (
                <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
                  <p className="font-semibold">{d.formula || "Endpoint"}</p>
                  <p className="text-slate-500">
                    x(B) = {d.x_B?.toFixed(3)} | E = {d.energy?.toFixed(4)} eV/atom
                  </p>
                </div>
              );
            }}
          />
          <Legend
            verticalAlign="top"
            wrapperStyle={{ fontSize: 11, paddingBottom: 8 }}
          />

          <Line
            data={hullLine}
            dataKey="energy"
            stroke="#289ff0"
            strokeWidth={2}
            dot={false}
            name="Convex hull"
            legendType="line"
          />
          <Scatter
            data={aboveHull}
            dataKey="energy"
            fill="#94a3b8"
            name="Above hull"
            opacity={0.6}
            r={4}
          />
          <Scatter
            data={onHull}
            dataKey="energy"
            fill="#289ff0"
            name="On / below hull"
            r={6}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
