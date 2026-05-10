"use client";

import type { MaceResult } from "@/lib/types";

interface Props {
  results: MaceResult[];
}

export function ResultsTable({ results }: Props) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">Formula</th>
            <th className="px-4 py-3 font-medium">Atoms</th>
            <th className="px-4 py-3 font-medium">x(B)</th>
            <th className="px-4 py-3 font-medium">ML Form. E</th>
            <th className="px-4 py-3 font-medium">DFT E</th>
            <th className="px-4 py-3 font-medium">DFT E above hull</th>
            <th className="px-4 py-3 font-medium">DFT Stable?</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {results.map((r, i) => (
            <tr
              key={`${r.composition}-${i}`}
              className={
                r.mace_stable
                  ? "bg-blue-50/50"
                  : "bg-white"
              }
            >
              <td className="px-4 py-3 font-medium text-[var(--foreground)]">
                {r.formula}
              </td>
              <td className="px-4 py-3 text-slate-500">{r.n_atoms}</td>
              <td className="px-4 py-3 tabular-nums text-slate-500">
                {r.x_B.toFixed(3)}
              </td>
              <td className="px-4 py-3 tabular-nums text-slate-500">
                {r.formation_energy_eV_atom.toFixed(4)}
              </td>
              <td className="px-4 py-3 tabular-nums text-slate-500">
                {r.mace_energy_eV_atom.toFixed(4)}
              </td>
              <td className="px-4 py-3 tabular-nums text-slate-500">
                {r.mace_e_above_hull_eV_atom.toFixed(4)}
              </td>
              <td className="px-4 py-3">
                {r.mace_stable ? (
                  <span className="inline-flex items-center rounded-full bg-[var(--accent-dim)] px-2 py-0.5 text-xs font-semibold text-[var(--accent-dark)]">
                    Confirmed
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-400">
                    Unstable
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
