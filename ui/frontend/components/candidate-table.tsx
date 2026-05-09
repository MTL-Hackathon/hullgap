"use client";

import type { CandidateResult } from "@/lib/types";

interface Props {
  candidates: CandidateResult[];
  selected: Set<number>;
  originalIndices: number[];
  onToggle: (index: number) => void;
  onToggleAll: () => void;
}

export function CandidateTable({ candidates, selected, originalIndices, onToggle, onToggleAll }: Props) {
  const allSelected = candidates.length > 0 && originalIndices.every((i) => selected.has(i));

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={onToggleAll}
                className="h-4 w-4 rounded border-slate-300 text-[var(--accent)] accent-[var(--accent)]"
              />
            </th>
            <th className="px-4 py-3 font-medium">Formula</th>
            <th className="px-4 py-3 font-medium">Crystal System</th>
            <th className="px-4 py-3 font-medium">Atoms</th>
            <th className="px-4 py-3 font-medium">x(B)</th>
            <th className="px-4 py-3 font-medium">Form. E (eV/at)</th>
            <th className="px-4 py-3 font-medium">E above hull</th>
            <th className="px-4 py-3 font-medium">Stable?</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {candidates.map((c, localIdx) => {
            const origIdx = originalIndices[localIdx];
            return (
              <tr
                key={`${c.composition}-${origIdx}`}
                className={`transition ${
                  selected.has(origIdx) ? "bg-[var(--accent-dim)]" : "bg-white hover:bg-slate-50"
                }`}
              >
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={selected.has(origIdx)}
                    onChange={() => onToggle(origIdx)}
                    className="h-4 w-4 rounded border-slate-300 text-[var(--accent)] accent-[var(--accent)]"
                  />
                </td>
                <td className="px-4 py-3 font-medium text-[var(--foreground)]">
                  {c.formula}
                </td>
                <td className="px-4 py-3 text-slate-500">
                  {c.crystal_system}
                </td>
                <td className="px-4 py-3 text-slate-500">{c.n_atoms}</td>
                <td className="px-4 py-3 tabular-nums text-slate-500">
                  {c.x_B.toFixed(3)}
                </td>
                <td className="px-4 py-3 tabular-nums text-slate-500">
                  {c.formation_energy_eV_atom.toFixed(4)}
                </td>
                <td className="px-4 py-3 tabular-nums text-slate-500">
                  {c.e_above_hull_eV_atom.toFixed(4)}
                </td>
                <td className="px-4 py-3">
                  {c.predicted_stable ? (
                    <span className="inline-flex items-center rounded-full bg-[var(--accent-dim)] px-2 py-0.5 text-xs font-semibold text-[var(--accent-dark)]">
                      Stable
                    </span>
                  ) : (
                    <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-400">
                      Above hull
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
