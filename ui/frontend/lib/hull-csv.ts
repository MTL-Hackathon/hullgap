import type { CandidateResult } from "./types";

export function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h.trim()] = values[i]?.trim() ?? "";
    });
    return row;
  });
}

/** Rows from `candidates_curated.csv` (first column `system`). */
export function curatedDemoRowsToCandidates(
  rows: Record<string, string>[],
  elementA: string,
  elementB: string,
): CandidateResult[] | null {
  const s1 = `${elementA}-${elementB}`;
  const s2 = `${elementB}-${elementA}`;
  const matched = rows.filter((r) => r.system === s1 || r.system === s2);
  if (matched.length === 0) return null;
  const system = matched[0].system as string;
  const hullRows = matched.map((r) => {
    const { system: _s, ...rest } = r;
    return rest;
  });
  return hullCsvRowsToCandidates(system, hullRows);
}

export function hullCsvRowsToCandidates(
  system: string,
  rows: Record<string, string>[],
): CandidateResult[] {
  return rows.map((row) => {
    const parsedIdx = parseInt(row.idx, 10);
    return {
      composition: row.formula,
      formula: row.formula,
      n_atoms: parseInt(row.n_atoms, 10),
      x_B: parseFloat(row.x_B),
      formation_energy_eV_atom: parseFloat(row.e_form_eV_atom),
      e_above_hull_eV_atom: parseFloat(row.e_above_hull_eV_atom),
      predicted_stable: row.on_hull === "True",
      crystal_system: row.crystal_system || "Unknown",
      idx: Number.isFinite(parsedIdx) ? parsedIdx : undefined,
      system,
    };
  });
}

/** Co–Bi legacy predictions layout (optional file in data/results). */
export function predictionsCsvRowsToCandidates(
  elementA: string,
  elementB: string,
  rows: Record<string, string>[],
): CandidateResult[] {
  const bKey = elementA === "Co" ? "bi_fraction" : "co_fraction";
  return rows.map((row) => ({
    composition: row.task_id || row.reduced_formula,
    formula: row.reduced_formula,
    n_atoms: parseInt(row.n_atoms, 10),
    x_B: parseFloat(row[bKey]),
    formation_energy_eV_atom: -(parseFloat(row.predicted_probability_stable) || 0),
    e_above_hull_eV_atom: 1 - (parseFloat(row.predicted_probability_stable) || 0),
    predicted_stable: row.predicted_label === "stable",
    crystal_system: row.crystal_system || "Unknown",
  }));
}
