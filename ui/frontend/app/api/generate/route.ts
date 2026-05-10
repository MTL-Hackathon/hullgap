import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join } from "path";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const DATA_DIR = join(process.cwd(), "..", "..", "data", "results");

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
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

async function loadPredictionsCSV(elementA: string, elementB: string) {
  const isCoBi =
    (elementA === "Co" && elementB === "Bi") ||
    (elementA === "Bi" && elementB === "Co");
  if (!isCoBi) return null;

  try {
    const csv = await readFile(
      join(DATA_DIR, "cobi_predictions.csv"),
      "utf-8"
    );
    const rows = parseCSV(csv);
    const bKey = elementA === "Co" ? "bi_fraction" : "co_fraction";

    return rows.map((row) => ({
      composition: row.task_id || row.reduced_formula,
      formula: row.reduced_formula,
      n_atoms: parseInt(row.n_atoms, 10),
      x_B: parseFloat(row[bKey]),
      formation_energy_eV_atom: -(parseFloat(row.predicted_probability_stable) || 0),
      e_above_hull_eV_atom: 1 - (parseFloat(row.predicted_probability_stable) || 0),
      predicted_stable: row.predicted_label === "stable",
      crystal_system: row.crystal_system,
    }));
  } catch {
    return null;
  }
}

async function loadHullCSV(elementA: string, elementB: string) {
  const filenames = [
    `${elementA}-${elementB}_mattersim_hull.csv`,
    `${elementB}-${elementA}_mattersim_hull.csv`,
  ];

  for (const filename of filenames) {
    try {
      const csv = await readFile(join(DATA_DIR, filename), "utf-8");
      const rows = parseCSV(csv);
      return rows.map((row) => ({
        composition: row.formula,
        formula: row.formula,
        n_atoms: parseInt(row.n_atoms, 10),
        x_B: parseFloat(row.x_B),
        formation_energy_eV_atom: parseFloat(row.e_form_eV_atom),
        e_above_hull_eV_atom: parseFloat(row.e_above_hull_eV_atom),
        predicted_stable: row.on_hull === "True",
        crystal_system: row.crystal_system || "Unknown",
      }));
    } catch {
      continue;
    }
  }
  return null;
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { element_a, element_b } = body as {
    element_a: string;
    element_b: string;
  };

  // Prefer local CSV data when available
  const csvCandidates =
    (await loadPredictionsCSV(element_a, element_b)) ??
    (await loadHullCSV(element_a, element_b));

  if (csvCandidates) {
    return NextResponse.json(csvCandidates);
  }

  // Fall back to backend
  try {
    const res = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
    const text = await res.text();
    return NextResponse.json(
      { error: text || "Backend error" },
      { status: res.status }
    );
  } catch {
    return NextResponse.json(
      { error: `No data available for ${element_a}-${element_b}` },
      { status: 404 }
    );
  }
}
