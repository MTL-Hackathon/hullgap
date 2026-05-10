import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join } from "path";

export const dynamic: "force-static" | "auto" =
  process.env.NEXT_PUBLIC_STATIC_EXPORT === "true" ? "force-static" : "auto";

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

async function loadMaceResultsFromCSV(elementA: string, elementB: string) {
  const filenames = [
    `${elementA}-${elementB}_mattersim_hull.csv`,
    `${elementB}-${elementA}_mattersim_hull.csv`,
  ];

  for (const filename of filenames) {
    try {
      const csv = await readFile(join(DATA_DIR, filename), "utf-8");
      const rows = parseCSV(csv);
      const system = filename.replace(/_mattersim_hull\.csv$/, "");
      return rows.map((row) => {
        const formE = parseFloat(row.e_form_eV_atom);
        const ePerAtom = parseFloat(row.e_per_atom_eV);
        const eAboveHull = parseFloat(row.e_above_hull_eV_atom);
        const onHull = row.on_hull === "True";
        const parsedIdx = parseInt(row.idx, 10);

        return {
          composition: row.formula,
          formula: row.formula,
          n_atoms: parseInt(row.n_atoms, 10),
          x_B: parseFloat(row.x_B),
          formation_energy_eV_atom: formE,
          e_above_hull_eV_atom: eAboveHull,
          predicted_stable: onHull,
          crystal_system: row.crystal_system || "Unknown",
          idx: Number.isFinite(parsedIdx) ? parsedIdx : undefined,
          system,
          mace_energy_eV_atom: ePerAtom,
          mace_e_above_hull_eV_atom: eAboveHull,
          mace_stable: onHull,
        };
      });
    } catch {
      continue;
    }
  }
  return null;
}

export async function POST(request: NextRequest) {
  const body = await request.json();

  // Try to determine elements from the candidates payload
  const candidates = body.candidates as Array<{
    formula: string;
    composition?: string;
  }>;
  let elementA = "";
  let elementB = "";

  if (candidates?.length > 0) {
    const formula = candidates[0].formula || candidates[0].composition || "";
    const elems = formula.match(/[A-Z][a-z]?/g);
    if (elems && elems.length >= 2) {
      elementA = elems[0];
      elementB = elems[1];
    }
  }

  // Prefer local CSV data when available
  if (elementA && elementB) {
    const csvResults = await loadMaceResultsFromCSV(elementA, elementB);
    if (csvResults) {
      return NextResponse.json(csvResults);
    }
  }

  // Fall back to backend
  try {
    const res = await fetch(`${BACKEND_URL}/validate`, {
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
      { error: "No validation data available" },
      { status: 404 }
    );
  }
}
