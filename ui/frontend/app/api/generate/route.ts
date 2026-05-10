import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join } from "path";
import {
  hullCsvRowsToCandidates,
  parseCSV,
  predictionsCsvRowsToCandidates,
} from "@/lib/hull-csv";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const DATA_DIR = join(process.cwd(), "..", "..", "data", "results");

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
    return predictionsCsvRowsToCandidates(elementA, elementB, rows);
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
      const system = filename.replace(/_mattersim_hull\.csv$/, "");
      return hullCsvRowsToCandidates(system, rows);
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
