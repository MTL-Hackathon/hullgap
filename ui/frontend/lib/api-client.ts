import { parseCif } from "./cif";
import {
  curatedDemoRowsToCandidates,
  hullCsvRowsToCandidates,
  parseCSV,
  predictionsCsvRowsToCandidates,
} from "./hull-csv";
import { isStaticExport, withBasePath } from "./site";
import type { CandidateResult, MaceResult, MpPhase, StructureData } from "./types";

let cifIndexPromise: Promise<Record<string, Record<string, string>>> | null = null;
let curatedRowsPromise: Promise<Record<string, string>[] | null> | null = null;

function loadCuratedDemoRows(): Promise<Record<string, string>[] | null> {
  if (!curatedRowsPromise) {
    curatedRowsPromise = fetch(withBasePath("/demo/candidates_curated.csv")).then(
      async (r) => {
        if (!r.ok) return null;
        const text = await r.text();
        const rows = parseCSV(text);
        return rows.length ? rows : null;
      },
    );
  }
  return curatedRowsPromise;
}

function loadCifIndex(): Promise<Record<string, Record<string, string>>> {
  if (!cifIndexPromise) {
    cifIndexPromise = fetch(withBasePath("/demo/cif-index.json")).then((r) => {
      if (!r.ok) {
        throw new Error("Missing demo/cif-index.json (run scripts/sync_static_demo_assets.py)");
      }
      return r.json() as Promise<Record<string, Record<string, string>>>;
    });
  }
  return cifIndexPromise;
}

async function loadStaticHullCandidates(
  elementA: string,
  elementB: string,
): Promise<CandidateResult[] | null> {
  const names = [
    `${elementA}-${elementB}_mattersim_hull.csv`,
    `${elementB}-${elementA}_mattersim_hull.csv`,
  ];
  for (const name of names) {
    const res = await fetch(withBasePath(`/demo/results/${name}`));
    if (!res.ok) continue;
    const rows = parseCSV(await res.text());
    const system = name.replace("_mattersim_hull.csv", "");
    return hullCsvRowsToCandidates(system, rows);
  }

  const isCoBi =
    (elementA === "Co" && elementB === "Bi") ||
    (elementA === "Bi" && elementB === "Co");
  if (isCoBi) {
    const res = await fetch(withBasePath("/demo/results/cobi_predictions.csv"));
    if (res.ok) {
      const rows = parseCSV(await res.text());
      return predictionsCsvRowsToCandidates(elementA, elementB, rows);
    }
  }
  return null;
}

export async function generateCandidates(
  elementA: string,
  elementB: string,
  nCandidates: number
): Promise<CandidateResult[]> {
  const fromCurated = await loadCuratedDemoRows().then((rows) =>
    rows ? curatedDemoRowsToCandidates(rows, elementA, elementB) : null,
  );
  if (fromCurated?.length) {
    return nCandidates > 0 ? fromCurated.slice(0, nCandidates) : fromCurated;
  }

  if (isStaticExport) {
    const local = await loadStaticHullCandidates(elementA, elementB);
    if (local) {
      return nCandidates > 0 ? local.slice(0, nCandidates) : local;
    }
    throw new Error(
      `No bundled hull data for ${elementA}-${elementB}. Try a system from the demo dataset or run the full app locally.`,
    );
  }

  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      element_a: elementA,
      element_b: elementB,
      n_candidates: nCandidates,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Candidate generation failed");
  }
  return res.json();
}

export async function validateWithMace(
  candidates: CandidateResult[]
): Promise<MaceResult[]> {
  if (isStaticExport) {
    throw new Error("MACE validation is not available on the static GitHub Pages demo.");
  }
  const res = await fetch("/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidates }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "MACE validation failed");
  }
  return res.json();
}

export async function fetchStructure(
  elementA: string,
  elementB: string,
  prototype: string
): Promise<StructureData> {
  if (isStaticExport) {
    throw new Error(
      "Prototype-based structures need the FastAPI backend. Open a candidate that has hull CSV idx/system for the bundled CIF.",
    );
  }
  const res = await fetch("/api/structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      element_a: elementA,
      element_b: elementB,
      prototype,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to fetch structure");
  }
  return res.json();
}

/**
 * Load the actual relaxed CIF for a candidate. The backend reads
 * data/mattergen/{system}/relaxed/{system}_{idx:03d}_*.cif so the rendered
 * structure agrees with the candidate row's crystal_system label.
 */
export async function fetchStructureByIdx(
  system: string,
  idx: number
): Promise<StructureData> {
  if (isStaticExport) {
    const index = await loadCifIndex();
    const file = index[system]?.[String(idx)];
    if (!file) {
      throw new Error(`No bundled CIF for ${system} #${idx}`);
    }
    const res = await fetch(
      withBasePath(`/demo/mattergen/${encodeURIComponent(system)}/relaxed/${encodeURIComponent(file)}`),
    );
    if (!res.ok) {
      throw new Error(`Failed to load CIF ${file}`);
    }
    const text = await res.text();
    return parseCif(text, file.replace(/\.cif$/, "")) as StructureData;
  }

  const res = await fetch("/api/structure_by_idx", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ system, idx }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to fetch structure");
  }
  return res.json();
}

export async function fetchMpPhases(
  elementA: string,
  elementB: string
): Promise<MpPhase[]> {
  if (isStaticExport) {
    return [];
  }
  const res = await fetch(`/api/mp-phases?a=${elementA}&b=${elementB}`);
  if (!res.ok) return [];
  const json = await res.json();
  return json.phases ?? [];
}
