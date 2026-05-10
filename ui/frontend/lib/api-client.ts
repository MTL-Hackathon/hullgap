import type { CandidateResult, MaceResult, MpPhase, StructureData } from "./types";

export async function generateCandidates(
  elementA: string,
  elementB: string,
  nCandidates: number
): Promise<CandidateResult[]> {
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
  const res = await fetch(`/api/mp-phases?a=${elementA}&b=${elementB}`);
  if (!res.ok) return [];
  const json = await res.json();
  return json.phases ?? [];
}
