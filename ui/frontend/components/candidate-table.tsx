"use client";

import { Fragment, useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import type { CandidateResult, MpPhase, StructureData } from "@/lib/types";
import { fetchStructure, fetchStructureByIdx } from "@/lib/api-client";
import {
  StructureScene,
  ELEMENT_COLORS,
  guessPrototype,
} from "./crystal-viewer";

const subscribeNoop = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

type ExpandMode = "details" | "compare";

interface Props {
  candidates: CandidateResult[];
  selected: Set<number>;
  originalIndices: number[];
  onToggle: (index: number) => void;
  onToggleAll: () => void;
  elementA: string;
  elementB: string;
  mpPhases: MpPhase[];
}

function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    [a, b] = [b, a % b];
  }
  return a || 1;
}

function reducedComposition(formula: string): Record<string, number> {
  const re = /([A-Z][a-z]?)(\d*)/g;
  const counts: Record<string, number> = {};
  let m: RegExpExecArray | null;
  while ((m = re.exec(formula)) !== null) {
    if (!m[1]) continue;
    const n = m[2] ? parseInt(m[2], 10) : 1;
    counts[m[1]] = (counts[m[1]] || 0) + n;
  }
  const values = Object.values(counts);
  if (values.length === 0) return counts;
  const g = values.reduce((a, b) => gcd(a, b));
  if (g <= 1) return counts;
  for (const k of Object.keys(counts)) counts[k] = counts[k] / g;
  return counts;
}

function compositionsEqual(
  a: Record<string, number>,
  b: Record<string, number>,
): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => a[k] === b[k]);
}

function findMpMatch(
  candidate: CandidateResult,
  mpPhases: MpPhase[],
): MpPhase | null {
  if (!mpPhases.length) return null;
  const target = reducedComposition(candidate.formula);
  const matches = mpPhases.filter((p) =>
    compositionsEqual(reducedComposition(p.formula), target),
  );
  if (!matches.length) return null;
  // Prefer the lowest-energy MP entry for this composition.
  return matches.reduce((best, p) =>
    p.formation_energy < best.formation_energy ? p : best,
  );
}

function relAgreement(a: number | null | undefined, b: number | null | undefined): number | null {
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return null;
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-6);
  return Math.max(0, 1 - Math.abs(a - b) / denom);
}

function agreementScore(candidate: CandidateResult, mp: MpPhase): number {
  const parts: number[] = [];
  const fe = relAgreement(candidate.formation_energy_eV_atom, mp.formation_energy);
  if (fe != null) parts.push(fe);
  const eh = relAgreement(candidate.e_above_hull_eV_atom, mp.e_above_hull);
  if (eh != null) parts.push(eh);
  parts.push(
    candidate.crystal_system && mp.crystal_system &&
      candidate.crystal_system.toLowerCase() === mp.crystal_system.toLowerCase()
      ? 1
      : 0,
  );
  if (!parts.length) return 0;
  return parts.reduce((s, v) => s + v, 0) / parts.length;
}

function ComparisonCell({
  label,
  candidate,
  mp,
  unit,
  digits = 4,
}: {
  label: string;
  candidate: number | string | null | undefined;
  mp: number | string | null | undefined;
  unit?: string;
  digits?: number;
}) {
  const fmt = (v: number | string | null | undefined) => {
    if (v == null) return "—";
    if (typeof v === "number") {
      if (!Number.isFinite(v)) return "—";
      return v.toFixed(digits);
    }
    return v;
  };
  let agree: number | null = null;
  if (typeof candidate === "number" && typeof mp === "number") {
    agree = relAgreement(candidate, mp);
  } else if (typeof candidate === "string" && typeof mp === "string") {
    agree = candidate.toLowerCase() === mp.toLowerCase() ? 1 : 0;
  }
  const tone =
    agree == null
      ? "text-slate-400"
      : agree >= 0.95
        ? "text-emerald-600"
        : agree >= 0.8
          ? "text-amber-600"
          : "text-rose-600";
  return (
    <>
      <div className="font-medium text-[var(--foreground)]">{label}</div>
      <div className="tabular-nums text-slate-700">
        {fmt(candidate)}
        {unit && candidate != null ? ` ${unit}` : ""}
      </div>
      <div className="tabular-nums text-slate-700">
        {fmt(mp)}
        {unit && mp != null ? ` ${unit}` : ""}
      </div>
      <div className={`tabular-nums text-right ${tone}`}>
        {agree == null ? "—" : `${(agree * 100).toFixed(1)}%`}
      </div>
    </>
  );
}

function ComparisonView({
  candidate,
  mp,
  structure,
}: {
  candidate: CandidateResult;
  mp: MpPhase;
  structure: StructureData | null;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Comparison with Materials Project ({mp.id})
        </h4>
        <span className="text-xs font-semibold text-[var(--accent-dark)]">
          Overall agreement: {(agreementScore(candidate, mp) * 100).toFixed(1)}%
        </span>
      </div>

      <div className="grid grid-cols-[1.1fr_1fr_1fr_0.6fr] gap-x-4 gap-y-2 text-sm">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Quantity
        </div>
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Candidate
        </div>
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          MP
        </div>
        <div className="text-right text-xs font-semibold uppercase tracking-wide text-slate-400">
          Agreement
        </div>

        <ComparisonCell
          label="Formula"
          candidate={candidate.formula}
          mp={mp.formula}
        />
        <ComparisonCell
          label="Crystal system"
          candidate={candidate.crystal_system}
          mp={mp.crystal_system}
        />
        <ComparisonCell
          label="Formation energy"
          candidate={candidate.formation_energy_eV_atom}
          mp={mp.formation_energy}
          unit="eV/atom"
        />
        <ComparisonCell
          label="E above hull"
          candidate={candidate.e_above_hull_eV_atom}
          mp={mp.e_above_hull}
          unit="eV/atom"
        />
        <ComparisonCell
          label="x(B)"
          candidate={candidate.x_B}
          mp={null}
        />
        <ComparisonCell
          label="Atoms in cell"
          candidate={candidate.n_atoms}
          mp={mp.n_sites}
          digits={0}
        />
        <ComparisonCell
          label="Volume"
          candidate={structure?.volume ?? null}
          mp={mp.volume}
          unit="Å³"
          digits={2}
        />
        <ComparisonCell
          label="a"
          candidate={structure?.lattice_params.a ?? null}
          mp={mp.a}
          unit="Å"
          digits={3}
        />
        <ComparisonCell
          label="b"
          candidate={structure?.lattice_params.b ?? null}
          mp={mp.b}
          unit="Å"
          digits={3}
        />
        <ComparisonCell
          label="c"
          candidate={structure?.lattice_params.c ?? null}
          mp={mp.c}
          unit="Å"
          digits={3}
        />
        <ComparisonCell
          label="α"
          candidate={structure?.lattice_params.alpha ?? null}
          mp={mp.alpha}
          unit="°"
          digits={2}
        />
        <ComparisonCell
          label="β"
          candidate={structure?.lattice_params.beta ?? null}
          mp={mp.beta}
          unit="°"
          digits={2}
        />
        <ComparisonCell
          label="γ"
          candidate={structure?.lattice_params.gamma ?? null}
          mp={mp.gamma}
          unit="°"
          digits={2}
        />
      </div>
      {structure == null && (
        <p className="text-xs text-slate-400">
          Lattice parameters and volume become available once the structure has
          loaded.
        </p>
      )}
    </div>
  );
}

function ExpandedRow({
  candidate,
  elementA,
  elementB,
  mode,
  mpMatch,
  colSpan,
}: {
  candidate: CandidateResult;
  elementA: string;
  elementB: string;
  mode: ExpandMode;
  mpMatch: MpPhase | null;
  colSpan: number;
}) {
  const mounted = useSyncExternalStore(subscribeNoop, getClientSnapshot, getServerSnapshot);
  const [structure, setStructure] = useState<StructureData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    // Same logic as CrystalViewer: prefer the real relaxed CIF when the
    // candidate carries a hull-CSV row identifier. Only fall back to the
    // synthetic prototype when no idx/system is available.
    const hasRealStructure =
      typeof candidate.idx === "number" &&
      Number.isFinite(candidate.idx) &&
      !!candidate.system;

    const fetchPromise = hasRealStructure
      ? fetchStructureByIdx(candidate.system as string, candidate.idx as number)
          .catch(() =>
            fetchStructure(elementA, elementB, guessPrototype(candidate.x_B))
          )
      : fetchStructure(elementA, elementB, guessPrototype(candidate.x_B));

    fetchPromise
      .then((data) => { if (!cancelled) setStructure(data); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load structure"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [candidate, elementA, elementB]);

  const cameraDistance = useMemo(() => {
    if (!structure) return 10;
    const { a, b, c } = structure.lattice_params;
    // 2x2x2 supercell -> need ~3x max(a,b,c) to frame the whole cluster.
    return Math.max(a, b, c) * 3.2;
  }, [structure]);

  return (
    <tr>
      <td colSpan={colSpan} className="p-0">
        <div
          className="border-t border-dashed border-slate-200 bg-slate-50/60 px-5 py-5"
          style={{ animation: "expandRow 0.2s ease-out" }}
        >
          {mode === "compare" && mpMatch ? (
            <ComparisonView
              candidate={candidate}
              mp={mpMatch}
              structure={structure}
            />
          ) : (
          <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
            {/* Left: details */}
            <div className="space-y-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Candidate details
              </h4>
              <div className="grid gap-x-6 gap-y-2.5 text-sm sm:grid-cols-2">
                <div>
                  <span className="font-medium text-[var(--foreground)]">Composition:</span>{" "}
                  <span className="text-slate-600">{candidate.composition}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Formula:</span>{" "}
                  <span className="text-slate-600">{candidate.formula}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Lattice geometry:</span>{" "}
                  <span className="text-slate-600">{structure?.crystal_system ?? "—"}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Detected symmetry:</span>{" "}
                  <span className="text-slate-600">{candidate.crystal_system}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Atoms:</span>{" "}
                  <span className="text-slate-600">{candidate.n_atoms}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">x(B):</span>{" "}
                  <span className="tabular-nums text-slate-600">{candidate.x_B.toFixed(4)}</span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Formation energy:</span>{" "}
                  <span className="tabular-nums text-slate-600">
                    {candidate.formation_energy_eV_atom.toFixed(4)} eV/atom
                  </span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">E above hull:</span>{" "}
                  <span className="tabular-nums text-slate-600">
                    {candidate.e_above_hull_eV_atom.toFixed(4)} eV/atom
                  </span>
                </div>
                <div>
                  <span className="font-medium text-[var(--foreground)]">Predicted stable:</span>{" "}
                  {candidate.predicted_stable ? (
                    <span className="inline-flex items-center rounded-full bg-[var(--accent-dim)] px-2 py-0.5 text-xs font-semibold text-[var(--accent-dark)]">
                      Yes
                    </span>
                  ) : (
                    <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-400">
                      No
                    </span>
                  )}
                </div>
              </div>

              {structure && (
                <>
                  <h4 className="pt-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Lattice parameters
                  </h4>
                  <div className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
                    <div>
                      <span className="font-medium text-[var(--foreground)]">a:</span>{" "}
                      <span className="tabular-nums text-slate-600">
                        {structure.lattice_params.a.toFixed(3)} &#x212B;
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-[var(--foreground)]">b:</span>{" "}
                      <span className="tabular-nums text-slate-600">
                        {structure.lattice_params.b.toFixed(3)} &#x212B;
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-[var(--foreground)]">c:</span>{" "}
                      <span className="tabular-nums text-slate-600">
                        {structure.lattice_params.c.toFixed(3)} &#x212B;
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-[var(--foreground)]">Volume:</span>{" "}
                      <span className="tabular-nums text-slate-600">
                        {structure.volume.toFixed(2)} &#x212B;&sup3;
                      </span>
                    </div>
                    <div className="sm:col-span-2">
                      <span className="font-medium text-[var(--foreground)]">Angles:</span>{" "}
                      <span className="tabular-nums text-slate-600">
                        &alpha;={structure.lattice_params.alpha.toFixed(1)}&deg;{" "}
                        &beta;={structure.lattice_params.beta.toFixed(1)}&deg;{" "}
                        &gamma;={structure.lattice_params.gamma.toFixed(1)}&deg;
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3 pt-1">
                    {[...new Set(structure.species)].map((sym) => (
                      <div key={sym} className="flex items-center gap-1.5">
                        <span
                          className="inline-block h-3 w-3 rounded-full"
                          style={{ backgroundColor: ELEMENT_COLORS[sym] || "#888" }}
                        />
                        <span className="text-xs font-medium text-slate-600">{sym}</span>
                      </div>
                    ))}
                    <div className="flex items-center gap-1.5">
                      <span className="inline-block h-0.5 w-4 rounded bg-[#e03030]" />
                      <span className="text-xs font-medium text-slate-600">Unit cell</span>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Right: 3D viewer */}
            <div>
              <div className="mb-2">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Structure preview
                </h4>
              </div>
              <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-slate-200 bg-white">
                {loading && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/70">
                    <Loader2 className="h-5 w-5 animate-spin text-[var(--accent)]" />
                  </div>
                )}
                {error && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center px-4 text-center text-xs text-rose-500">
                    {error}
                  </div>
                )}
                {mounted && structure && (
                  <Canvas
                    camera={{
                      position: [
                        cameraDistance * 0.7,
                        cameraDistance * 0.5,
                        cameraDistance,
                      ],
                      fov: 50,
                    }}
                    gl={{ antialias: true }}
                    style={{ width: "100%", height: "100%" }}
                  >
                    <StructureScene structure={structure} />
                  </Canvas>
                )}
              </div>
            </div>
          </div>
          )}
        </div>
      </td>
    </tr>
  );
}

export function CandidateTable({
  candidates,
  selected,
  originalIndices,
  onToggle,
  onToggleAll,
  elementA,
  elementB,
  mpPhases,
}: Props) {
  const allSelected =
    candidates.length > 0 && originalIndices.every((i) => selected.has(i));
  const [expanded, setExpanded] = useState<{ idx: number; mode: ExpandMode } | null>(null);

  const mpMatches = useMemo(() => {
    return candidates.map((c) => findMpMatch(c, mpPhases));
  }, [candidates, mpPhases]);

  const toggleExpand = useCallback((origIdx: number, mode: ExpandMode) => {
    setExpanded((prev) => {
      if (!prev || prev.idx !== origIdx) return { idx: origIdx, mode };
      if (prev.mode !== mode) return { idx: origIdx, mode };
      return null;
    });
  }, []);

  const colSpan = 10;

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="w-10 px-3 py-3 font-medium" />
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
            <th className="px-4 py-3 font-medium">Stable?</th>
            <th className="px-4 py-3 font-medium">In MP</th>
            <th className="px-4 py-3 font-medium">Agreement</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {candidates.map((c, localIdx) => {
            const origIdx = originalIndices[localIdx];
            const isExpanded = expanded?.idx === origIdx;
            const mp = mpMatches[localIdx];
            const agreement = mp ? agreementScore(c, mp) : null;
            const agreementTone =
              agreement == null
                ? "text-slate-400"
                : agreement >= 0.95
                  ? "text-emerald-600"
                  : agreement >= 0.8
                    ? "text-amber-600"
                    : "text-rose-600";

            return (
              <Fragment key={`${c.composition}-${origIdx}`}>
                <tr
                  className={`cursor-pointer transition ${
                    isExpanded
                      ? "bg-[var(--accent-dim)]"
                      : selected.has(origIdx)
                        ? "bg-[var(--accent-dim)]/50"
                        : "bg-white hover:bg-slate-50"
                  }`}
                  onClick={() => toggleExpand(origIdx, "details")}
                >
                  <td className="px-3 py-3 text-slate-400">
                    <ChevronRight
                      className={`h-4 w-4 transition-transform duration-200 ${
                        isExpanded ? "rotate-90" : ""
                      }`}
                    />
                  </td>
                  <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
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
                  <td className="px-4 py-3 text-slate-500">{c.crystal_system}</td>
                  <td className="px-4 py-3 text-slate-500">{c.n_atoms}</td>
                  <td className="px-4 py-3 tabular-nums text-slate-500">
                    {c.x_B.toFixed(3)}
                  </td>
                  <td className="px-4 py-3 tabular-nums text-slate-500">
                    {c.formation_energy_eV_atom.toFixed(4)}
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
                  <td className="px-4 py-3">
                    {mp ? (
                      <span
                        className="inline-flex items-center rounded-full bg-[var(--accent-dim)] px-2 py-0.5 text-xs font-semibold text-[var(--accent-dark)]"
                        title={`Match: ${mp.id} (${mp.formula})`}
                      >
                        Yes
                      </span>
                    ) : (
                      <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-400">
                        No
                      </span>
                    )}
                  </td>
                  <td
                    className="px-4 py-3"
                    onClick={(e) => {
                      if (!mp) return;
                      e.stopPropagation();
                      toggleExpand(origIdx, "compare");
                    }}
                  >
                    {agreement == null ? (
                      <span className="text-slate-300">—</span>
                    ) : (
                      <button
                        type="button"
                        className={`tabular-nums font-semibold underline-offset-4 hover:underline ${agreementTone}`}
                        title="Click to compare every value with Materials Project"
                      >
                        {(agreement * 100).toFixed(1)}%
                      </button>
                    )}
                  </td>
                </tr>
                {isExpanded && (
                  <ExpandedRow
                    candidate={c}
                    elementA={elementA}
                    elementB={elementB}
                    mode={expanded?.mode ?? "details"}
                    mpMatch={mp}
                    colSpan={colSpan}
                  />
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
