"use client";

import { Fragment, useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import type { CandidateResult, StructureData } from "@/lib/types";
import { fetchStructure } from "@/lib/api-client";
import {
  StructureScene,
  ELEMENT_COLORS,
  PROTOTYPES,
  guessPrototype,
} from "./crystal-viewer";

const subscribeNoop = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

interface Props {
  candidates: CandidateResult[];
  selected: Set<number>;
  originalIndices: number[];
  onToggle: (index: number) => void;
  onToggleAll: () => void;
  elementA: string;
  elementB: string;
}

function ExpandedRow({
  candidate,
  elementA,
  elementB,
}: {
  candidate: CandidateResult;
  elementA: string;
  elementB: string;
}) {
  const mounted = useSyncExternalStore(subscribeNoop, getClientSnapshot, getServerSnapshot);
  const [structure, setStructure] = useState<StructureData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedProto, setSelectedProto] = useState(() => guessPrototype(candidate.x_B));

  const loadStructure = useCallback(
    async (proto: string) => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchStructure(elementA, elementB, proto);
        setStructure(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load structure");
      } finally {
        setLoading(false);
      }
    },
    [elementA, elementB],
  );

  useEffect(() => {
    loadStructure(selectedProto);
  }, [selectedProto, loadStructure]);

  const cameraDistance = useMemo(() => {
    if (!structure) return 10;
    const { a, b, c } = structure.lattice_params;
    return Math.max(a, b, c) * 3.0;
  }, [structure]);

  return (
    <tr>
      <td colSpan={8} className="p-0">
        <div
          className="border-t border-dashed border-slate-200 bg-slate-50/60 px-5 py-5"
          style={{ animation: "expandRow 0.2s ease-out" }}
        >
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
                  <span className="font-medium text-[var(--foreground)]">Crystal system:</span>{" "}
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
              <div className="mb-2 flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Structure preview
                </h4>
                <select
                  value={selectedProto}
                  onChange={(e) => setSelectedProto(e.target.value)}
                  className="h-7 rounded-md border border-[var(--border)] bg-white px-2 text-xs text-[var(--foreground)]"
                >
                  {PROTOTYPES.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
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
}: Props) {
  const allSelected =
    candidates.length > 0 && originalIndices.every((i) => selected.has(i));
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const toggleExpand = useCallback((origIdx: number) => {
    setExpandedIdx((prev) => (prev === origIdx ? null : origIdx));
  }, []);

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
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {candidates.map((c, localIdx) => {
            const origIdx = originalIndices[localIdx];
            const isExpanded = expandedIdx === origIdx;

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
                  onClick={() => toggleExpand(origIdx)}
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
                </tr>
                {isExpanded && (
                  <ExpandedRow
                    candidate={c}
                    elementA={elementA}
                    elementB={elementB}
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
