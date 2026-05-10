"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { Loader2 } from "lucide-react";
import { fetchStructure, fetchStructureByIdx } from "@/lib/api-client";
import type { CandidateResult, StructureData } from "@/lib/types";

const subscribeNoop = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

export const ELEMENT_COLORS: Record<string, string> = {
  Co: "#3366cc",
  Bi: "#cc6633",
  Ni: "#228b22",
  Sb: "#8b008b",
  Fe: "#b22222",
  Mn: "#ff8c00",
  Cr: "#4682b4",
  Ti: "#708090",
  Al: "#b0b0b0",
  Cu: "#b87333",
  Zn: "#7d7d7d",
  Au: "#ffd700",
  Ag: "#c0c0c0",
  Pt: "#e5e4e2",
};

export const ELEMENT_RADII: Record<string, number> = {
  Co: 0.3,
  Bi: 0.4,
  Ni: 0.3,
  Sb: 0.38,
  Fe: 0.3,
  Mn: 0.3,
  Cr: 0.3,
  Ti: 0.32,
  Al: 0.32,
  Cu: 0.3,
  Zn: 0.3,
  Au: 0.34,
  Ag: 0.34,
  Pt: 0.34,
};

export const PROTOTYPES = [
  "CsCl_B2",
  "NaCl_B1",
  "ZnS_B3",
  "NiAs_B81",
  "FeSi_B20",
  "Cu3Au_L12",
  "Ni3Sn_D019",
  "Au3Cu_L12",
  "Sn3Ni_D019",
  "MoSi2_C11b",
  "CaF2_C1",
  "CaF2_C1_inv",
  "CaCu5_D2d",
  "CaCu5_D2d_inv",
];

export function guessPrototype(xB: number): string {
  if (xB < 0.01 || xB > 0.99) return "CsCl_B2";
  const ratio = xB / (1 - xB);
  if (ratio > 4) return "CaCu5_D2d_inv";
  if (ratio > 2.5) return "Sn3Ni_D019";
  if (ratio > 1.8) return "Au3Cu_L12";
  if (ratio > 1.2) return "MoSi2_C11b";
  if (ratio > 0.8) return "CsCl_B2";
  if (ratio > 0.55) return "MoSi2_C11b";
  if (ratio > 0.4) return "Cu3Au_L12";
  if (ratio > 0.25) return "Ni3Sn_D019";
  return "CaCu5_D2d";
}

const EDGE_PAIRS: [number, number][] = [
  [0, 1], [0, 2], [0, 3],
  [1, 4], [1, 5],
  [2, 4], [2, 6],
  [3, 5], [3, 6],
  [4, 7], [5, 7], [6, 7],
];

function cellCorners(
  latticeMatrix: [number, number, number][],
  offset: [number, number, number] = [0, 0, 0],
): [number, number, number][] {
  const [a, b, c] = latticeMatrix;
  const o = offset;
  const base = [
    [0, 0, 0],
    a,
    b,
    c,
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    [a[0] + c[0], a[1] + c[1], a[2] + c[2]],
    [b[0] + c[0], b[1] + c[1], b[2] + c[2]],
    [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]],
  ];
  return base.map(
    (p) => [p[0] + o[0], p[1] + o[1], p[2] + o[2]] as [number, number, number],
  );
}

/** Wrap a fractional coordinate into [0, 1). Relaxation can leave frac
 *  components slightly outside that range; without wrapping, those atoms
 *  appear floating outside the cell wireframe. */
function wrapFrac(x: number): number {
  const w = x - Math.floor(x);
  return w === 1 ? 0 : w;
}

/** Wireframe of the unit cell — twelve red edges built from the eight
 *  corners of the lattice parallelepiped. */
function UnitCellEdges({ latticeMatrix }: { latticeMatrix: [number, number, number][] }) {
  const edges = useMemo(() => {
    const corners = cellCorners(latticeMatrix);
    return EDGE_PAIRS.map(([m, n]) => [corners[m], corners[n]] as [
      [number, number, number],
      [number, number, number],
    ]);
  }, [latticeMatrix]);

  return (
    <>
      {edges.map((points, ei) => (
        <Line key={ei} points={points} color="#e03030" lineWidth={2} />
      ))}
    </>
  );
}

function Atoms({
  species,
  fracCoords,
  latticeMatrix,
}: {
  species: string[];
  fracCoords: [number, number, number][];
  latticeMatrix: [number, number, number][];
}) {
  // Render each atom exactly once at its actual fractional position
  // (wrapped into [0, 1) so post-relaxation drift doesn't leak atoms
  // outside the wireframe). No tiling and no centring shift — both used
  // to fabricate spurious atoms on the cell corners.
  const renderAtoms = useMemo(() => {
    const [aVec, bVec, cVec] = latticeMatrix;
    return fracCoords.map((f, i) => {
      const fx = wrapFrac(f[0]);
      const fy = wrapFrac(f[1]);
      const fz = wrapFrac(f[2]);
      return {
        species: species[i],
        pos: [
          fx * aVec[0] + fy * bVec[0] + fz * cVec[0],
          fx * aVec[1] + fy * bVec[1] + fz * cVec[1],
          fx * aVec[2] + fy * bVec[2] + fz * cVec[2],
        ] as [number, number, number],
      };
    });
  }, [species, fracCoords, latticeMatrix]);

  return (
    <>
      {renderAtoms.map((atom, ai) => {
        const color = ELEMENT_COLORS[atom.species] || "#888888";
        const radius = ELEMENT_RADII[atom.species] || 0.3;
        return (
          <mesh key={ai} position={atom.pos}>
            <sphereGeometry args={[radius, 24, 24]} />
            <meshStandardMaterial color={color} roughness={0.4} metalness={0.3} />
          </mesh>
        );
      })}
    </>
  );
}

export function StructureScene({ structure }: { structure: StructureData }) {
  // Centre of the unit cell parallelepiped is (a + b + c) / 2. Translate by
  // -centre so the orbit target sits at the geometric middle of the cell.
  const center = useMemo(() => {
    const [a, b, c] = structure.lattice_matrix;
    return [
      0.5 * (a[0] + b[0] + c[0]),
      0.5 * (a[1] + b[1] + c[1]),
      0.5 * (a[2] + b[2] + c[2]),
    ] as [number, number, number];
  }, [structure.lattice_matrix]);

  const cameraDistance = useMemo(() => {
    const { a, b, c } = structure.lattice_params;
    return Math.max(a, b, c) * 1.8;
  }, [structure.lattice_params]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <directionalLight position={[-3, -3, 2]} intensity={0.3} />
      <group position={[-center[0], -center[1], -center[2]]}>
        <UnitCellEdges latticeMatrix={structure.lattice_matrix} />
        <Atoms
          species={structure.species}
          fracCoords={structure.frac_coords}
          latticeMatrix={structure.lattice_matrix}
        />
      </group>
      <OrbitControls
        makeDefault
        target={[0, 0, 0]}
        minDistance={cameraDistance * 0.2}
        maxDistance={cameraDistance * 4}
      />
    </>
  );
}

interface CrystalViewerProps {
  elementA: string;
  elementB: string;
  candidate: CandidateResult;
}

export function CrystalViewer({ elementA, elementB, candidate }: CrystalViewerProps) {
  const mounted = useSyncExternalStore(subscribeNoop, getClientSnapshot, getServerSnapshot);
  const [structure, setStructure] = useState<StructureData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    // Prefer the actual relaxed CIF when the candidate carries a CSV row
    // identifier — that way the rendered structure matches the row's
    // crystal_system label. Fall back to the synthetic prototype only when
    // we have no idx/system to look up.
    const hasRealStructure =
      typeof candidate.idx === "number" &&
      Number.isFinite(candidate.idx) &&
      !!candidate.system;

    const fetchPromise = hasRealStructure
      ? fetchStructureByIdx(candidate.system as string, candidate.idx as number)
          .catch(() =>
            // CIF missing on disk — degrade gracefully to the prototype guess
            fetchStructure(elementA, elementB, guessPrototype(candidate.x_B))
          )
      : fetchStructure(elementA, elementB, guessPrototype(candidate.x_B));

    fetchPromise
      .then((data) => { if (!cancelled) setStructure(data); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load structure"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [elementA, elementB, candidate]);

  const cameraDistance = useMemo(() => {
    if (!structure) return 10;
    const { a, b, c } = structure.lattice_params;
    return Math.max(a, b, c) * 1.8;
  }, [structure]);

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
        <div>
          <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
            Structure Preview
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            {candidate.formula} &mdash; {structure?.crystal_system ?? candidate.crystal_system}
          </p>
        </div>

        {error && (
          <div role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        <div className="relative mt-5 aspect-[4/3] w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
          {loading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
            </div>
          )}
          {mounted && structure && (
            <Canvas
              camera={{ position: [cameraDistance * 0.7, cameraDistance * 0.5, cameraDistance], fov: 50 }}
              gl={{ antialias: true }}
              style={{ width: "100%", height: "100%" }}
            >
              <StructureScene structure={structure} />
            </Canvas>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
        <h3 className="text-sm font-semibold tracking-[-0.01em] text-[var(--foreground)]">
          Structure Information
        </h3>
        <div className="mt-3 grid gap-x-8 gap-y-2 text-sm text-slate-600 sm:grid-cols-2">
          <div>
            <span className="font-medium text-[var(--foreground)]">Formula:</span>{" "}
            {candidate.formula}
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Atoms:</span>{" "}
            {candidate.n_atoms}
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Formation energy:</span>{" "}
            {candidate.formation_energy_eV_atom.toFixed(4)} eV/atom
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">E above hull:</span>{" "}
            {candidate.e_above_hull_eV_atom.toFixed(4)} eV/atom
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Predicted stable:</span>{" "}
            <span className={candidate.predicted_stable ? "text-emerald-600 font-semibold" : "text-slate-400"}>
              {candidate.predicted_stable ? "Yes" : "No"}
            </span>
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Lattice geometry:</span>{" "}
            {structure?.crystal_system ?? "—"}
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Detected symmetry:</span>{" "}
            {candidate.crystal_system}
          </div>
          {structure && (
            <>
              <div>
                <span className="font-medium text-[var(--foreground)]">Prototype:</span>{" "}
                {structure.prototype}
              </div>
              <div>
                <span className="font-medium text-[var(--foreground)]">Volume:</span>{" "}
                {structure.volume.toFixed(2)} &#x212B;&sup3;
              </div>
              <div className="sm:col-span-2">
                <span className="font-medium text-[var(--foreground)]">Lattice:</span>{" "}
                a={structure.lattice_params.a.toFixed(3)} &#x212B;,{" "}
                b={structure.lattice_params.b.toFixed(3)} &#x212B;,{" "}
                c={structure.lattice_params.c.toFixed(3)} &#x212B;
              </div>
              <div className="sm:col-span-2">
                <span className="font-medium text-[var(--foreground)]">Angles:</span>{" "}
                &alpha;={structure.lattice_params.alpha.toFixed(1)}&deg;,{" "}
                &beta;={structure.lattice_params.beta.toFixed(1)}&deg;,{" "}
                &gamma;={structure.lattice_params.gamma.toFixed(1)}&deg;
              </div>
            </>
          )}
        </div>

        {structure && (
          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Legend
            </h4>
            <div className="mt-2 flex flex-wrap gap-3">
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
          </div>
        )}
      </div>
    </div>
  );
}
