"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ArrowLeft, Loader2, Search, Download, Eye } from "lucide-react";
import { generateCandidates, validateWithMace, fetchMpPhases } from "@/lib/api-client";
import type { CandidateResult, MaceResult, MpPhase } from "@/lib/types";
import { HeroSection } from "./hero-section";
import { ElementMap } from "./element-map";
import { FloatingElements } from "./floating-elements";
import { StepTracker, type Step } from "./step-tracker";
import { CandidateTable } from "./candidate-table";
import { ResultsTable } from "./results-table";
import { HullChart } from "./hull-chart";
import { CrystalViewer } from "./crystal-viewer";

function computeXB(formula: string, elementB: string): number {
  const re = /([A-Z][a-z]?)(\d*)/g;
  let totalAtoms = 0;
  let bAtoms = 0;
  let match;
  while ((match = re.exec(formula)) !== null) {
    if (!match[1]) continue;
    const count = match[2] ? parseInt(match[2], 10) : 1;
    totalAtoms += count;
    if (match[1] === elementB) bAtoms += count;
  }
  return totalAtoms > 0 ? bAtoms / totalAtoms : 0;
}

export function Workspace() {
  const [elementA, setElementA] = useState("Co");
  const [elementB, setElementB] = useState("Bi");
  const [step, setStep] = useState<Step>("candidates");
  const [candidates, setCandidates] = useState<CandidateResult[] | null>(null);
  const [maceResults, setMaceResults] = useState<MaceResult[] | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [genLoading, setGenLoading] = useState(false);
  const [valLoading, setValLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sliderHeight, setSliderHeight] = useState<number | undefined>(undefined);
  const [crystalSystemFilter, setCrystalSystemFilter] = useState<Set<string>>(new Set());

  const [mpPhases, setMpPhases] = useState<MpPhase[]>([]);
  const [viewerVisited, setViewerVisited] = useState(false);

  const [assembled, setAssembled] = useState(false);
  const [canvasRevealed, setCanvasRevealed] = useState(false);

  const mainRef = useRef<HTMLElement>(null);
  const slideRefs = useRef<(HTMLDivElement | null)[]>([null, null, null]);

  const stepIndex = step === "candidates" ? 0 : step === "validation" ? 1 : 2;

  const goBack = useCallback(() => {
    if (step === "validation") setStep("candidates");
    else if (step === "viewer") setStep("validation");
  }, [step]);

  const scrollToPeriodicTable = useCallback(() => {
    if (!assembled) setAssembled(true);
    const el = document.getElementById("periodic-table");
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 120;
    window.scrollTo({ top, behavior: "smooth" });
  }, [assembled]);

  useEffect(() => {
    const onScroll = () => {
      const threshold = assembled ? 80 : 200;
      const shouldAssemble = window.scrollY > threshold;
      if (shouldAssemble && !assembled) setAssembled(true);
      if (!shouldAssemble && assembled) setAssembled(false);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [assembled]);

  const handleOverlayLanded = useCallback(() => {
    setCanvasRevealed(true);
  }, []);

  const handleDisassembled = useCallback(() => {
    setCanvasRevealed(false);
  }, []);

  const runGenerate = useCallback(async (elA: string, elB: string, n: number) => {
    setElementA(elA);
    setElementB(elB);
    setError(null);
    setGenLoading(true);
    setCandidates(null);
    setMaceResults(null);
    setMpPhases([]);
    setSelected(new Set());
    setCrystalSystemFilter(new Set());
    try {
      const [res] = await Promise.all([
        generateCandidates(elA, elB, n),
        fetchMpPhases(elA, elB).then(setMpPhases).catch(() => {}),
      ]);
      setCandidates(res);
      const stableIndices = new Set<number>();
      res.forEach((c, i) => {
        if (c.predicted_stable) stableIndices.add(i);
      });
      setSelected(stableIndices);
      setStep("candidates");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setGenLoading(false);
    }
  }, []);

  const runValidation = useCallback(async () => {
    if (!candidates) return;
    setError(null);
    setValLoading(true);
    setMaceResults(null);
    try {
      const selectedCandidates = candidates.filter((_, i) => selected.has(i));
      const res = await validateWithMace(selectedCandidates);
      setMaceResults(res);
      setStep("validation");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setValLoading(false);
    }
  }, [candidates, selected]);

  const reset = useCallback(() => {
    setStep("candidates");
    setCandidates(null);
    setMaceResults(null);
    setSelected(new Set());
    setCrystalSystemFilter(new Set());
    setError(null);
    setViewerVisited(false);
  }, []);

  const toggleSelect = useCallback((index: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const downloadCsv = useCallback(() => {
    if (!maceResults) return;
    const header = [
      "formula", "n_atoms", "x_B",
      "formation_energy_eV_atom", "mace_energy_eV_atom",
      "mace_e_above_hull_eV_atom", "mace_stable",
    ].join(",");
    const rows = maceResults.map((r) =>
      [
        r.formula, r.n_atoms, r.x_B,
        r.formation_energy_eV_atom, r.mace_energy_eV_atom,
        r.mace_e_above_hull_eV_atom, r.mace_stable,
      ].join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mof_mace_results_${elementA}-${elementB}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [maceResults, elementA, elementB]);

  const candidateHullData = useMemo(() => {
    if (!candidates) return [];
    return candidates.map((c) => ({
      x_B: c.x_B,
      energy: c.formation_energy_eV_atom,
      formula: c.formula,
      stable: c.predicted_stable,
    }));
  }, [candidates]);

  const mpHullData = useMemo(() => {
    if (!mpPhases.length) return [];
    return mpPhases
      .filter((p) => p.formation_energy != null)
      .map((p) => {
        const xB = computeXB(p.formula, elementB);
        return {
          x_B: xB,
          energy: p.formation_energy,
          formula: p.formula,
          stable: p.is_stable,
          source: "mp" as const,
        };
      })
      .filter((p) => p.x_B > 0 && p.x_B < 1);
  }, [mpPhases, elementB]);

  const maceHullData = useMemo(() => {
    if (!maceResults) return [];
    return maceResults.map((r) => ({
      x_B: r.x_B,
      energy: r.mace_energy_eV_atom,
      formula: r.formula,
      stable: r.mace_stable,
    }));
  }, [maceResults]);

  const uniqueCrystalSystems = useMemo(() => {
    if (!candidates) return [];
    const systems = new Set(candidates.map((c) => c.crystal_system));
    return Array.from(systems).sort();
  }, [candidates]);

  const filteredIndices = useMemo(() => {
    if (!candidates) return [];
    if (crystalSystemFilter.size === 0) {
      return candidates.map((_, i) => i);
    }
    return candidates
      .map((c, i) => (crystalSystemFilter.has(c.crystal_system) ? i : -1))
      .filter((i) => i !== -1);
  }, [candidates, crystalSystemFilter]);

  const filteredCandidates = useMemo(() => {
    if (!candidates) return [];
    return filteredIndices.map((i) => candidates[i]);
  }, [candidates, filteredIndices]);

  const toggleCrystalSystem = useCallback((system: string) => {
    setCrystalSystemFilter((prev) => {
      const next = new Set(prev);
      if (next.has(system)) next.delete(system);
      else next.add(system);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    if (!candidates) return;
    setSelected((prev) => {
      const allFilteredSelected = filteredIndices.every((i) => prev.has(i));
      const next = new Set(prev);
      if (allFilteredSelected) {
        filteredIndices.forEach((i) => next.delete(i));
      } else {
        filteredIndices.forEach((i) => next.add(i));
      }
      return next;
    });
  }, [candidates, filteredIndices]);

  useEffect(() => {
    const el = slideRefs.current[stepIndex];
    if (!el) return;
    setSliderHeight(el.scrollHeight);
    const obs = new ResizeObserver(() => setSliderHeight(el.scrollHeight));
    obs.observe(el);
    return () => obs.disconnect();
  }, [stepIndex, candidates, maceResults]);

  useEffect(() => {
    if (!mainRef.current) return;
    const top = mainRef.current.getBoundingClientRect().top + window.scrollY - 56;
    window.scrollTo({ top, behavior: "smooth" });
  }, [stepIndex, candidates]);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-white text-[var(--foreground)]">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-slate-100 bg-white/90 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/logo.png"
                alt="Matter of Fact"
                className="h-[40px] object-contain"
              />
            </div>
          </div>
        </div>
      </header>

      {/* Floating elements overlay */}
      <FloatingElements
        assembled={assembled}
        onLanded={handleOverlayLanded}
        onDisassembled={handleDisassembled}
      />

      {/* Hero */}
      <HeroSection onGetStarted={scrollToPeriodicTable} />

      {/* Step tracker + Element map — fades in as tiles land */}
      <div
        style={{
          opacity: canvasRevealed ? 1 : 0,
          transition: "opacity 0.35s ease",
          pointerEvents: canvasRevealed ? "auto" : "none",
        }}
      >

      <div className="mx-auto max-w-6xl px-4 pt-6 sm:px-6">
        <StepTracker
          step={step}
          stepIndex={stepIndex}
          canOpenCandidates={Boolean(candidates)}
          canOpenValidation={Boolean(maceResults)}
          canOpenViewer={viewerVisited}
          onOpenCandidates={() => {
            if (candidates) setStep("candidates");
          }}
          onOpenValidation={() => {
            if (maceResults) setStep("validation");
          }}
          onOpenViewer={() => {
            if (viewerVisited) setStep("viewer");
          }}
        />
      </div>

      {/* Element map */}
      <ElementMap onGenerate={runGenerate} isGenerating={genLoading} />

      </div>{/* end fade-in wrapper */}

      {/* Main workspace — only shown once candidates exist */}
      {candidates && (
      <main ref={mainRef} className="relative mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <div className="mb-6">
          <StepTracker
            step={step}
            stepIndex={stepIndex}
            canOpenCandidates={Boolean(candidates)}
            canOpenValidation={Boolean(maceResults)}
            canOpenViewer={viewerVisited}
            onOpenCandidates={() => {
              if (candidates) setStep("candidates");
            }}
            onOpenValidation={() => {
              if (maceResults) setStep("validation");
            }}
            onOpenViewer={() => {
              if (viewerVisited) setStep("viewer");
            }}
          />
        </div>

        <div className="relative">
          {step !== "candidates" && (
            <button
              type="button"
              onClick={goBack}
              aria-label="Go back"
              className="absolute -top-6 left-0 flex items-center gap-1 text-[11px] text-slate-400 transition hover:text-[var(--foreground)]"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </button>
          )}
        </div>

        {/* Sliding panels */}
        <div
          className="overflow-hidden transition-[height] duration-500 ease-in-out"
          style={{ height: sliderHeight }}
        >
          <div
            className="flex items-start transition-transform duration-500 ease-in-out will-change-transform"
            style={{ transform: `translateX(-${stepIndex * 100}%)` }}
          >
            {/* Step 1: Candidate Generation Results */}
            <div
              ref={(el) => { slideRefs.current[0] = el; }}
              className="w-full min-w-full shrink-0"
            >
              {candidates && (
                <div className="space-y-4">
                  <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
                          Generated candidates for {elementA}&ndash;{elementB}
                        </h2>
                        <p className="mt-1 text-sm text-slate-500">
                          <strong>{candidates.length}</strong> candidates generated &middot;{" "}
                          <strong>{candidates.filter((c) => c.predicted_stable).length}</strong>{" "}
                          predicted near/on hull
                        </p>
                      </div>
                    </div>

                    <div className="mt-5">
                      <HullChart
                        data={candidateHullData}
                        mpData={mpHullData}
                        elementA={elementA}
                        elementB={elementB}
                        title={`${elementA}\u2013${elementB} convex hull`}
                      />
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-sm font-semibold tracking-[-0.01em] text-[var(--foreground)]">
                          Select candidates for MACE validation
                        </h3>
                        <p className="mt-1 text-xs text-slate-400">
                          {selected.size} of {candidates.length} selected
                          {crystalSystemFilter.size > 0 && (
                            <span> &middot; showing {filteredCandidates.length} of {candidates.length}</span>
                          )}
                        </p>
                      </div>
                    </div>

                    {uniqueCrystalSystems.length > 0 && (
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-slate-400">
                          Crystal system
                        </span>
                        {uniqueCrystalSystems.map((sys) => (
                          <button
                            key={sys}
                            type="button"
                            onClick={() => toggleCrystalSystem(sys)}
                            className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                              crystalSystemFilter.has(sys)
                                ? "border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--accent-dark)]"
                                : "border-[var(--border)] bg-[var(--input-bg)] text-[var(--foreground)] hover:border-[var(--accent)]/35 hover:bg-[var(--accent-dim)]"
                            }`}
                          >
                            {sys}
                          </button>
                        ))}
                        {crystalSystemFilter.size > 0 && (
                          <button
                            type="button"
                            onClick={() => setCrystalSystemFilter(new Set())}
                            className="text-xs font-medium text-slate-400 underline-offset-4 hover:text-[var(--foreground)] hover:underline"
                          >
                            Clear filter
                          </button>
                        )}
                      </div>
                    )}

                    <div className="mt-4">
                      <CandidateTable
                        candidates={filteredCandidates}
                        selected={selected}
                        originalIndices={filteredIndices}
                        onToggle={toggleSelect}
                        onToggleAll={toggleSelectAll}
                      />
                    </div>
                  </div>

                  <div className="pt-2">
                    <button
                      type="button"
                      onClick={runValidation}
                      disabled={selected.size === 0 || valLoading}
                      className="inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[var(--accent)]/20 transition hover:bg-[var(--accent-dark)] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {valLoading ? (
                        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                      ) : (
                        <Search className="h-4 w-4" aria-hidden />
                      )}
                      Validate {selected.size} selected with MACE
                    </button>
                  </div>
                </div>
              )}
              {error && step === "candidates" && (
                <div role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {error}
                </div>
              )}
            </div>

            {/* Step 2: MACE Validation Results */}
            <div
              ref={(el) => { slideRefs.current[1] = el; }}
              className="w-full min-w-full shrink-0"
            >
              {maceResults && (
                <div className="space-y-4">
                  <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
                          MACE validation results
                        </h2>
                        <p className="mt-1 text-sm text-slate-500">
                          <strong>{maceResults.length}</strong> structures validated &middot;{" "}
                          <strong>{maceResults.filter((r) => r.mace_stable).length}</strong>{" "}
                          confirmed stable by MACE
                        </p>
                      </div>
                    </div>

                    <div className="mt-5">
                      <HullChart
                        data={maceHullData}
                        mpData={mpHullData}
                        elementA={elementA}
                        elementB={elementB}
                        title={`${elementA}\u2013${elementB} MACE convex hull`}
                      />
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
                    <h3 className="text-sm font-semibold tracking-[-0.01em] text-[var(--foreground)]">
                      Detailed results
                    </h3>
                    <div className="mt-4">
                      <ResultsTable results={maceResults} />
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3 pt-2">
                    <button
                      type="button"
                      onClick={downloadCsv}
                      className="inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[var(--accent)]/20 transition hover:bg-[var(--accent-dark)]"
                    >
                      <Download className="h-4 w-4" aria-hidden />
                      Download results as CSV
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setViewerVisited(true);
                        setStep("viewer");
                      }}
                      className="inline-flex items-center justify-center gap-2 rounded-xl border border-[var(--border)] bg-white px-5 py-2.5 text-sm font-semibold text-[var(--foreground)] shadow-sm transition hover:border-[var(--accent)]/35 hover:bg-[var(--accent-dim)]"
                    >
                      <Eye className="h-4 w-4" aria-hidden />
                      View Structures
                    </button>
                  </div>
                </div>
              )}
              {error && step === "validation" && (
                <div role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {error}
                </div>
              )}
            </div>

            {/* Step 3: Crystal Viewer */}
            <div
              ref={(el) => { slideRefs.current[2] = el; }}
              className="w-full min-w-full shrink-0"
            >
              {viewerVisited && (
                <CrystalViewer elementA={elementA} elementB={elementB} />
              )}
            </div>
          </div>
        </div>
      </main>
      )}

      {/* Footer */}
      <footer className="relative mt-16 border-t border-slate-100 py-8 text-center text-xs text-slate-400">
        Matter of Fact &middot; MIT MTL Hackathon 2026
      </footer>
    </div>
  );
}
