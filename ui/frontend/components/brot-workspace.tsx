"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ArrowLeft, Loader2, Search, Download, FlaskConical, Eye } from "lucide-react";
import { generateCandidates, validateWithMace } from "@/lib/api-client";
import type { CandidateResult, MaceResult } from "@/lib/types";
import { HeroSection } from "./hero-section";
import { StepTracker, type Step } from "./step-tracker";
import { CandidateTable } from "./candidate-table";
import { ResultsTable } from "./results-table";
import { HullChart } from "./hull-chart";
import { CrystalViewer } from "./crystal-viewer";

const ELEMENTS = [
  "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr",
  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I", "Xe",
  "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po",
];

const PRESETS = [
  { label: "Co + Bi (50)", elA: "Co", elB: "Bi", n: 50 },
  { label: "Fe + Ti (100)", elA: "Fe", elB: "Ti", n: 100 },
  { label: "Ni + Al (80)", elA: "Ni", elB: "Al", n: 80 },
];

export function BrotWorkspace() {
  const [elementA, setElementA] = useState("Co");
  const [elementB, setElementB] = useState("Bi");
  const [nCandidates, setNCandidates] = useState(50);
  const [step, setStep] = useState<Step>("input");
  const [candidates, setCandidates] = useState<CandidateResult[] | null>(null);
  const [maceResults, setMaceResults] = useState<MaceResult[] | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [genLoading, setGenLoading] = useState(false);
  const [valLoading, setValLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sliderHeight, setSliderHeight] = useState<number | undefined>(undefined);

  const [viewerVisited, setViewerVisited] = useState(false);

  const mainRef = useRef<HTMLElement>(null);
  const slideRefs = useRef<(HTMLDivElement | null)[]>([null, null, null, null]);

  const stepIndex = step === "input" ? 0 : step === "candidates" ? 1 : step === "validation" ? 2 : 3;
  const canGenerate = elementA !== elementB;

  const goBack = useCallback(() => {
    if (step === "candidates") setStep("input");
    else if (step === "validation") setStep("candidates");
    else if (step === "viewer") setStep("validation");
  }, [step]);

  const scrollToMain = useCallback(() => {
    if (!mainRef.current) return;
    const top = mainRef.current.getBoundingClientRect().top + window.scrollY - 56;
    window.scrollTo({ top, behavior: "smooth" });
  }, []);

  const runGenerate = useCallback(async () => {
    setError(null);
    setGenLoading(true);
    setCandidates(null);
    setMaceResults(null);
    setSelected(new Set());
    try {
      const res = await generateCandidates(elementA, elementB, nCandidates);
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
  }, [elementA, elementB, nCandidates]);

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
    setStep("input");
    setCandidates(null);
    setMaceResults(null);
    setSelected(new Set());
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

  const toggleSelectAll = useCallback(() => {
    if (!candidates) return;
    setSelected((prev) => {
      if (prev.size === candidates.length) return new Set();
      return new Set(candidates.map((_, i) => i));
    });
  }, [candidates]);

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
    a.download = `brot_mace_results_${elementA}-${elementB}.csv`;
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

  const maceHullData = useMemo(() => {
    if (!maceResults) return [];
    return maceResults.map((r) => ({
      x_B: r.x_B,
      energy: r.mace_energy_eV_atom,
      formula: r.formula,
      stable: r.mace_stable,
    }));
  }, [maceResults]);

  useEffect(() => {
    const el = slideRefs.current[stepIndex];
    if (!el) return;
    setSliderHeight(el.scrollHeight);
    const obs = new ResizeObserver(() => setSliderHeight(el.scrollHeight));
    obs.observe(el);
    return () => obs.disconnect();
  }, [stepIndex, candidates, maceResults]);

  useEffect(() => {
    if (stepIndex === 0) return;
    if (!mainRef.current) return;
    const top = mainRef.current.getBoundingClientRect().top + window.scrollY - 56;
    window.scrollTo({ top, behavior: "smooth" });
  }, [stepIndex]);

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
                alt="Project BROT"
                className="h-[42px] w-[42px] rounded-lg object-contain"
              />
              <p className="text-[20px] font-medium tracking-[-0.02em] text-[var(--foreground)]">
                <span className="font-semibold">Project</span>{" "}
                <span className="font-bold text-[var(--accent)]">BROT</span>
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-400">
              <span>Beyond DFT Optimization Toolkit</span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero */}
      <HeroSection onGetStarted={scrollToMain} />

      {/* Main workspace */}
      <main ref={mainRef} className="relative mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <div className="relative">
          {step !== "input" && (
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
          <StepTracker
            step={step}
            stepIndex={stepIndex}
            canOpenCandidates={Boolean(candidates)}
            canOpenValidation={Boolean(maceResults)}
            canOpenViewer={viewerVisited}
            onOpenInput={() => setStep("input")}
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

        {/* Sliding panels */}
        <div
          className="overflow-hidden transition-[height] duration-500 ease-in-out"
          style={{ height: sliderHeight }}
        >
          <div
            className="flex items-start transition-transform duration-500 ease-in-out will-change-transform"
            style={{ transform: `translateX(-${stepIndex * 100}%)` }}
          >
            {/* Step 1: Element Selection */}
            <div
              ref={(el) => { slideRefs.current[0] = el; }}
              className="w-full min-w-full shrink-0"
            >
              <div className="w-full rounded-2xl border border-[var(--border)] bg-white p-5 shadow-[var(--shadow)] sm:p-6">
                <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
                  Select elements & candidates
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Choose two elements and the number of candidate crystal structures to generate.
                </p>

                <div className="mt-5 grid gap-4 sm:grid-cols-3">
                  <label className="text-sm text-slate-600">
                    Element A
                    <select
                      value={elementA}
                      onChange={(e) => setElementA(e.target.value)}
                      className="mt-1 h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-3 text-sm text-[var(--foreground)]"
                    >
                      {ELEMENTS.map((el) => (
                        <option key={el} value={el}>{el}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm text-slate-600">
                    Element B
                    <select
                      value={elementB}
                      onChange={(e) => setElementB(e.target.value)}
                      className="mt-1 h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-3 text-sm text-[var(--foreground)]"
                    >
                      {ELEMENTS.map((el) => (
                        <option key={el} value={el}>{el}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm text-slate-600">
                    Number of candidates
                    <input
                      type="number"
                      min={10}
                      max={200}
                      step={10}
                      value={nCandidates}
                      onChange={(e) => setNCandidates(Number(e.target.value) || 50)}
                      className="mt-1 h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-3 text-sm text-[var(--foreground)]"
                    />
                  </label>
                </div>

                {elementA === elementB && (
                  <p className="mt-3 text-sm text-amber-600">
                    Please choose two different elements.
                  </p>
                )}

                <div className="mt-4 flex flex-wrap gap-2">
                  <span className="w-full text-xs font-medium text-slate-400">
                    Example presets
                  </span>
                  {PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => {
                        setElementA(p.elA);
                        setElementB(p.elB);
                        setNCandidates(p.n);
                      }}
                      className="rounded-full border border-[var(--border)] bg-[var(--input-bg)] px-3 py-1.5 text-xs font-medium text-[var(--foreground)] transition hover:border-[var(--accent)]/35 hover:bg-[var(--accent-dim)]"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>

                <div className="mt-5 flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    disabled={!canGenerate || genLoading}
                    onClick={runGenerate}
                    className="inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[var(--accent)]/20 transition hover:bg-[var(--accent-dark)] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {genLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                    ) : (
                      <FlaskConical className="h-4 w-4" aria-hidden />
                    )}
                    Generate Candidates
                  </button>
                  {(candidates || maceResults) && (
                    <button
                      type="button"
                      onClick={reset}
                      className="text-sm font-medium text-slate-400 underline-offset-4 hover:text-[var(--foreground)] hover:underline"
                    >
                      Start over
                    </button>
                  )}
                </div>
              </div>
              {error && step === "input" && (
                <div role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {error}
                </div>
              )}
            </div>

            {/* Step 2: Candidate Generation Results */}
            <div
              ref={(el) => { slideRefs.current[1] = el; }}
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
                        </p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <CandidateTable
                        candidates={candidates}
                        selected={selected}
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

            {/* Step 3: MACE Validation Results */}
            <div
              ref={(el) => { slideRefs.current[2] = el; }}
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

            {/* Step 4: Crystal Viewer */}
            <div
              ref={(el) => { slideRefs.current[3] = el; }}
              className="w-full min-w-full shrink-0"
            >
              {viewerVisited && (
                <CrystalViewer elementA={elementA} elementB={elementB} />
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative mt-16 border-t border-slate-100 py-8 text-center text-xs text-slate-400">
        Project BROT &mdash; Beyond DFT Optimization Toolkit &middot; MIT MTL Hackathon 2026
      </footer>
    </div>
  );
}
