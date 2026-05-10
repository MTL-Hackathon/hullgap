"use client";

export type Step = "elements" | "candidates" | "validation" | "viewer";

const TOTAL_STEPS = 3;

interface StepItem {
  n: string;
  title: string;
  active: boolean;
  locked: boolean;
  onClick: () => void;
}

export function StepTracker({
  step,
  canOpenElements,
  canOpenCandidates,
  canOpenValidation,
  onOpenElements,
  onOpenCandidates,
  onOpenValidation,
}: {
  step: Step;
  canOpenElements: boolean;
  canOpenCandidates: boolean;
  canOpenValidation: boolean;
  onOpenElements: () => void;
  onOpenCandidates: () => void;
  onOpenValidation: () => void;
}) {
  const items: StepItem[] = [
    {
      n: "01",
      title: "Element Mapping",
      active: step === "elements",
      locked: !canOpenElements,
      onClick: onOpenElements,
    },
    {
      n: "02",
      title: "Candidate Generation",
      active: step === "candidates",
      locked: !canOpenCandidates,
      onClick: onOpenCandidates,
    },
    {
      n: "03",
      title: "DFT Verification",
      active: step === "validation" || step === "viewer",
      locked: !canOpenValidation,
      onClick: onOpenValidation,
    },
  ];

  const stepIndex = step === "elements" ? 0 : step === "candidates" ? 1 : 2;

  return (
    <ol className="relative grid w-full grid-cols-3 gap-4">
      <div
        aria-hidden
        className="pointer-events-none absolute left-0 right-0 top-[22px] hidden h-px bg-slate-200 sm:block"
      >
        <div
          className="h-full bg-[var(--accent)] transition-[width] duration-500 ease-in-out"
          style={{
            width:
              stepIndex === TOTAL_STEPS - 1
                ? "100%"
                : `calc(${((stepIndex + 1) / TOTAL_STEPS) * 100}% - 10px)`,
          }}
        />
      </div>
      {items.map((item) => (
        <li key={item.n} className="relative">
          <button
            type="button"
            disabled={item.locked}
            onClick={item.onClick}
            className={`text-left font-medium tracking-[-0.01em] transition ${
              item.locked ? "cursor-not-allowed" : "cursor-pointer"
            }`}
          >
            <span
              className={`block text-[11px] tracking-[0.12em] ${
                item.active ? "text-[var(--accent)]" : "text-slate-400"
              }`}
            >
              {item.n}
            </span>
            <span
              className={`mt-4 block text-sm ${
                item.active
                  ? "text-[var(--foreground)]"
                  : item.locked
                    ? "text-slate-300"
                    : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {item.title}
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}
