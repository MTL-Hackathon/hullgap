"use client";

import { ArrowDown } from "lucide-react";

interface Props {
  onGetStarted: () => void;
}

export function HeroSection({ onGetStarted }: Props) {
  return (
    <section
      className="relative z-[20] flex min-h-[calc(100vh-49px)] flex-col items-center justify-center overflow-hidden px-4 pb-10 pt-20 text-center"
    >
      <div className="relative z-10 flex flex-col items-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/mf-icon.png"
          alt="Mf"
          className="mb-8 h-28 w-28 rounded-2xl object-contain sm:h-36 sm:w-36"
        />
        <h1 className="max-w-2xl text-[44px] font-semibold tracking-[-0.03em] text-[var(--foreground)] sm:text-[68px] sm:leading-[0.93]">
          <span className="block">
            Matter
          </span>
          <span className="block text-[var(--accent)]">
            of Fact
          </span>
        </h1>
        <p className="mt-5 text-[18px] font-light tracking-[-0.01em] text-slate-500">
          Discover stable crystal structures with machine-learned interatomic potentials for microelectronics.
        </p>
        <button
          type="button"
          onClick={onGetStarted}
          className="mt-10 inline-flex items-center gap-2.5 rounded-2xl bg-[var(--accent)] px-7 py-3.5 text-[15px] font-semibold text-white shadow-xl shadow-[var(--accent)]/20 transition hover:bg-[var(--accent-dark)] active:scale-[0.98]"
        >
          Get started
          <ArrowDown className="h-4 w-4" />
        </button>
      </div>
    </section>
  );
}
