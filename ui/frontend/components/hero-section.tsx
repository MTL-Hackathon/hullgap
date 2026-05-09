"use client";

import { useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import { HeroSphere } from "./hero-sphere";

interface Props {
  onGetStarted: () => void;
}

export function HeroSection({ onGetStarted }: Props) {
  const heroRef = useRef<HTMLElement>(null);
  const [heroMouse, setHeroMouse] = useState({ x: 0, y: 0, hovered: false });

  return (
    <section
      ref={heroRef}
      className="relative flex min-h-[calc(100vh-49px)] flex-col items-center justify-center overflow-hidden px-4 py-20 text-center"
      onMouseMove={(e) => {
        if (!heroRef.current) return;
        const r = heroRef.current.getBoundingClientRect();
        setHeroMouse({
          x: ((e.clientX - r.left) / r.width) * 2 - 1,
          y: -((e.clientY - r.top) / r.height) * 2 + 1,
          hovered: true,
        });
      }}
      onMouseLeave={() => setHeroMouse({ x: 0, y: 0, hovered: false })}
    >
      <div className="pointer-events-none absolute inset-0" style={{ opacity: 0.45 }}>
        <HeroSphere
          mouseX={heroMouse.x}
          mouseY={heroMouse.y}
          isHovered={heroMouse.hovered}
        />
      </div>
      <div className="relative z-10 flex flex-col items-center">
        <div className="mb-6 flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/logo.png"
            alt="Project BROT"
            className="h-20 w-20 rounded-xl object-contain"
          />
        </div>
        <div className="mb-5 inline-flex items-center gap-1.5 rounded-full border border-[var(--accent)]/15 bg-[var(--accent-dim)] px-3 py-1 text-xs tracking-wide text-[var(--accent-dark)]">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
          Crystal structure discovery
        </div>
        <h1 className="max-w-2xl text-[44px] font-semibold tracking-[-0.03em] text-[var(--foreground)] sm:text-[68px] sm:leading-[0.93]">
          <span
            className="block transition-transform duration-700 ease-out"
            style={{
              transform: `translate(${heroMouse.x * -18}px, ${heroMouse.y * -10}px)`,
            }}
          >
            Project
          </span>
          <span
            className="block text-[var(--accent)] transition-transform duration-700 ease-out"
            style={{
              transform: `translate(${heroMouse.x * -8}px, ${heroMouse.y * -5}px)`,
            }}
          >
            BROT
          </span>
        </h1>
        <p className="mt-2 text-sm font-medium tracking-widest text-slate-400 uppercase">
          Beyond DFT Optimization Toolkit
        </p>
        <p className="mt-5 text-[18px] font-light tracking-[-0.01em] text-slate-500">
          Discover stable crystal structures with machine-learned interatomic potentials.
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
