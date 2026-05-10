import { NextRequest, NextResponse } from "next/server";

const MP_API_KEY = process.env.MP_API_KEY || "LvVhu7zKRmTIDG60frvRksYdC0q4W0C3";
const MP_BASE = "https://api.materialsproject.org";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const elA = searchParams.get("a");
  const elB = searchParams.get("b");

  if (!elA || !elB) {
    return NextResponse.json({ error: "a and b query params required" }, { status: 400 });
  }

  try {
    const chemsys = [elA, elB].sort().join("-");

    const url = new URL(`${MP_BASE}/materials/summary/`);
    url.searchParams.set("chemsys", chemsys);
    url.searchParams.set(
      "_fields",
      "material_id,formula_pretty,symmetry,energy_above_hull,formation_energy_per_atom,volume,nsites,density,is_stable,structure",
    );
    url.searchParams.set("_limit", "50");

    const res = await fetch(url.toString(), {
      headers: { "X-API-KEY": MP_API_KEY },
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: `MP API error ${res.status}: ${text.slice(0, 200)}` },
        { status: res.status },
      );
    }

    const json = await res.json();
    const entries = (json.data ?? []).map((d: Record<string, unknown>) => {
      const struct = d.structure as Record<string, unknown> | undefined;
      const lattice = struct?.lattice as Record<string, number> | undefined;
      return {
        id: d.material_id,
        formula: d.formula_pretty,
        spacegroup: (d.symmetry as Record<string, unknown>)?.symbol ?? "—",
        crystal_system: (d.symmetry as Record<string, unknown>)?.crystal_system ?? "—",
        e_above_hull: d.energy_above_hull,
        formation_energy: d.formation_energy_per_atom,
        density: d.density,
        n_sites: d.nsites,
        volume: d.volume,
        is_stable: d.is_stable,
        a: lattice?.a ?? null,
        b: lattice?.b ?? null,
        c: lattice?.c ?? null,
        alpha: lattice?.alpha ?? null,
        beta: lattice?.beta ?? null,
        gamma: lattice?.gamma ?? null,
      };
    });

    entries.sort(
      (a: { e_above_hull: number }, b: { e_above_hull: number }) =>
        (a.e_above_hull ?? 99) - (b.e_above_hull ?? 99),
    );

    return NextResponse.json({
      chemsys,
      count: entries.length,
      phases: entries,
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to reach MP API" },
      { status: 502 },
    );
  }
}
