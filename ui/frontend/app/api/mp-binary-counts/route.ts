import { NextRequest, NextResponse } from "next/server";

export const dynamic: "force-static" | "auto" =
  process.env.NEXT_PUBLIC_STATIC_EXPORT === "true" ? "force-static" : "auto";

const MP_API_KEY = process.env.MP_API_KEY || "LvVhu7zKRmTIDG60frvRksYdC0q4W0C3";
const MP_BASE = "https://api.materialsproject.org";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const el = searchParams.get("el");

  if (!el) {
    return NextResponse.json({ error: "el query param required" }, { status: 400 });
  }

  try {
    const url = new URL(`${MP_BASE}/materials/summary/`);
    url.searchParams.set("elements", el);
    url.searchParams.set("nelements", "2");
    url.searchParams.set("_fields", "material_id,elements,is_stable");
    url.searchParams.set("_limit", "500");

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
    const counts: Record<string, { total: number; stable: number }> = {};

    for (const entry of json.data ?? []) {
      const elems: string[] = entry.elements ?? [];
      const partner = elems.find((e: string) => e !== el);
      if (!partner) continue;

      if (!counts[partner]) counts[partner] = { total: 0, stable: 0 };
      counts[partner].total++;
      if (entry.is_stable) counts[partner].stable++;
    }

    return NextResponse.json({ el, counts });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to reach MP API" },
      { status: 502 },
    );
  }
}
