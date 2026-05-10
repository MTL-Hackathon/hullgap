import { NextRequest, NextResponse } from "next/server";
import { readFile, readdir } from "fs/promises";
import { join } from "path";
import { parseCif } from "@/lib/cif";

/** Literal required by Turbopack; satisfies `output: "export"` route collection. */
export const dynamic = "force-static";

const DATA_DIR = join(process.cwd(), "..", "..", "data", "mattergen");

/**
 * Loads the actual relaxed CIF for a given system + idx and returns it in
 * the StructureData shape the viewer expects. Implemented entirely in
 * Node, with no dependency on the Python FastAPI backend, so the viewer
 * works as long as the Next.js app is running.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { system, idx } = body as { system?: string; idx?: number };

    if (!system || typeof idx !== "number" || !Number.isFinite(idx)) {
      return NextResponse.json(
        { error: "system (string) and idx (number) are required" },
        { status: 400 }
      );
    }

    const relaxedDir = join(DATA_DIR, system, "relaxed");
    let files: string[];
    try {
      files = await readdir(relaxedDir);
    } catch {
      return NextResponse.json(
        { error: `No relaxed/ directory for system ${system}` },
        { status: 404 }
      );
    }

    const idxStr = idx.toString().padStart(3, "0");
    const prefix = `${system}_${idxStr}_`;
    const match = files.find((f) => f.startsWith(prefix) && f.endsWith(".cif"));
    if (!match) {
      return NextResponse.json(
        { error: `No CIF found for ${system} idx=${idx}` },
        { status: 404 }
      );
    }

    const cifPath = join(relaxedDir, match);
    const text = await readFile(cifPath, "utf-8");
    const parsed = parseCif(text, match.replace(/\.cif$/, ""));

    return NextResponse.json(parsed);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to read CIF" },
      { status: 500 }
    );
  }
}
