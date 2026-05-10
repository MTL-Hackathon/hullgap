import { NextRequest, NextResponse } from "next/server";

export const dynamic: "force-static" | "auto" =
  process.env.NEXT_PUBLIC_STATIC_EXPORT === "true" ? "force-static" : "auto";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const res = await fetch(`${BACKEND_URL}/structure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: text || "Backend error" },
        { status: res.status }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to reach backend" },
      { status: 502 }
    );
  }
}
