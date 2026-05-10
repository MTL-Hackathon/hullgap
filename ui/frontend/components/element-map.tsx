"use client";

import { useCallback, useEffect, useRef, useState, useMemo } from "react";

// ─── Types ──────────────────────────────────────────────────────────────
type Cat =
  | "alkali" | "alkaline" | "tm" | "post-tm"
  | "metalloid" | "nonmetal" | "noble"
  | "lanthanide" | "actinide";

type Vec3 = { x: number; y: number; z: number };

// ─── 3D projection ──────────────────────────────────────────────────────
const FOCAL = 900;

function projectPoint(
  p: Vec3,
  rot: { x: number; y: number },
  cx: number,
  cy: number,
): { x: number; y: number; s: number } {
  const dx = p.x - cx, dy = p.y - cy, dz = p.z;
  const cosX = Math.cos(rot.x), sinX = Math.sin(rot.x);
  const cosY = Math.cos(rot.y), sinY = Math.sin(rot.y);
  const x1 =  dx * cosY + dz * sinY;
  const z1 = -dx * sinY + dz * cosY;
  const y2 =  dy * cosX - z1 * sinX;
  const z2 =  dy * sinX + z1 * cosX;
  const s  = FOCAL / (FOCAL + z2);
  return { x: cx + x1 * s, y: cy + y2 * s, s };
}

function zoomedPoint(p: Vec3, zoom: number, cx: number, cy: number): Vec3 {
  return {
    x: cx + (p.x - cx) * zoom,
    y: cy + (p.y - cy) * zoom,
    z: p.z * zoom,
  };
}

// Connection line styling: count → { width, colour }
function connStyle(count: number): { width: number; color: string } {
  if (count >= 15) return { width: 3.5, color: "rgba(40,40,40,0.70)" };
  if (count >= 9)  return { width: 2.5, color: "rgba(70,70,70,0.55)" };
  if (count >= 5)  return { width: 1.8, color: "rgba(100,100,100,0.40)" };
  if (count >= 3)  return { width: 1.2, color: "rgba(140,140,140,0.28)" };
  return { width: 0.8, color: "rgba(180,180,180,0.18)" };
}

interface El {
  z: number; s: string; n: string; en: number; m: number;
  cat: Cat; col: number; row: number; relevant: boolean;
  rad: number;
}

// ─── Microelectronics-relevant elements ─────────────────────────────────
const RELEVANT = new Set([
  "H","B","C","N","O","F",
  "Mg","Al","Si","P","S","Cl",
  "Ti","Cr","Co","Ni","Cu","Zn",
  "Ga","Ge","As","Se",
  "Sr","Y","Zr","Nb","Mo","Ru","Pd","Ag",
  "In","Sn","Sb","Te",
  "Ba","La",
  "Hf","Ta","W","Ir","Pt","Au","Pb","Bi",
]);

// ─── Element data ───────────────────────────────────────────────────────
// [Z, symbol, name, EN(Pauling), mass, category, col(0-17), row(0-9)]
// rows 0-6 = periods 1-7,  row 8 = lanthanides,  row 9 = actinides
const RAW: [number,string,string,number,number,Cat,number,number][] = [
  // Period 1
  [1,"H","Hydrogen",2.20,1.01,"nonmetal",0,0],
  [2,"He","Helium",0,4.00,"noble",17,0],
  // Period 2
  [3,"Li","Lithium",0.98,6.94,"alkali",0,1],
  [4,"Be","Beryllium",1.57,9.01,"alkaline",1,1],
  [5,"B","Boron",2.04,10.81,"metalloid",12,1],
  [6,"C","Carbon",2.55,12.01,"nonmetal",13,1],
  [7,"N","Nitrogen",3.04,14.01,"nonmetal",14,1],
  [8,"O","Oxygen",3.44,16.00,"nonmetal",15,1],
  [9,"F","Fluorine",3.98,19.00,"nonmetal",16,1],
  [10,"Ne","Neon",0,20.18,"noble",17,1],
  // Period 3
  [11,"Na","Sodium",0.93,22.99,"alkali",0,2],
  [12,"Mg","Magnesium",1.31,24.31,"alkaline",1,2],
  [13,"Al","Aluminium",1.61,26.98,"post-tm",12,2],
  [14,"Si","Silicon",1.90,28.09,"metalloid",13,2],
  [15,"P","Phosphorus",2.19,30.97,"nonmetal",14,2],
  [16,"S","Sulfur",2.58,32.07,"nonmetal",15,2],
  [17,"Cl","Chlorine",3.16,35.45,"nonmetal",16,2],
  [18,"Ar","Argon",0,39.95,"noble",17,2],
  // Period 4
  [19,"K","Potassium",0.82,39.10,"alkali",0,3],
  [20,"Ca","Calcium",1.00,40.08,"alkaline",1,3],
  [21,"Sc","Scandium",1.36,44.96,"tm",2,3],
  [22,"Ti","Titanium",1.54,47.87,"tm",3,3],
  [23,"V","Vanadium",1.63,50.94,"tm",4,3],
  [24,"Cr","Chromium",1.66,52.00,"tm",5,3],
  [25,"Mn","Manganese",1.55,54.94,"tm",6,3],
  [26,"Fe","Iron",1.83,55.85,"tm",7,3],
  [27,"Co","Cobalt",1.88,58.93,"tm",8,3],
  [28,"Ni","Nickel",1.91,58.69,"tm",9,3],
  [29,"Cu","Copper",1.90,63.55,"tm",10,3],
  [30,"Zn","Zinc",1.65,65.38,"tm",11,3],
  [31,"Ga","Gallium",1.81,69.72,"post-tm",12,3],
  [32,"Ge","Germanium",2.01,72.63,"metalloid",13,3],
  [33,"As","Arsenic",2.18,74.92,"metalloid",14,3],
  [34,"Se","Selenium",2.55,78.97,"nonmetal",15,3],
  [35,"Br","Bromine",2.96,79.90,"nonmetal",16,3],
  [36,"Kr","Krypton",3.00,83.80,"noble",17,3],
  // Period 5
  [37,"Rb","Rubidium",0.82,85.47,"alkali",0,4],
  [38,"Sr","Strontium",0.95,87.62,"alkaline",1,4],
  [39,"Y","Yttrium",1.22,88.91,"tm",2,4],
  [40,"Zr","Zirconium",1.33,91.22,"tm",3,4],
  [41,"Nb","Niobium",1.60,92.91,"tm",4,4],
  [42,"Mo","Molybdenum",2.16,95.95,"tm",5,4],
  [43,"Tc","Technetium",1.90,97.91,"tm",6,4],
  [44,"Ru","Ruthenium",2.20,101.1,"tm",7,4],
  [45,"Rh","Rhodium",2.28,102.9,"tm",8,4],
  [46,"Pd","Palladium",2.20,106.4,"tm",9,4],
  [47,"Ag","Silver",1.93,107.9,"tm",10,4],
  [48,"Cd","Cadmium",1.69,112.4,"tm",11,4],
  [49,"In","Indium",1.78,114.8,"post-tm",12,4],
  [50,"Sn","Tin",1.96,118.7,"post-tm",13,4],
  [51,"Sb","Antimony",2.05,121.8,"metalloid",14,4],
  [52,"Te","Tellurium",2.10,127.6,"metalloid",15,4],
  [53,"I","Iodine",2.66,126.9,"nonmetal",16,4],
  [54,"Xe","Xenon",2.60,131.3,"noble",17,4],
  // Period 6 (col 2 empty — La in lanthanide row)
  [55,"Cs","Caesium",0.79,132.9,"alkali",0,5],
  [56,"Ba","Barium",0.89,137.3,"alkaline",1,5],
  [72,"Hf","Hafnium",1.30,178.5,"tm",3,5],
  [73,"Ta","Tantalum",1.50,181.0,"tm",4,5],
  [74,"W","Tungsten",2.36,183.8,"tm",5,5],
  [75,"Re","Rhenium",1.90,186.2,"tm",6,5],
  [76,"Os","Osmium",2.20,190.2,"tm",7,5],
  [77,"Ir","Iridium",2.20,192.2,"tm",8,5],
  [78,"Pt","Platinum",2.28,195.1,"tm",9,5],
  [79,"Au","Gold",2.54,197.0,"tm",10,5],
  [80,"Hg","Mercury",2.00,200.6,"tm",11,5],
  [81,"Tl","Thallium",1.62,204.4,"post-tm",12,5],
  [82,"Pb","Lead",1.87,207.2,"post-tm",13,5],
  [83,"Bi","Bismuth",2.02,209.0,"post-tm",14,5],
  [84,"Po","Polonium",2.00,209,"metalloid",15,5],
  [85,"At","Astatine",2.20,210,"nonmetal",16,5],
  [86,"Rn","Radon",0,222,"noble",17,5],
  // Period 7 (col 2 empty — Ac in actinide row)
  [87,"Fr","Francium",0.70,223,"alkali",0,6],
  [88,"Ra","Radium",0.90,226,"alkaline",1,6],
  [104,"Rf","Rutherfordium",0,267,"tm",3,6],
  [105,"Db","Dubnium",0,268,"tm",4,6],
  [106,"Sg","Seaborgium",0,269,"tm",5,6],
  [107,"Bh","Bohrium",0,270,"tm",6,6],
  [108,"Hs","Hassium",0,277,"tm",7,6],
  [109,"Mt","Meitnerium",0,278,"tm",8,6],
  [110,"Ds","Darmstadtium",0,281,"tm",9,6],
  [111,"Rg","Roentgenium",0,282,"tm",10,6],
  [112,"Cn","Copernicium",0,285,"tm",11,6],
  [113,"Nh","Nihonium",0,286,"post-tm",12,6],
  [114,"Fl","Flerovium",0,289,"post-tm",13,6],
  [115,"Mc","Moscovium",0,290,"post-tm",14,6],
  [116,"Lv","Livermorium",0,293,"post-tm",15,6],
  [117,"Ts","Tennessine",0,294,"nonmetal",16,6],
  [118,"Og","Oganesson",0,294,"noble",17,6],
  // Lanthanides (row 8, cols 2-16)
  [57,"La","Lanthanum",1.10,138.9,"lanthanide",2,8],
  [58,"Ce","Cerium",1.12,140.1,"lanthanide",3,8],
  [59,"Pr","Praseodymium",1.13,140.9,"lanthanide",4,8],
  [60,"Nd","Neodymium",1.14,144.2,"lanthanide",5,8],
  [61,"Pm","Promethium",1.13,145,"lanthanide",6,8],
  [62,"Sm","Samarium",1.17,150.4,"lanthanide",7,8],
  [63,"Eu","Europium",1.20,152.0,"lanthanide",8,8],
  [64,"Gd","Gadolinium",1.20,157.3,"lanthanide",9,8],
  [65,"Tb","Terbium",1.10,158.9,"lanthanide",10,8],
  [66,"Dy","Dysprosium",1.22,162.5,"lanthanide",11,8],
  [67,"Ho","Holmium",1.23,164.9,"lanthanide",12,8],
  [68,"Er","Erbium",1.24,167.3,"lanthanide",13,8],
  [69,"Tm","Thulium",1.25,168.9,"lanthanide",14,8],
  [70,"Yb","Ytterbium",1.10,173.0,"lanthanide",15,8],
  [71,"Lu","Lutetium",1.27,175.0,"lanthanide",16,8],
  // Actinides (row 9, cols 2-16)
  [89,"Ac","Actinium",1.10,227,"actinide",2,9],
  [90,"Th","Thorium",1.30,232.0,"actinide",3,9],
  [91,"Pa","Protactinium",1.50,231.0,"actinide",4,9],
  [92,"U","Uranium",1.38,238.0,"actinide",5,9],
  [93,"Np","Neptunium",1.36,237,"actinide",6,9],
  [94,"Pu","Plutonium",1.28,244,"actinide",7,9],
  [95,"Am","Americium",1.30,243,"actinide",8,9],
  [96,"Cm","Curium",1.30,247,"actinide",9,9],
  [97,"Bk","Berkelium",1.30,247,"actinide",10,9],
  [98,"Cf","Californium",1.30,251,"actinide",11,9],
  [99,"Es","Einsteinium",1.30,252,"actinide",12,9],
  [100,"Fm","Fermium",1.30,257,"actinide",13,9],
  [101,"Md","Mendelevium",1.30,258,"actinide",14,9],
  [102,"No","Nobelium",1.30,259,"actinide",15,9],
  [103,"Lr","Lawrencium",1.30,266,"actinide",16,9],
];

// Empirical atomic radii (pm)
const ATOMIC_RADIUS: Record<number, number> = {
  1:25,2:120,3:145,4:105,5:85,6:70,7:65,8:60,9:50,10:160,
  11:180,12:150,13:125,14:110,15:100,16:100,17:100,18:71,
  19:220,20:180,21:160,22:140,23:135,24:140,25:140,26:140,27:135,28:135,29:135,30:135,
  31:130,32:125,33:115,34:115,35:115,36:88,
  37:235,38:200,39:180,40:155,41:145,42:145,43:135,44:130,45:135,46:140,47:160,48:155,
  49:155,50:145,51:145,52:140,53:140,54:108,
  55:260,56:215,72:155,73:145,74:135,75:135,76:130,77:135,78:135,79:135,80:150,81:190,82:180,83:160,84:190,85:150,86:120,
  87:260,88:215,104:150,105:150,106:150,107:150,108:150,109:150,110:150,111:150,112:150,113:150,114:150,115:150,116:150,117:150,118:150,
  57:195,58:185,59:185,60:185,61:185,62:185,63:185,64:180,65:175,66:175,67:175,68:175,69:175,70:175,71:175,
  89:195,90:180,91:180,92:175,93:175,94:175,95:175,96:175,97:175,98:175,99:175,100:175,101:175,102:175,103:175,
};

const ELEMENTS: El[] = RAW.map(([z,s,n,en,m,cat,col,row]) => ({
  z, s, n, en, m, cat, col, row, relevant: RELEVANT.has(s),
  rad: ATOMIC_RADIUS[z] ?? 150,
}));

const RAD_MIN = 25;
const RAD_MAX = 260;
const SCATTER_2D_SCALE = 0.82;
function scatterR(el: El, cs: number, mode: "2d" | "3d" = "3d"): number {
  const t = (el.rad - RAD_MIN) / (RAD_MAX - RAD_MIN);
  const base = (cs / 2) * (0.55 + t * 0.9);
  return mode === "2d" ? base * SCATTER_2D_SCALE : base;
}

// ─── Color config ───────────────────────────────────────────────────────
const CAT_COLORS: Record<Cat, { bg: string; stroke: string; text: string; fill: string }> = {
  alkali:     { bg: "#fff7ed", stroke: "#fb923c", text: "#c2410c", fill: "#f97316" },
  alkaline:   { bg: "#fffbeb", stroke: "#fbbf24", text: "#b45309", fill: "#f59e0b" },
  tm:         { bg: "#eff6ff", stroke: "#60a5fa", text: "#1d4ed8", fill: "#3b82f6" },
  "post-tm":  { bg: "#f5f3ff", stroke: "#a78bfa", text: "#6d28d9", fill: "#8b5cf6" },
  metalloid:  { bg: "#ecfdf5", stroke: "#34d399", text: "#047857", fill: "#10b981" },
  nonmetal:   { bg: "#fdf4ff", stroke: "#e879f9", text: "#a21caf", fill: "#d946ef" },
  noble:      { bg: "#f1f5f9", stroke: "#cbd5e1", text: "#94a3b8", fill: "#64748b" },
  lanthanide: { bg: "#f0fdfa", stroke: "#2dd4bf", text: "#0f766e", fill: "#14b8a6" },
  actinide:   { bg: "#fef2f2", stroke: "#fca5a5", text: "#b91c1c", fill: "#ef4444" },
};

const CAT_LABELS: Record<string, string> = {
  alkali:     "Alkali metal",
  alkaline:   "Alkaline earth",
  tm:         "Transition metal",
  "post-tm":  "Post-transition metal",
  metalloid:  "Metalloid",
  nonmetal:   "Nonmetal",
  lanthanide: "Lanthanide",
  actinide:   "Actinide",
  noble:      "Noble gas",
};

const LEGEND_GROUPS: Cat[] = [
  "alkali","alkaline","tm","post-tm","metalloid","nonmetal","lanthanide","noble",
];

// ─── Layout ─────────────────────────────────────────────────────────────
const N_COLS    = 18;
const PAD       = 16;
const GAP       = 2;
const SCATTER_H = 560;
const LERP       = 0.11;
const SHAPE_LERP = 0.09;
const INACTIVE_ALPHA = 0.15;

// ─── Group elevation bands on the sphere ────────────────────────────────
// Each chemical group lives on a fixed latitude (polar angle θ from the
// north pole). Within a band, members are spread along the azimuth φ
// using the golden angle so they fan out without overlap.
//
// Coordinate convention (canvas / camera space):
//   +x = right    -x = left
//   +y = down     -y = up        (canvas y grows downward)
//   +z = away     -z = toward camera
//
// θ = 0   → north pole = straight up    (-y)
// θ = π   → south pole = straight down  (+y)
// θ = π/2 → equator (x–z ring)
//
// Bands are ordered roughly by Pauling EN: most electronegative groups
// (nonmetal, noble) near the top pole, least electronegative (alkaline,
// alkali) near the bottom pole. This matches the underlying chemistry
// and makes the sphere feel "right" when rotated.
const GROUP_THETA: Record<Cat, number> = (() => {
  const order: Cat[] = [
    "nonmetal", "noble", "metalloid", "post-tm",
    "tm", "lanthanide", "actinide", "alkaline", "alkali",
  ];
  // Stay clear of the exact poles so sin(θ) doesn't collapse the band.
  const thetaMin = Math.PI * 0.18;
  const thetaMax = Math.PI * 0.82;
  const out = {} as Record<Cat, number>;
  order.forEach((cat, i) => {
    out[cat] = thetaMin + (thetaMax - thetaMin) * (i / (order.length - 1));
  });
  return out;
})();

// Golden angle (~137.5°) — successive members of a band step by this in
// azimuth so adjacent EN-rank atoms don't end up at adjacent longitudes.
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

// ─── 2D scatter angle / spread per group ────────────────────────────────
// Used by the flat (2D) view: each group fans out from the anchor in a
// fixed sector of the plane; spread sets the angular width of that fan.
const GROUP_ANGLE: Record<string, number> = {
  nonmetal:   -Math.PI * 0.50,
  metalloid:  -Math.PI * 0.15,
  "post-tm":   Math.PI * 0.15,
  lanthanide:  Math.PI * 0.50,
  alkaline:    Math.PI * 0.82,
  alkali:     -Math.PI * 0.82,
  tm:          Math.PI,
  actinide:    Math.PI * 0.35,
  noble:      -Math.PI * 0.90,
};
const GROUP_SPREAD: Record<string, number> = {
  nonmetal:  0.70, metalloid: 0.55, "post-tm": 0.60,
  lanthanide: 1.10, alkaline: 0.40, alkali: 0.30,
  tm: 2.20, actinide: 0.50, noble: 0.30,
};

// ─── Helpers ────────────────────────────────────────────────────────────
function gridYFn(row: number, step: number): number {
  if (row <= 6) return PAD + row * step;
  return PAD + (row - 0.5) * step;
}

function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ─── 3D scatter computation ─────────────────────────────────────────────
// The anchor sits at the origin. Every other relevant element is placed
// at:
//   • a DIRECTION (θ, φ) on the unit sphere — θ is the chemical group's
//     fixed elevation band, φ is index-within-group × golden angle. So
//     latitude encodes chemistry; longitude spreads members of the same
//     family around their band so they don't overlap.
//   • a RADIAL DISTANCE r monotone in |ΔEN(anchor, atom)| — chemically
//     close-EN atoms appear close to the centre, distant atoms sit far
//     out. minR keeps the nearest atom clear of the anchor's circle.
//
// Members of a group are sorted by EN closeness before φ is assigned,
// so adjacent indices (= adjacent golden-angle longitudes) walk
// outward along the radial axis instead of jumping randomly.
function compute3DScatter(
  anchorIdx: number, cx: number, cy: number,
  cs: number, w: number, h: number,
): { pts: Vec3[]; enScale: number } {
  const anchor = ELEMENTS[anchorIdx];

  // Actual max ΔEN among relevant non-anchor atoms with known EN.
  let maxDEN = 0;
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant || el.en === 0 || anchor.en === 0) return;
    maxDEN = Math.max(maxDEN, Math.abs(anchor.en - el.en));
  });
  if (maxDEN < 0.1) maxDEN = 1;
  const enScale = maxDEN * 1.05;

  // Radial distance bounds. minR keeps even the closest-EN atom clear
  // of the anchor's drawn circle plus some buffer. maxR is the canvas
  // fit budget.
  const aR       = scatterR(anchor, cs);
  const margin   = cs;
  const maxAtomR = (cs / 2) * 1.45;
  const minR     = Math.max(aR + maxAtomR + cs * 1.2, 120);
  const maxR     = Math.max(minR + cs * 2, Math.min(w / 2, h / 2) - margin);

  // Group every relevant non-anchor element by category, then sort each
  // group by EN closeness so index-within-group walks outward from the
  // anchor along the radial axis.
  const byGroup: Record<string, number[]> = {};
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant) return;
    (byGroup[el.cat] ??= []).push(i);
  });
  Object.values(byGroup).forEach(idxs =>
    idxs.sort((a, b) =>
      Math.abs(anchor.en - ELEMENTS[a].en) -
      Math.abs(anchor.en - ELEMENTS[b].en),
    ),
  );

  // Anchor sits at the centre. Non-relevant elements collapse there
  // too — they're invisible in scatter/compound mode so we never see
  // them.
  const pts: Vec3[] = ELEMENTS.map(() => ({ x: cx, y: cy, z: 0 }));

  for (const [group, idxs] of Object.entries(byGroup)) {
    const theta = GROUP_THETA[group as Cat] ?? Math.PI / 2;
    const sinT  = Math.sin(theta);
    const cosT  = Math.cos(theta);

    idxs.forEach((idx, k) => {
      // Azimuthal spread by golden angle. The +0.5 offset keeps a single
      // member from landing exactly at φ=0 and gives small groups a more
      // pleasing rotation around their band.
      const phi = (k + 0.5) * GOLDEN_ANGLE;
      const cp  = Math.cos(phi);
      const sp  = Math.sin(phi);

      // Spherical → canvas-camera Cartesian:
      //   north pole (θ=0) → -y (up of canvas), south pole → +y (down).
      //   Azimuth wraps in the x–z plane.
      const dirX =  sinT * cp;
      const dirY = -cosT;
      const dirZ =  sinT * sp;

      // Radial distance is *linearly* proportional to |ΔEN| so the
      // visual distance honestly reads as the chemical distance. Atoms
      // with unknown EN go to maxR.
      const el = ELEMENTS[idx];
      const dEN = (el.en === 0 || anchor.en === 0)
        ? maxDEN
        : Math.abs(anchor.en - el.en);
      const dNorm = Math.min(dEN / enScale, 1);
      const r = minR + dNorm * (maxR - minR);

      pts[idx] = {
        x: cx + dirX * r,
        y: cy + dirY * r,
        z:        dirZ * r,
      };
    });
  }

  return { pts, enScale };
}

// ─── 2D scatter computation (flat view) ─────────────────────────────────
// Original radial layout. Each group occupies a sector of the plane
// (GROUP_ANGLE ± GROUP_SPREAD/2); members are placed along that fan
// with sqrt-scaled radius, then a collision relaxer pushes overlapping
// circles apart while keeping the anchor pinned.
function compute2DScatter(
  anchorIdx: number, cx: number, cy: number,
  cs: number, w: number, h: number,
): { pts: Vec3[]; enScale: number } {
  const anchor = ELEMENTS[anchorIdx];
  const aR     = scatterR(anchor, cs, "2d");
  const maxR   = Math.min(w / 2 - cs * 0.8, h / 2 - cs * 0.8);
  const minR   = Math.max(aR + cs * 0.6, 50);

  let maxDEN = 0;
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant || el.en === 0) return;
    maxDEN = Math.max(maxDEN, Math.abs(anchor.en - el.en));
  });
  if (maxDEN < 0.1) maxDEN = 1;
  const enScale = maxDEN * 1.05;

  const byGroup: Record<string, number[]> = {};
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant) return;
    (byGroup[el.cat] ??= []).push(i);
  });

  const pts: Vec3[] = ELEMENTS.map(() => ({ x: cx, y: cy, z: 0 }));

  for (const [group, idxs] of Object.entries(byGroup)) {
    const base   = GROUP_ANGLE[group]  ?? 0;
    const spread = GROUP_SPREAD[group] ?? 0.5;
    idxs.sort((a, b) =>
      Math.abs(anchor.en - ELEMENTS[a].en) - Math.abs(anchor.en - ELEMENTS[b].en),
    );
    idxs.forEach((idx, k) => {
      const dEN = Math.abs(anchor.en - ELEMENTS[idx].en);
      const r   = minR + Math.sqrt(Math.min(dEN / enScale, 1)) * (maxR - minR);
      const off = idxs.length === 1 ? 0 : spread * (k / (idxs.length - 1) - 0.5);
      const a   = base + off;
      pts[idx]  = { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, z: 0 };
    });
  }

  // Collision resolution — anchor stays fixed, pushes others away.
  for (let iter = 0; iter < 300; iter++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++) {
      if (i !== anchorIdx && !ELEMENTS[i].relevant) continue;
      const rI = scatterR(ELEMENTS[i], cs, "2d");
      for (let j = i + 1; j < pts.length; j++) {
        if (j !== anchorIdx && !ELEMENTS[j].relevant) continue;
        const rJ = scatterR(ELEMENTS[j], cs, "2d");
        const minD = rI + rJ + 8;
        const dx = pts[j].x - pts[i].x;
        const dy = pts[j].y - pts[i].y;
        const d  = Math.hypot(dx, dy);
        if (d < minD && d > 0.001) {
          const push = (minD - d) / 2 + 0.5;
          const nx = dx / d, ny = dy / d;
          if (i === anchorIdx) {
            pts[j].x += nx * push * 2; pts[j].y += ny * push * 2;
          } else if (j === anchorIdx) {
            pts[i].x -= nx * push * 2; pts[i].y -= ny * push * 2;
          } else {
            pts[i].x -= nx * push; pts[i].y -= ny * push;
            pts[j].x += nx * push; pts[j].y += ny * push;
          }
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  pts.forEach((p, i) => {
    if (i === anchorIdx || !ELEMENTS[i].relevant) return;
    const margin = scatterR(ELEMENTS[i], cs, "2d") + 6;
    p.x = Math.max(margin, Math.min(w - margin, p.x));
    p.y = Math.max(margin, Math.min(h - margin, p.y));
  });

  return { pts, enScale };
}

// ─── Component ──────────────────────────────────────────────────────────
interface ElementMapProps {
  onGenerate?: (elA: string, elB: string, n: number) => void;
  isGenerating?: boolean;
}

export function ElementMap({ onGenerate, isGenerating }: ElementMapProps = {}) {
  const wrapRef    = useRef<HTMLDivElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const compoundPanelRef = useRef<HTMLDivElement>(null);

  const posRef     = useRef<Vec3[]>([]);
  const tgtRef     = useRef<Vec3[]>([]);
  const opacityRef = useRef<number[]>([]);
  const cellRef    = useRef(0);
  const cwRef      = useRef(0);
  const chRef      = useRef(0);
  // Target canvas height — lerped toward by the tick loop so the canvas
  // morphs smoothly between grid_h and SCATTER_H instead of snapping.
  const targetHRef = useRef(0);
  const rafRef     = useRef(0);
  const drawRafRef = useRef(0);
  const selARef    = useRef<number | null>(null);
  const selBRef    = useRef<number | null>(null);
  const hoverRef   = useRef<number | null>(null);
  const animRef    = useRef(false);
  const binaryCountsRef = useRef<Record<string, { total: number; stable: number }>>({});
  const binaryAbortRef  = useRef<AbortController | null>(null);
  const enScaleRef = useRef(2.5);
  const shapeRef   = useRef<number[]>([]);

  // 3D camera state
  const rotRef       = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const zoomRef      = useRef<number>(1);
  const dragRef      = useRef<{ mx: number; my: number; rx: number; ry: number } | null>(null);
  const dragMovedRef = useRef(false);
  // Cached projected positions (computed at the top of every draw, reused
  // by hit-testing & tooltip placement after the draw completes).
  const projRef      = useRef<{ x: number; y: number; s: number }[]>([]);

  const [selectedA, setSelectedA] = useState<number | null>(null);
  const [selectedB, setSelectedB] = useState<number | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [nCandidates, setNCandidates] = useState(12);
  // 2D is the default scatter view; 3D enables the spherical band layout
  // with rotation + zoom. The ref tracks the same value for the canvas
  // loop so handlers can read it without a re-render dependency.
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const viewModeRef = useRef<"2d" | "3d">("2d");

  // MP phase data
  interface MpPhase {
    id: string; formula: string; spacegroup: string; crystal_system: string;
    e_above_hull: number; formation_energy: number; density: number;
    n_sites: number; volume: number; is_stable: boolean;
    a: number | null; b: number | null; c: number | null;
    alpha: number | null; beta: number | null; gamma: number | null;
  }
  const [mpPhases, setMpPhases]     = useState<MpPhase[] | null>(null);
  const [mpLoading, setMpLoading]   = useState(false);
  const [mpError, setMpError]       = useState<string | null>(null);
  const mpAbortRef = useRef<AbortController | null>(null);

  const mode = selectedA !== null && selectedB !== null
    ? "compound" as const
    : selectedA !== null ? "scatter" as const : "grid" as const;

  const stablePhases = useMemo(
    () => mpPhases?.filter(p => p.is_stable) ?? [],
    [mpPhases],
  );

  // ── canvas helpers ────────────────────────────────────────────────────

  const resizeCanvas = useCallback((w: number, h: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";
    canvas.getContext("2d")?.scale(dpr, dpr);
    cwRef.current = w;
    chRef.current = h;
  }, []);

  const gridTargets = useCallback((w: number, cs: number): Vec3[] => {
    const step = cs + GAP;
    return ELEMENTS.map(el => ({
      x: PAD + el.col * step + cs / 2,
      y: gridYFn(el.row, step) + cs / 2,
      z: 0,
    }));
  }, []);

  // ── helpers for current mode ──────────────────────────────────────────

  function modeRef(): "grid" | "scatter" | "compound" {
    if (selARef.current !== null && selBRef.current !== null) return "compound";
    if (selARef.current !== null) return "scatter";
    return "grid";
  }

  function opaTarget(el: El, _i: number, m: "grid"|"scatter"|"compound"): number {
    if (m === "compound" || m === "scatter") return el.relevant ? 1.0 : 0;
    return el.relevant ? 1.0 : INACTIVE_ALPHA;
  }

  // ── draw ──────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w    = cwRef.current;
    const h    = chRef.current;
    const cs   = cellRef.current;
    const selA = selARef.current;
    const selB = selBRef.current;
    const hov  = hoverRef.current;
    const m    = modeRef();
    const font = "ui-sans-serif,system-ui,-apple-system,sans-serif";
    const cx   = w / 2;
    const cy   = h / 2;
    const rot  = rotRef.current;
    const zoom = zoomRef.current;

    ctx.clearRect(0, 0, w, h);

    // ── Project all positions through zoom + rotation + perspective ───
    const projected: { x: number; y: number; s: number }[] =
      ELEMENTS.map((_, i) => {
        const p = posRef.current[i];
        if (!p) return { x: 0, y: 0, s: 1 };
        const zp = zoomedPoint(p, zoom, cx, cy);
        return projectPoint(zp, rot, cx, cy);
      });
    projRef.current = projected;

    // Helper: draw radius for an element at index i in current mode +
    // current shape morph state (does NOT include hover scaling).
    const drawnRadius = (i: number): number => {
      const el = ELEMENTS[i];
      const proj = projected[i];
      const shape = shapeRef.current[i] ?? 0;
      const gridR    = cs / 2;
      const scatterRR = scatterR(el, cs, viewModeRef.current) * proj.s;
      return gridR + (scatterRR - gridR) * shape;
    };

    // Lanthanide / actinide indicators in grid mode
    if (m === "grid") {
      const step = cs + GAP;
      const indX = PAD + 2 * step + cs / 2;
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.font = `600 ${Math.max(6, Math.round(cs * 0.18))}px ${font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = CAT_COLORS.lanthanide.fill;
      ctx.fillText("57-71", indX, gridYFn(5, step) + cs / 2);
      ctx.fillStyle = CAT_COLORS.actinide.fill;
      ctx.fillText("89-103", indX, gridYFn(6, step) + cs / 2);
      ctx.restore();
    }

    // ── Connection lines (binary-compound counts) ─────────────────────
    // One line per connection. Encode count via thickness + grey shade
    // only. Lines start/end at the node circle edges so they kiss the
    // borders cleanly at any zoom level.
    if ((m === "scatter" || m === "compound") && selA !== null) {
      const aProj = projected[selA];
      const counts = binaryCountsRef.current;
      if (aProj && Object.keys(counts).length > 0) {
        const aR = drawnRadius(selA);
        ctx.save();
        ctx.lineCap = "round";
        ELEMENTS.forEach((el, i) => {
          if (i === selA || !el.relevant) return;
          if ((opacityRef.current[i] ?? 0) < 0.05) return;
          const cnt = counts[el.s];
          if (!cnt || cnt.total === 0) return;

          const p = projected[i];
          if (!p) return;

          const dx = p.x - aProj.x, dy = p.y - aProj.y;
          const len = Math.hypot(dx, dy);
          if (len < 1) return;

          const eR = drawnRadius(i);
          const f0 = aR / len;
          const f1 = 1 - eR / len;
          if (f1 <= f0) return; // circles overlap — skip

          const { width, color } = connStyle(cnt.total);
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(aProj.x + dx * f0, aProj.y + dy * f0);
          ctx.lineTo(aProj.x + dx * f1, aProj.y + dy * f1);
          ctx.stroke();
        });
        ctx.restore();
      }
    }

    // ── elements ──────────────────────────────────────────────────────
    // Unified draw: interpolates shape between square (grid) and circle
    // (scatter) using shapeRef (0 = square, 1 = circle), scaling by the
    // perspective factor s so far nodes appear smaller.
    const drawEl = (i: number) => {
      const el   = ELEMENTS[i];
      const proj = projected[i];
      const opa  = opacityRef.current[i];
      if (!proj || opa < 0.01) return;

      const isAnchor   = i === selA;
      const isSelected = i === selB;
      const isHovered  = i === hov && el.relevant;
      const hScale     = (isHovered || isSelected) ? 1.14 : 1.0;
      const c          = CAT_COLORS[el.cat];
      const shape      = shapeRef.current[i] ?? 0;

      // Grid square has no perspective (z=0 → s=1), scatter circle uses
      // perspective scaling so far nodes shrink.
      const gridSize    = cs * hScale;
      const scatterDiam = scatterR(el, cs, viewModeRef.current) * 2 * proj.s * hScale;
      const size        = gridSize + (scatterDiam - gridSize) * shape;
      const bRadius     = 5 + (size / 2 - 5) * shape;

      ctx.save();
      ctx.globalAlpha = opa;

      const bx = proj.x - size / 2;
      const by = proj.y - size / 2;
      drawRoundRect(ctx, bx, by, size, size, bRadius);

      const isScatter = shape > 0.5;

      if (isAnchor) {
        if (isScatter) {
          ctx.fillStyle = c.fill;
          ctx.fill();
          ctx.strokeStyle = "#3b82f6";
          ctx.lineWidth = 3;
          ctx.stroke();
        } else {
          ctx.fillStyle = "#3b82f6";
          ctx.fill();
        }
        ctx.fillStyle = "#fff";
      } else if (isSelected) {
        ctx.fillStyle = c.fill;
        ctx.fill();
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = "#fff";
      } else if (isHovered) {
        ctx.fillStyle = c.fill;
        ctx.fill();
        ctx.fillStyle = "#fff";
      } else {
        ctx.fillStyle = c.bg;
        ctx.fill();
        ctx.strokeStyle = c.stroke;
        ctx.lineWidth = el.relevant ? 1.5 : 0.8;
        ctx.stroke();
        ctx.fillStyle = c.text;
      }

      const scatterFont = Math.max(9, Math.round(scatterDiam * 0.35));
      const gridFont    = Math.max(8, Math.round(gridSize * 0.32));
      const fontSize    = Math.round(gridFont + (scatterFont - gridFont) * shape);
      ctx.font = `600 ${fontSize}px ${font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(el.s, proj.x, proj.y);

      ctx.restore();
    };

    // Depth sort: furthest first (smallest s) so closer nodes render on
    // top. Anchor is always drawn last so it stays visible when zoomed
    // in close. Hovered element draws second-to-last so its halo shows.
    if (m === "grid") {
      ELEMENTS.forEach((el, i) => { if (!el.relevant && i !== hov) drawEl(i); });
      ELEMENTS.forEach((el, i) => { if (el.relevant && i !== hov && i !== selA) drawEl(i); });
      if (hov !== null && hov !== selA) drawEl(hov);
      if (selA !== null) drawEl(selA);
    } else {
      // 3D: include the anchor in the z-order sort so atoms in front of
      // it (rotated z < 0) draw on top and atoms behind draw underneath.
      // 2D: everything is on the z=0 plane, so the sort is meaningless;
      // fall back to the original behaviour where the anchor always
      // draws last so it stays visible on top.
      const is3D = viewModeRef.current === "3d";
      const order = ELEMENTS.map((_, i) => i)
        .filter(i => i !== hov && (is3D || i !== selA))
        .sort((a, b) => projected[a].s - projected[b].s);
      order.forEach(i => drawEl(i));
      if (hov !== null && (is3D || hov !== selA)) drawEl(hov);
      if (!is3D && selA !== null) drawEl(selA);
    }

    // Tooltip — position above the hovered node using its projected
    // screen coords + drawn radius (with hover scale).
    if (tooltipRef.current && hov !== null && projected[hov]) {
      const p     = projected[hov];
      const hovR  = drawnRadius(hov) * 1.14;
      tooltipRef.current.style.transform =
        `translate(${p.x}px, ${p.y - hovR - 6}px)`;
    }
  }, []);

  // Schedule a single redraw on the next animation frame. Used by event
  // handlers and async callbacks — never call draw() directly from a
  // handler, otherwise renders stack up and motion feels rocky.
  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(drawRafRef.current);
    drawRafRef.current = requestAnimationFrame(draw);
  }, [draw]);

  // ── animation loop ────────────────────────────────────────────────────

  const tick = useCallback(() => {
    const m = modeRef();
    let done = true;

    posRef.current.forEach((p, i) => {
      const t = tgtRef.current[i];
      if (!t) return;
      const dx = t.x - p.x, dy = t.y - p.y, dz = t.z - p.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > 0.3) {
        const speed = LERP + Math.min(dist / 800, 0.08);
        p.x += dx * speed;
        p.y += dy * speed;
        p.z += dz * speed;
        done = false;
      } else { p.x = t.x; p.y = t.y; p.z = t.z; }
    });

    opacityRef.current.forEach((o, i) => {
      const el = ELEMENTS[i];
      const target = opaTarget(el, i, m);
      const diff = target - o;
      if (Math.abs(diff) > 0.005) {
        const fast = (m !== "grid" && target === 0);
        opacityRef.current[i] = o + diff * (fast ? 0.32 : LERP);
        done = false;
      } else {
        opacityRef.current[i] = target;
      }
    });

    // Morph shape: 0 = square (grid), 1 = circle (scatter/compound)
    const shapeTarget = (m === "scatter" || m === "compound") ? 1 : 0;
    shapeRef.current.forEach((s, i) => {
      const diff = shapeTarget - s;
      if (Math.abs(diff) > 0.005) {
        shapeRef.current[i] = s + diff * SHAPE_LERP;
        done = false;
      } else {
        shapeRef.current[i] = shapeTarget;
      }
    });

    // Morph canvas height alongside the rest so the page doesn't snap
    // when transitioning between the periodic table and the nodegraph.
    const tH = targetHRef.current;
    if (tH > 0) {
      const dh = tH - chRef.current;
      if (Math.abs(dh) > 0.5) {
        resizeCanvas(cwRef.current, chRef.current + dh * LERP);
        done = false;
      } else if (chRef.current !== tH) {
        resizeCanvas(cwRef.current, tH);
      }
    }

    draw();

    if (!done) {
      animRef.current = true;
      rafRef.current = requestAnimationFrame(tick);
    } else {
      animRef.current = false;
    }
  }, [draw, resizeCanvas]);

  const startAnim = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    animRef.current = true;
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  // ── layout transitions ────────────────────────────────────────────────

  const goGrid = useCallback((w: number) => {
    const cs   = Math.floor((w - PAD * 2) / N_COLS);
    const step = cs + GAP;
    const h    = gridYFn(9, step) + cs + PAD;
    cellRef.current = cs;
    // Width changes apply immediately; height is lerped by the tick loop
    // so the canvas morphs smoothly into / out of the scatter view.
    if (cwRef.current !== w) resizeCanvas(w, chRef.current || h);
    targetHRef.current = h;
    tgtRef.current = gridTargets(w, cs);
  }, [resizeCanvas, gridTargets]);

  const goScatter = useCallback((w: number, anchorIdx: number) => {
    if (cwRef.current !== w) resizeCanvas(w, chRef.current || SCATTER_H);
    targetHRef.current = SCATTER_H;
    const compute = viewModeRef.current === "3d" ? compute3DScatter : compute2DScatter;
    const result = compute(
      anchorIdx, w / 2, SCATTER_H / 2, cellRef.current, w, SCATTER_H,
    );
    tgtRef.current = result.pts;
    enScaleRef.current = result.enScale;
  }, [resizeCanvas]);

  // ── reset helper ──────────────────────────────────────────────────────

  const resetToGrid = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    selARef.current = null;
    selBRef.current = null;
    setSelectedA(null);
    setSelectedB(null);
    hoverRef.current = null;
    setHoveredIdx(null);
    setMpPhases(null);
    setMpError(null);
    mpAbortRef.current?.abort();
    binaryCountsRef.current = {};
    binaryAbortRef.current?.abort();
    rotRef.current = { x: 0, y: 0 };
    zoomRef.current = 1;
    goGrid(wrap.clientWidth);
    startAnim();
  }, [goGrid, startAnim]);

  // ── MP API fetch ────────────────────────────────────────────────────

  const fetchMpPhases = useCallback(async (symA: string, symB: string) => {
    mpAbortRef.current?.abort();
    const ctrl = new AbortController();
    mpAbortRef.current = ctrl;
    setMpPhases(null);
    setMpError(null);
    setMpLoading(true);
    try {
      const res = await fetch(
        `/api/mp-phases?a=${encodeURIComponent(symA)}&b=${encodeURIComponent(symB)}`,
        { signal: ctrl.signal },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
      }
      const json = await res.json() as { phases: MpPhase[] };
      if (!ctrl.signal.aborted) setMpPhases(json.phases);
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setMpError(e instanceof Error ? e.message : "Failed to fetch");
      }
    } finally {
      if (!ctrl.signal.aborted) setMpLoading(false);
    }
  }, []);

  // ── MP binary counts fetch ──────────────────────────────────────────

  const fetchBinaryCounts = useCallback(async (symbol: string) => {
    binaryAbortRef.current?.abort();
    const ctrl = new AbortController();
    binaryAbortRef.current = ctrl;
    binaryCountsRef.current = {};
    try {
      const res = await fetch(
        `/api/mp-binary-counts?el=${encodeURIComponent(symbol)}`,
        { signal: ctrl.signal },
      );
      if (res.ok) {
        const json = (await res.json()) as {
          counts: Record<string, { total: number; stable: number }>;
        };
        if (!ctrl.signal.aborted) {
          binaryCountsRef.current = json.counts;
          scheduleDraw();
        }
      }
    } catch { /* ignore abort / network errors */ }
  }, [scheduleDraw]);

  // ── event handlers ────────────────────────────────────────────────────

  // Find the topmost element under (mx, my) using the most recent
  // projected positions. In scatter/compound mode, the topmost is the
  // one with the largest perspective scale (closest to camera).
  const hitTest = useCallback((mx: number, my: number, m: "grid" | "scatter" | "compound"): number => {
    const cs = cellRef.current;
    const projected = projRef.current;
    let hitIdx = -1;
    let hitS = -Infinity;
    for (let i = 0; i < ELEMENTS.length; i++) {
      if (!ELEMENTS[i].relevant) continue;
      if ((m === "scatter" || m === "compound") && (opacityRef.current[i] ?? 0) < 0.1) continue;
      const p = projected[i];
      if (!p) continue;
      if (m === "scatter" || m === "compound") {
        const hitR = scatterR(ELEMENTS[i], cs, viewModeRef.current) * p.s + 3;
        if (Math.hypot(mx - p.x, my - p.y) <= hitR) {
          if (p.s > hitS) { hitS = p.s; hitIdx = i; }
        }
      } else {
        const half = cs / 2 + 3;
        if (Math.abs(mx - p.x) <= half && Math.abs(my - p.y) <= half) {
          // Grid mode — one shot, first match wins.
          return i;
        }
      }
    }
    return hitIdx;
  }, []);

  const onCanvasClick = useCallback((e: MouseEvent) => {
    // Suppress click when the user just finished a rotation drag.
    if (dragMovedRef.current) {
      dragMovedRef.current = false;
      return;
    }

    const canvas = canvasRef.current;
    const wrap   = wrapRef.current;
    if (!canvas || !wrap) return;
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const m    = modeRef();
    const w    = wrap.clientWidth;

    const hitIdx = hitTest(mx, my, m);

    if (m === "grid") {
      if (hitIdx === -1) return;
      selARef.current = hitIdx;
      setSelectedA(hitIdx);
      // Reset rotation/zoom so the sphere starts in a neutral pose.
      rotRef.current = { x: 0, y: 0 };
      zoomRef.current = 1;
      goScatter(w, hitIdx);
      startAnim();
      fetchBinaryCounts(ELEMENTS[hitIdx].s);
    } else if (m === "scatter" || m === "compound") {
      if (hitIdx === selARef.current) {
        // Click anchor again — reset to grid
        selARef.current = null;
        selBRef.current = null;
        setSelectedA(null);
        setSelectedB(null);
        setMpPhases(null);
        binaryCountsRef.current = {};
        binaryAbortRef.current?.abort();
        rotRef.current = { x: 0, y: 0 };
        zoomRef.current = 1;
        goGrid(w);
        startAnim();
      } else if (hitIdx === -1 && m === "compound") {
        // Click empty space in compound mode — deselect B, stay in scatter
        selBRef.current = null;
        setSelectedB(null);
        setMpPhases(null);
        scheduleDraw();
      } else if (hitIdx >= 0 && hitIdx !== selBRef.current) {
        // Select (or change) element B
        selBRef.current = hitIdx;
        setSelectedB(hitIdx);
        fetchMpPhases(ELEMENTS[selARef.current!].s, ELEMENTS[hitIdx].s);
        scheduleDraw();
        setTimeout(() => {
          compoundPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 100);
      }
    }
  }, [goGrid, goScatter, startAnim, scheduleDraw, fetchBinaryCounts, fetchMpPhases, hitTest]);

  const prevHoverRef = useRef<number | null>(null);

  const onMouseDown = useCallback((e: MouseEvent) => {
    const m = modeRef();
    // Rotation only makes sense in the 3D scatter/compound view.
    if (m === "grid" || viewModeRef.current !== "3d") return;
    dragRef.current = {
      mx: e.clientX, my: e.clientY,
      rx: rotRef.current.x, ry: rotRef.current.y,
    };
    dragMovedRef.current = false;
    if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
  }, []);

  const onWindowMouseUp = useCallback(() => {
    if (dragRef.current) {
      dragRef.current = null;
      const canvas = canvasRef.current;
      if (canvas) {
        const m = modeRef();
        const grabable = m !== "grid" && viewModeRef.current === "3d";
        canvas.style.cursor = grabable ? "grab" : "default";
      }
    }
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const m = modeRef();

    // ── Rotation drag ─────────────────────────────────────────────────
    if (dragRef.current) {
      const d = dragRef.current;
      const dx = e.clientX - d.mx;
      const dy = e.clientY - d.my;
      if (!dragMovedRef.current && Math.hypot(dx, dy) > 3) {
        dragMovedRef.current = true;
      }
      // No clamping on X — allow free rotation past the poles.
      rotRef.current = {
        x: d.rx + dy * 0.006,
        y: d.ry - dx * 0.006,
      };
      // While dragging clear hover to keep tooltip from flickering.
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        prevHoverRef.current = null;
        setHoveredIdx(null);
      }
      scheduleDraw();
      return;
    }

    // While position-lerp animation is running, suppress hover.
    if (animRef.current) {
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        prevHoverRef.current = null;
        setHoveredIdx(null);
      }
      const grabable = m !== "grid" && viewModeRef.current === "3d";
      canvas.style.cursor = grabable ? "grab" : "default";
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const hitIdx = hitTest(mx, my, m);

    const newHov = hitIdx >= 0 ? hitIdx : null;
    hoverRef.current = newHov;
    if (newHov !== null) {
      canvas.style.cursor = "pointer";
    } else {
      const grabable = m !== "grid" && viewModeRef.current === "3d";
      canvas.style.cursor = grabable ? "grab" : "default";
    }

    if (newHov !== prevHoverRef.current) {
      prevHoverRef.current = newHov;
      setHoveredIdx(newHov);
    }

    scheduleDraw();
  }, [scheduleDraw, hitTest]);

  const onMouseLeave = useCallback(() => {
    hoverRef.current = null;
    prevHoverRef.current = null;
    setHoveredIdx(null);
    scheduleDraw();
  }, [scheduleDraw]);

  const onWheel = useCallback((e: WheelEvent) => {
    const m = modeRef();
    // Wheel zoom only in the 3D scatter view; 2D stays at zoom = 1.
    if (m === "grid" || viewModeRef.current !== "3d") return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.09 : 1 / 1.09;
    zoomRef.current = Math.max(0.2, Math.min(5, zoomRef.current * factor));
    scheduleDraw();
  }, [scheduleDraw]);

  // ── view-mode toggle ─────────────────────────────────────────────────
  // Toggling 2D ↔ 3D resets the camera and re-runs the scatter layout
  // for the current anchor (if any). Animates the morph via the regular
  // tick loop.
  const switchViewMode = useCallback((next: "2d" | "3d") => {
    if (viewModeRef.current === next) return;
    viewModeRef.current = next;
    setViewMode(next);
    rotRef.current = { x: 0, y: 0 };
    zoomRef.current = 1;
    const wrap = wrapRef.current;
    if (wrap && selARef.current !== null) {
      goScatter(wrap.clientWidth, selARef.current);
      startAnim();
    } else {
      scheduleDraw();
    }
  }, [goScatter, startAnim, scheduleDraw]);

  const onResize = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const w = wrap.clientWidth;
    if (w === 0) return;
    const m = modeRef();
    if ((m === "compound" || m === "scatter") && selARef.current !== null)
      goScatter(w, selARef.current);
    else goGrid(w);
    // Window resize: snap canvas to the new target — no morph animation.
    resizeCanvas(w, targetHRef.current);
    posRef.current = tgtRef.current.map(p => ({ x: p.x, y: p.y, z: p.z }));
    const targetShape = (m === "scatter" || m === "compound") ? 1 : 0;
    ELEMENTS.forEach((el, i) => {
      opacityRef.current[i] = opaTarget(el, i, m);
      shapeRef.current[i] = targetShape;
    });
    scheduleDraw();
  }, [goGrid, goScatter, scheduleDraw, resizeCanvas]);

  // ── mount ─────────────────────────────────────────────────────────────

  useEffect(() => {
    const wrap   = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const w    = wrap.clientWidth;
    const cs   = Math.floor((w - PAD * 2) / N_COLS);
    const step = cs + GAP;
    const h    = gridYFn(9, step) + cs + PAD;
    cellRef.current = cs;
    resizeCanvas(w, h);
    targetHRef.current = h;

    const tgts = gridTargets(w, cs);
    tgtRef.current = tgts;
    posRef.current = tgts.map(p => ({ x: p.x, y: p.y, z: p.z }));
    opacityRef.current = ELEMENTS.map(el => el.relevant ? 1.0 : INACTIVE_ALPHA);
    shapeRef.current = ELEMENTS.map(() => 0);
    draw();

    canvas.addEventListener("click",      onCanvasClick);
    canvas.addEventListener("mousemove",  onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    window.addEventListener("mouseup",    onWindowMouseUp);
    const ro = new ResizeObserver(onResize);
    ro.observe(wrap);

    return () => {
      canvas.removeEventListener("click",      onCanvasClick);
      canvas.removeEventListener("mousemove",  onMouseMove);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("wheel",      onWheel);
      window.removeEventListener("mouseup",    onWindowMouseUp);
      ro.disconnect();
      cancelAnimationFrame(rafRef.current);
      cancelAnimationFrame(drawRafRef.current);
    };
  }, [resizeCanvas, gridTargets, draw, onCanvasClick, onMouseMove, onMouseLeave, onMouseDown, onWheel, onWindowMouseUp, onResize]);

  // ── render ────────────────────────────────────────────────────────────

  const elA = selectedA !== null ? ELEMENTS[selectedA] : null;
  const elB = selectedB !== null ? ELEMENTS[selectedB] : null;
  const hoveredEl = hoveredIdx !== null ? ELEMENTS[hoveredIdx] : null;

  // Sticky mirror of elA so the pair-builder JSX stays mounted while the
  // wrapper is collapsing back to grid (otherwise the inner content
  // unmounts the same frame selectedA→null and the close animation
  // collapses an empty box).
  const [lastElA, setLastElA] = useState<typeof elA>(null);
  useEffect(() => {
    if (elA) setLastElA(elA);
  }, [elA]);
  const renderElA = elA ?? lastElA;
  const expanded = (mode === "scatter" || mode === "compound") && elA != null;

  return (
    <section id="periodic-table" className="mx-auto max-w-6xl px-4 pb-10 pt-4 sm:px-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
          Microelectronically Relevant Elements
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          {mode === "compound" && elA && elB
            ? `${elA.n}–${elB.n} selected — see details below. Click another element to change, or ${elA.s} to reset.`
            : mode === "scatter" && elA
              ? `${elA.n} selected — pick a second element to explore the binary compound, or click ${elA.s} to reset.`
              : "Choose an element of interest to discover compound properties."}
        </p>
      </div>

      <div className="relative rounded-2xl border border-[var(--border)] bg-white p-4 shadow-[var(--shadow)]">
        {mode !== "grid" && (
        <div
          role="tablist"
          aria-label="View mode"
          className="absolute top-3 right-3 z-20 inline-flex shrink-0 rounded-lg border border-[var(--border)] bg-white p-0.5 text-xs font-medium shadow-sm"
        >
          {(["2d", "3d"] as const).map(v => {
            const active = viewMode === v;
            return (
              <button
                key={v}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => switchViewMode(v)}
                className={`px-3 py-1 rounded-md transition-colors ${
                  active
                    ? "bg-gray-900 text-white"
                    : "text-gray-500 hover:text-gray-900"
                }`}
              >
                {v.toUpperCase()}
              </button>
            );
          })}
        </div>
        )}

        <div ref={wrapRef} className="relative">
          <canvas ref={canvasRef} style={{ display: "block" }} />

          {/* Hover tooltip (grid / scatter only) */}
          <div
            ref={tooltipRef}
            className="pointer-events-none absolute top-0 left-0 z-10"
            style={{
              opacity: hoveredEl ? 1 : 0,
              transition: "opacity 120ms ease",
            }}
          >
            <div className="-translate-x-1/2 -translate-y-full pb-2">
              <div className="rounded-lg bg-gray-900/90 backdrop-blur-sm px-3 py-2 shadow-lg text-white whitespace-nowrap">
                {hoveredEl && (
                  <>
                    <div className="flex items-baseline gap-2">
                      <span className="text-base font-bold leading-tight">{hoveredEl.s}</span>
                      <span className="text-[10px] text-gray-400">{hoveredEl.z}</span>
                    </div>
                    <div className="text-[11px] text-gray-300 leading-tight">{hoveredEl.n}</div>
                    <div className="text-[10px] text-gray-400 mt-0.5 leading-tight">
                      {hoveredEl.en > 0 ? `EN ${hoveredEl.en.toFixed(2)}` : "EN —"}
                      {" · "}
                      {hoveredEl.m.toFixed(1)} u
                    </div>
                    {elA && hoveredIdx !== selectedA && hoveredEl.en > 0 && elA.en > 0 && (
                      <div className="text-[10px] text-blue-300 mt-0.5 leading-tight">
                        ΔEN {Math.abs(elA.en - hoveredEl.en).toFixed(2)}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ── Compound pair builder ──────────────────────────────────── */}
        {(mode === "scatter" || mode === "compound") && elA && (
          <div
            className={`flex items-center justify-center py-5 ${mode !== "grid" ? "border-t border-gray-100" : ""}`}
            onClick={undefined}
            style={{ cursor: undefined }}
          >
            {/* Element A */}
            <div className="flex flex-col items-center gap-1.5">
              <div
                className="w-[88px] h-[88px] rounded-full flex flex-col items-center justify-center shadow-sm"
                style={{
                  background: CAT_COLORS[elA.cat].fill,
                  border: `2.5px solid ${CAT_COLORS[elA.cat].stroke}`,
                }}
              >
                <span className="text-white text-xl font-bold leading-tight">{elA.s}</span>
                <span className="text-white/70 text-[10px] leading-tight">{elA.n}</span>
              </div>
              <span className="text-[10px] text-gray-400">{CAT_LABELS[elA.cat]}</span>
            </div>

            {/* Connecting line */}
            <div className="relative mx-4 w-20 flex items-center">
              <div className="w-full border-t-2 border-dashed border-gray-300" />
              {elB && elA.en > 0 && elB.en > 0 && (
                <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-medium text-gray-500 whitespace-nowrap">
                  ΔEN {Math.abs(elA.en - elB.en).toFixed(2)}
                </span>
              )}
            </div>

            {/* Element B or placeholder */}
            <div className="flex flex-col items-center gap-1.5">
              {elB ? (
                <>
                  <div
                    className="w-[88px] h-[88px] rounded-full flex flex-col items-center justify-center shadow-sm"
                    style={{
                      background: CAT_COLORS[elB.cat].fill,
                      border: `2.5px solid ${CAT_COLORS[elB.cat].stroke}`,
                    }}
                  >
                    <span className="text-white text-xl font-bold leading-tight">{elB.s}</span>
                    <span className="text-white/70 text-[10px] leading-tight">{elB.n}</span>
                  </div>
                  <span className="text-[10px] text-gray-400">{CAT_LABELS[elB.cat]}</span>
                </>
              ) : (
                <>
                  <div className="w-[88px] h-[88px] rounded-full border-2 border-dashed border-gray-300 flex flex-col items-center justify-center">
                    <svg className="w-5 h-5 text-gray-300 mb-0.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                    </svg>
                    <span className="text-[10px] text-gray-400 leading-tight">Select 2nd</span>
                  </div>
                  <span className="text-[10px] text-transparent select-none">placeholder</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Compound info panel ──────────────────────────────────────── */}
      {mode === "compound" && elA && elB && (
        <div ref={compoundPanelRef} className="mt-5 rounded-2xl border border-[var(--border)] bg-white p-6 shadow-[var(--shadow)]">
          <div className="flex items-start justify-between mb-5">
            <div>
              <h3 className="text-lg font-semibold text-[var(--foreground)]">
                {elA.s}–{elB.s} Binary System
              </h3>
              <p className="mt-0.5 text-sm text-[var(--muted)]">
                Exploring the {elA.n}–{elB.n} composition space
              </p>
            </div>
            <button
              onClick={() => { selBRef.current = null; setSelectedB(null); setMpPhases(null); scheduleDraw(); }}
              className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted)] hover:bg-gray-50 transition-colors"
            >
              ← Back to scatter
            </button>
          </div>

          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Electronegativity</div>
              <div className="text-sm font-semibold text-[var(--foreground)]">
                ΔEN = {Math.abs(elA.en - elB.en).toFixed(2)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {elA.s} ({elA.en.toFixed(2)}) · {elB.s} ({elB.en.toFixed(2)})
              </div>
            </div>

            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Element Categories</div>
              <div className="flex flex-wrap gap-1 mt-1">
                <span className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                  style={{ background: CAT_COLORS[elA.cat].bg, color: CAT_COLORS[elA.cat].text, border: `1px solid ${CAT_COLORS[elA.cat].stroke}` }}>
                  {CAT_LABELS[elA.cat]}
                </span>
                <span className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                  style={{ background: CAT_COLORS[elB.cat].bg, color: CAT_COLORS[elB.cat].text, border: `1px solid ${CAT_COLORS[elB.cat].stroke}` }}>
                  {CAT_LABELS[elB.cat]}
                </span>
              </div>
            </div>

            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Known Phases (MP)</div>
              {mpLoading && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
                  Loading…
                </div>
              )}
              {mpError && <div className="text-xs text-red-500">{mpError}</div>}
              {mpPhases && !mpLoading && (
                <div className="text-sm font-semibold text-[var(--foreground)]">
                  {mpPhases.length} phase{mpPhases.length !== 1 ? "s" : ""}
                  {stablePhases.length > 0 && (
                    <span className="ml-1 text-xs font-normal text-emerald-600">
                      ({stablePhases.length} stable)
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* MP phases table */}
          {mpPhases && mpPhases.length > 0 && !mpLoading && (
            <div className="mb-5 overflow-x-auto rounded-lg border border-gray-200">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-gray-200 bg-gray-50 text-gray-500">
                    <th className="px-3 py-2 font-medium">Formula</th>
                    <th className="px-3 py-2 font-medium">Space Group</th>
                    <th className="px-3 py-2 font-medium">Crystal System</th>
                    <th className="px-3 py-2 font-medium text-right">E<sub>hull</sub> (eV/at)</th>
                    <th className="px-3 py-2 font-medium text-right">E<sub>form</sub> (eV/at)</th>
                    <th className="px-3 py-2 font-medium text-right">Density (g/cm³)</th>
                    <th className="px-3 py-2 font-medium text-right">a (Å)</th>
                    <th className="px-3 py-2 font-medium text-right">b (Å)</th>
                    <th className="px-3 py-2 font-medium text-right">c (Å)</th>
                    <th className="px-3 py-2 font-medium text-center">Stable</th>
                    <th className="px-3 py-2 font-medium">MP ID</th>
                  </tr>
                </thead>
                <tbody>
                  {mpPhases.slice(0, 20).map((p) => (
                    <tr key={p.id} className="border-b border-gray-100 hover:bg-gray-50/60">
                      <td className="px-3 py-2 font-medium text-[var(--foreground)]">{p.formula}</td>
                      <td className="px-3 py-2 text-gray-600">{p.spacegroup}</td>
                      <td className="px-3 py-2 text-gray-600 capitalize">{p.crystal_system}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.e_above_hull === "number" ? p.e_above_hull.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.formation_energy === "number" ? p.formation_energy.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.density === "number" ? p.density.toFixed(2) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.a === "number" ? p.a.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.b === "number" ? p.b.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.c === "number" ? p.c.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-center">
                        {p.is_stable
                          ? <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" title="Stable" />
                          : <span className="inline-block h-2 w-2 rounded-full bg-gray-300" title="Metastable" />}
                      </td>
                      <td className="px-3 py-2">
                        <a href={`https://next-gen.materialsproject.org/materials/${p.id}`}
                          target="_blank" rel="noopener noreferrer"
                          className="text-blue-600 hover:underline">{p.id}</a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {mpPhases.length > 20 && (
                <div className="px-3 py-2 text-xs text-gray-400 bg-gray-50 border-t border-gray-200">
                  Showing 20 of {mpPhases.length} phases
                </div>
              )}
            </div>
          )}

          {mpPhases && mpPhases.length === 0 && !mpLoading && (
            <div className="mb-5 rounded-lg border border-dashed border-gray-300 p-4 text-center text-sm text-gray-400">
              No known phases found in Materials Project for {elA.s}–{elB.s}
            </div>
          )}

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-gray-600">
              Candidates
              <input
                type="number"
                min={10}
                max={200}
                step={10}
                value={nCandidates}
                onChange={(e) => setNCandidates(Number(e.target.value) || 50)}
                className="h-10 w-20 rounded-lg border border-gray-200 bg-gray-50 px-3 text-sm text-gray-900 tabular-nums"
              />
            </label>
            <button
              type="button"
              disabled={isGenerating}
              onClick={() => onGenerate?.(elA.s, elB.s, nCandidates)}
              className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 active:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isGenerating ? (
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714a2.25 2.25 0 0 0 .659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 0 1-1.59.659H9.06a2.25 2.25 0 0 1-1.591-.659L5 14.5m14 0H5" />
                </svg>
              )}
              {isGenerating ? "Generating…" : `Generate New Crystal Structure Candidates for ${elA.s}–${elB.s}`}
            </button>
          </div>
        </div>
      )}

      {/* Legend (hide in compound mode) */}
      {mode !== "compound" && (
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1">
          {LEGEND_GROUPS.map(cat => {
            const c = CAT_COLORS[cat];
            return (
              <div key={cat} className="flex items-center gap-1.5 text-xs text-slate-500">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm"
                  style={{ background: c.bg, border: `1.5px solid ${c.stroke}` }}
                />
                {CAT_LABELS[cat]}
              </div>
            );
          })}
          {elA && (
            <span className="ml-auto text-xs text-[var(--muted)]">
              Distance&nbsp;∝&nbsp;|ΔEN| from {elA.s}&nbsp;(EN&nbsp;=&nbsp;{elA.en})
            </span>
          )}
        </div>
      )}
    </section>
  );
}
