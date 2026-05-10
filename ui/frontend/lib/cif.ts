/**
 * Tiny CIF reader — just enough to render a relaxed structure in the 3D viewer.
 *
 * Supports the pymatgen-style "data_X / _cell_length_* / _cell_angle_* / loop_
 * _atom_site_*" layout that all our data/mattergen/{system}/relaxed/*.cif files
 * use. Not a general-purpose CIF parser.
 */

export interface ParsedCif {
  prototype: string;
  formula: string;
  n_atoms: number;
  crystal_system: string;
  lattice_matrix: [number, number, number][];
  lattice_params: {
    a: number;
    b: number;
    c: number;
    alpha: number;
    beta: number;
    gamma: number;
  };
  volume: number;
  species: string[];
  frac_coords: [number, number, number][];
  cart_coords: [number, number, number][];
}

function pickScalar(text: string, key: string): number {
  const re = new RegExp(`_${key}\\s+([0-9eE+\\-.]+)`);
  const m = text.match(re);
  if (!m) throw new Error(`CIF missing _${key}`);
  return parseFloat(m[1]);
}

/** Lattice matrix (a along x; b in xy-plane) from cell parameters. */
function latticeMatrix(
  a: number, b: number, c: number,
  alphaDeg: number, betaDeg: number, gammaDeg: number
): [number, number, number][] {
  const alpha = (alphaDeg * Math.PI) / 180;
  const beta = (betaDeg * Math.PI) / 180;
  const gamma = (gammaDeg * Math.PI) / 180;

  const ax = a;
  const bx = b * Math.cos(gamma);
  const by = b * Math.sin(gamma);
  const cx = c * Math.cos(beta);
  const cy = (c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma))) / Math.sin(gamma);
  const cz = Math.sqrt(Math.max(c * c - cx * cx - cy * cy, 0));

  return [
    [ax, 0, 0],
    [bx, by, 0],
    [cx, cy, cz],
  ];
}

function reducedFormula(species: string[]): string {
  const counts = new Map<string, number>();
  species.forEach((s) => counts.set(s, (counts.get(s) ?? 0) + 1));
  const values = Array.from(counts.values());
  const gcd = (x: number, y: number): number => (y === 0 ? x : gcd(y, x % y));
  const g = values.reduce((a, b) => gcd(a, b));
  const order = ["Li","Na","K","Mg","Ca","Fe","Co","Ni","Cu","Zn","Ti","Mo","W","Ru","Mn","Cr","Al"];
  const parts = Array.from(counts.entries()).sort(([a], [b]) => {
    const ia = order.indexOf(a); const ib = order.indexOf(b);
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    return a.localeCompare(b);
  });
  return parts.map(([el, cnt]) => {
    const n = cnt / g;
    return n === 1 ? el : `${el}${n}`;
  }).join("");
}

/** Naive lattice-only crystal system classifier — used only as a fallback;
 *  the viewer prefers candidate.crystal_system (from the CSV / spglib). */
function classifyLattice(p: ParsedCif["lattice_params"]): string {
  const { a, b, c, alpha, beta, gamma } = p;
  const eq = (x: number, y: number) => Math.abs(x - y) < 0.01 * Math.max(x, y, 1);
  const ang = (x: number, t: number) => Math.abs(x - t) < 0.5;
  const a90 = ang(alpha, 90), b90 = ang(beta, 90), g90 = ang(gamma, 90);
  const g120 = ang(gamma, 120);
  const ab = eq(a, b), bc = eq(b, c);
  if (ab && bc && a90 && b90 && g90) return "Cubic";
  if (ab && !bc && a90 && b90 && g120) return "Hexagonal";
  if (ab && !bc && a90 && b90 && g90) return "Tetragonal";
  if (ab && bc && a90 && b90 && !g90 && !g120) return "Trigonal";
  if (!ab && !bc && a90 && b90 && g90) return "Orthorhombic";
  if (a90 && !b90 && g90) return "Monoclinic";
  return "Triclinic";
}

export function parseCif(text: string, prototypeLabel: string): ParsedCif {
  const a = pickScalar(text, "cell_length_a");
  const b = pickScalar(text, "cell_length_b");
  const c = pickScalar(text, "cell_length_c");
  const alpha = pickScalar(text, "cell_angle_alpha");
  const beta = pickScalar(text, "cell_angle_beta");
  const gamma = pickScalar(text, "cell_angle_gamma");

  const lat = latticeMatrix(a, b, c, alpha, beta, gamma);
  const [av, bv, cv] = lat;
  // Volume = |a · (b × c)|
  const cross: [number, number, number] = [
    bv[1] * cv[2] - bv[2] * cv[1],
    bv[2] * cv[0] - bv[0] * cv[2],
    bv[0] * cv[1] - bv[1] * cv[0],
  ];
  const volume = Math.abs(av[0] * cross[0] + av[1] * cross[1] + av[2] * cross[2]);

  // Parse the _atom_site_* loop. Find the loop_ block that lists
  // _atom_site_fract_x / _y / _z and read fixed-width whitespace columns.
  const lines = text.split(/\r?\n/);
  const species: string[] = [];
  const frac: [number, number, number][] = [];

  let i = 0;
  while (i < lines.length) {
    if (/^\s*loop_/.test(lines[i])) {
      // Read header lines until we run out of leading "_..." entries
      const headers: string[] = [];
      i++;
      while (i < lines.length && /^\s*_/.test(lines[i])) {
        headers.push(lines[i].trim());
        i++;
      }
      const isAtomLoop = headers.some((h) => h === "_atom_site_fract_x");
      if (!isAtomLoop) continue;

      const colSym = headers.indexOf("_atom_site_type_symbol");
      const colX = headers.indexOf("_atom_site_fract_x");
      const colY = headers.indexOf("_atom_site_fract_y");
      const colZ = headers.indexOf("_atom_site_fract_z");

      while (i < lines.length) {
        const ln = lines[i];
        if (/^\s*$/.test(ln) || /^\s*loop_/.test(ln) || /^\s*_/.test(ln) || /^\s*data_/.test(ln)) {
          break;
        }
        const tokens = ln.trim().split(/\s+/);
        if (tokens.length < headers.length) { i++; continue; }
        const sym = tokens[colSym];
        const x = parseFloat(tokens[colX]);
        const y = parseFloat(tokens[colY]);
        const z = parseFloat(tokens[colZ]);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          i++; continue;
        }
        species.push(sym);
        frac.push([x, y, z]);
        i++;
      }
      break;
    }
    i++;
  }

  if (species.length === 0) {
    throw new Error("CIF has no _atom_site_* rows");
  }

  // Cartesian = frac · lattice  (row-vector convention to match Python)
  const cart: [number, number, number][] = frac.map((f) => [
    f[0] * lat[0][0] + f[1] * lat[1][0] + f[2] * lat[2][0],
    f[0] * lat[0][1] + f[1] * lat[1][1] + f[2] * lat[2][1],
    f[0] * lat[0][2] + f[1] * lat[1][2] + f[2] * lat[2][2],
  ]);

  const params = {
    a: Math.round(a * 1e4) / 1e4,
    b: Math.round(b * 1e4) / 1e4,
    c: Math.round(c * 1e4) / 1e4,
    alpha: Math.round(alpha * 100) / 100,
    beta: Math.round(beta * 100) / 100,
    gamma: Math.round(gamma * 100) / 100,
  };

  return {
    prototype: prototypeLabel,
    formula: reducedFormula(species),
    n_atoms: species.length,
    crystal_system: classifyLattice(params),
    lattice_matrix: lat,
    lattice_params: params,
    volume: Math.round(volume * 1e4) / 1e4,
    species,
    frac_coords: frac,
    cart_coords: cart,
  };
}
