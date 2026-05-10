export interface CandidateResult {
  composition: string;
  formula: string;
  n_atoms: number;
  x_B: number;
  formation_energy_eV_atom: number;
  e_above_hull_eV_atom: number;
  predicted_stable: boolean;
  crystal_system: string;
}

export interface MaceResult extends CandidateResult {
  mace_energy_eV_atom: number;
  mace_e_above_hull_eV_atom: number;
  mace_stable: boolean;
}

export interface GenerateRequest {
  element_a: string;
  element_b: string;
  n_candidates: number;
}

export interface ValidateRequest {
  candidates: CandidateResult[];
}

export interface MpPhase {
  id: string;
  formula: string;
  spacegroup: string;
  crystal_system: string;
  e_above_hull: number;
  formation_energy: number;
  density: number;
  n_sites: number;
  volume: number;
  is_stable: boolean;
  a: number | null;
  b: number | null;
  c: number | null;
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
}

export interface StructureData {
  prototype: string;
  formula: string;
  n_atoms: number;
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
