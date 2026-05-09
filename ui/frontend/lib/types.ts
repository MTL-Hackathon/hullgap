export interface CandidateResult {
  composition: string;
  formula: string;
  n_atoms: number;
  x_B: number;
  formation_energy_eV_atom: number;
  e_above_hull_eV_atom: number;
  predicted_stable: boolean;
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
