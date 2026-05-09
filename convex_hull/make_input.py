# 00_make_inputs.py
#!/usr/bin/env python3
"""
Create AIRSS/CASTEP seed files for a Co-Bi convex-hull search at 50 GPa.

Run:
    python3 00_make_inputs.py
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

ROOT = Path.cwd()
SEED_DIR = ROOT / "seeds"
SEED_DIR.mkdir(exist_ok=True)

PRESSURE_GPA = 50.0

# Initial AIRSS volumes, in A^3/atom. These are only starting volumes;
# CASTEP relaxes the cells at 50 GPa.
VARVOL_CO = 9.0
VARVOL_BI = 18.0
KPOINTS_MP_SPACING = 0.05
MAX_ATOMS_FIXED_FORMULA = 24

# Fixed stoichiometries added to force sampling around the full composition range.
# CoBi3, CoBi2, and CoBi are included explicitly.
FIXED_RATIOS = [
    (1, 8), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3),
    (1, 1), (3, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1),
]

PARAM = """task               : GeometryOptimization
xc_functional      : PBE
spin_polarised     : true
basis_precision    : precise
metals_method      : dm
smearing_width     : 0.10 eV
elec_energy_tol    : 1.0e-6 eV
max_scf_cycles     : 120
geom_method        : BFGS
geom_max_iter      : 200
geom_force_tol     : 0.03 eV/Ang
geom_stress_tol    : 0.10 GPa
geom_disp_tol      : 0.001 Ang
"""

COMMON_DIRECTIVES = f"""#SYMMOPS=1-8
##SGRANK=20
#ADJGEN=0-1
#SLACK=0.25
#OVERLAP=0.1
#MINSEP=1.7-3.2 AUTO
#COMPACT
##SYSTEM={{Rhom,Tric,Mono,Cubi,Hexa,Orth,Tetra}}

KPOINTS_MP_SPACING : {KPOINTS_MP_SPACING}
SYMMETRY_GENERATE
SNAP_TO_SYMMETRY

%BLOCK SPECIES_POT
QC5
%ENDBLOCK SPECIES_POT

%BLOCK EXTERNAL_PRESSURE
0 0 0
0 0
0
%ENDBLOCK EXTERNAL_PRESSURE
"""


def mixture_varvol(n_co: int, n_bi: int) -> float:
    return (n_co * VARVOL_CO + n_bi * VARVOL_BI) / (n_co + n_bi)


def reduced_formula(n_co: int, n_bi: int) -> str:
    if n_co < 0 or n_bi < 0 or n_co + n_bi == 0:
        raise ValueError("invalid composition")

    g = gcd(n_co, n_bi) if n_co and n_bi else max(n_co, n_bi)
    co = n_co // g if n_co else 0
    bi = n_bi // g if n_bi else 0

    def part(el: str, n: int) -> str:
        if n == 0:
            return ""
        return el if n == 1 else f"{el}{n}"

    return part("Co", co) + part("Bi", bi)


def write_seed(name: str, cell_text: str) -> None:
    (SEED_DIR / f"{name}.cell").write_text(cell_text.strip() + "\n")
    (SEED_DIR / f"{name}.param").write_text(PARAM.strip() + "\n")


# Pure elemental reference searches. These are required for formation enthalpies.
write_seed(
    "Co",
    f"""#VARVOL={VARVOL_CO:.3f}
#SPECIES=Co
#NATOM=1-24
{COMMON_DIRECTIVES}""",
)

write_seed(
    "Bi",
    f"""#VARVOL={VARVOL_BI:.3f}
#SPECIES=Bi
#NATOM=1-24
{COMMON_DIRECTIVES}""",
)

# Broad variable-composition binary search.
write_seed(
    "CoBi_variable",
    f"""#VARVOL={mixture_varvol(1, 1):.3f}
#SPECIES=Co,Bi
#NATOM=2-24
#FOCUS=2
#NFORM=1
{COMMON_DIRECTIVES}""",
)

seed_names = ["Co", "Bi", "CoBi_variable"]

# Fixed-composition searches.
for n_co, n_bi in FIXED_RATIOS:
    name = reduced_formula(n_co, n_bi)
    if name in seed_names:
        continue

    max_nform = max(1, MAX_ATOMS_FIXED_FORMULA // (n_co + n_bi))
    nform_line = "#NFORM=1" if max_nform == 1 else f"#NFORM=1-{max_nform}"

    cell = f"""#VARVOL={mixture_varvol(n_co, n_bi):.3f}
#SPECIES=Co%NUM={n_co},Bi%NUM={n_bi}
{nform_line}
{COMMON_DIRECTIVES}"""
    write_seed(name, cell)
    seed_names.append(name)

(ROOT / "seeds.txt").write_text("\n".join(seed_names) + "\n")

print(f"Wrote {len(seed_names)} AIRSS seeds to {SEED_DIR}")
print("Submit example:")
print("  sbatch --array=0-$(($(wc -l < seeds.txt)-1)) 01_run_airss.sh")
print("or local test:")
print("  MAX=20 ./01_run_airss.sh CoBi3 CoBi2 CoBi")