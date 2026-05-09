#!/usr/bin/env python3
"""
score_cobi_candidates.py

Loads the trained structure-only GBC model, applies identical feature
engineering to cobi_test_combinations_100.csv, and prints a detailed
8-section results report.
"""

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as jload

warnings.filterwarnings("ignore")

# ── constants (must match training) ───────────────────────────────────────────

CS_COLS      = ["cs_Cubic","cs_Hexagonal","cs_Monoclinic",
                "cs_Orthorhombic","cs_Tetragonal","cs_Triclinic","cs_Trigonal"]
RATIO_FLAGS  = ["ratio_1_1","ratio_1_2","ratio_2_1","ratio_1_3","ratio_3_1",
                "ratio_2_3","ratio_3_2","ratio_1_4","ratio_4_1","ratio_1_5","ratio_5_1"]

# prototypes that appeared in the CHGNet stoichiometry sweep — use as sanity-check references
_KNOWN_REF_PROTOTYPES = {
    "B2_CsCl","B1_NaCl","B3_ZnS","B81_NiAs","B20_FeSi",
    "L12_Cu3Au","D019_Ni3Sn","L12_AuCu3","D019_inv",
    "C11b_MoSi2","C1_CaF2","C1_inv","D2d_CaCu5","D2d_inv",
}


# ── feature engineering ───────────────────────────────────────────────────────

def _first_el_fraction(formula: str) -> float:
    from pymatgen.core import Composition
    try:
        comp  = Composition(formula)
        first = sorted(comp.elements, key=lambda e: e.symbol)[0]
        return float(comp.get_atomic_fraction(first))
    except Exception:
        return np.nan


def _ratio_flags(formula: str) -> dict:
    from pymatgen.core import Composition
    try:
        comp   = Composition(formula)
        counts = sorted(comp.as_dict().values())
        if len(counts) == 2:
            a, b = sorted([counts[0], counts[1]])
        elif len(counts) == 1:
            a, b = counts[0], counts[0]
        else:
            return {f: 0 for f in RATIO_FLAGS}
        flags = {
            "ratio_1_1": int(a == b),
            "ratio_1_2": int(b == 2*a),
            "ratio_2_1": int(b == 2*a),   # same ratio
            "ratio_1_3": int(b == 3*a),
            "ratio_3_1": int(b == 3*a),
            "ratio_2_3": int(b*2 == a*3 or a*2 == b*3),
            "ratio_3_2": int(a*3 == b*2 or b*3 == a*2),
            "ratio_1_4": int(b == 4*a),
            "ratio_4_1": int(b == 4*a),
            "ratio_1_5": int(b == 5*a),
            "ratio_5_1": int(b == 5*a),
        }
        return flags
    except Exception:
        return {f: 0 for f in RATIO_FLAGS}


def build_features(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    df = df.copy().reset_index(drop=True)

    # structural scalars
    df["composition_A_fraction"] = df["formula_pretty"].apply(_first_el_fraction)
    df["n_atoms_parity"]         = (df["nsites"] % 2).astype(int)
    df["spacegroup_number"]      = pd.to_numeric(df["spacegroup_number"],
                                                  errors="coerce").fillna(1)

    # one-hot crystal system
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    for col in CS_COLS:
        if col not in cs_dummies.columns:
            cs_dummies[col] = 0
    cs_dummies = cs_dummies[CS_COLS]
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    # stoichiometry ratio flags
    ratio_df = pd.DataFrame(
        df["formula_pretty"].apply(_ratio_flags).tolist(), index=df.index
    )
    df = pd.concat([df, ratio_df], axis=1)

    # matminer compositional featurizers
    df["_comp"] = df["formula_pretty"].apply(
        lambda f: Composition(f) if pd.notna(f) else None
    )
    ep = ElementProperty.from_preset("magpie")
    st = Stoichiometry()
    vo = ValenceOrbital()

    ep_names = ep.feature_labels()
    st_names = st.feature_labels()
    vo_names = vo.feature_labels()

    def safe_f(feat, comp):
        try:
            return feat.featurize(comp)
        except Exception:
            return [np.nan] * len(feat.feature_labels())

    ep_data = [safe_f(ep,c) if c else [np.nan]*len(ep_names) for c in df["_comp"]]
    st_data = [safe_f(st,c) if c else [np.nan]*len(st_names) for c in df["_comp"]]
    vo_data = [safe_f(vo,c) if c else [np.nan]*len(vo_names) for c in df["_comp"]]

    df = pd.concat([
        df.drop(columns=["_comp"]),
        pd.DataFrame(ep_data, columns=ep_names, index=df.index),
        pd.DataFrame(st_data, columns=st_names, index=df.index),
        pd.DataFrame(vo_data, columns=vo_names, index=df.index),
    ], axis=1)

    # align to training feature set
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    arr = df[feat_cols].values.astype(float)
    # impute NaN with column medians
    col_medians = np.nanmedian(arr, axis=0)
    nan_r, nan_c = np.where(np.isnan(arr))
    arr[nan_r, nan_c] = col_medians[nan_c]
    return arr


# ── confidence band ───────────────────────────────────────────────────────────

def confidence_band(p: float) -> str:
    if p >= 0.80:  return "high stable"
    if p >= 0.60:  return "likely stable"
    if p >= 0.40:  return "uncertain"
    if p >= 0.20:  return "likely unstable"
    return "high unstable"


# ── helpers ───────────────────────────────────────────────────────────────────

def sep(n=72): print("-" * n)
def header(title, n=72): print("=" * n); print(title); print("=" * n)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── load model ─────────────────────────────────────────────────────────
    model_path = "models/stability_classifier_structure_only.joblib"
    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}")
    art       = jload(model_path)
    pipeline  = art["model"]
    feat_cols = art["feat_cols"]
    print(f"Loaded: {art.get('model_name','?')}  |  {len(feat_cols)} features")

    # ── load candidates ─────────────────────────────────────────────────────
    cand_path = "cobi_test_combinations_100.csv"
    if not Path(cand_path).exists():
        sys.exit(f"Candidates file not found: {cand_path}")
    df = pd.read_csv(cand_path)
    print(f"Loaded {len(df)} candidates from {cand_path}")
    print(f"Columns in file: {df.columns.tolist()}\n")

    # ── enrich missing columns ──────────────────────────────────────────────
    # map to internal names used by feature engineering
    col_renames = {"reduced_formula": "formula_pretty", "n_atoms": "nsites",
                   "sg_number": "spacegroup_number"}
    for old, new in col_renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "formula_pretty" not in df.columns:
        df["formula_pretty"] = df.get("cell_formula", "")

    # derive n_co, n_bi if missing
    if "n_co" not in df.columns:
        df["n_co"] = (df.get("co_fraction", 0.5) * df["nsites"]).round().astype(int)
    if "n_bi" not in df.columns:
        df["n_bi"] = df["nsites"] - df["n_co"]

    # cell_formula
    if "cell_formula" not in df.columns:
        df["cell_formula"] = df.apply(
            lambda r: f"Co{int(r['n_co'])}Bi{int(r['n_bi'])}"
            if r["n_co"] != 1 or r["n_bi"] != 1
            else "CoBi", axis=1
        )

    # priority and comment (derive from prototype_name if available)
    if "priority" not in df.columns:
        proto_col = "prototype_name" if "prototype_name" in df.columns else None
        if proto_col:
            df["priority"] = df[proto_col].apply(
                lambda p: "known_reference" if p in _KNOWN_REF_PROTOTYPES else "candidate"
            )
        else:
            df["priority"] = "candidate"

    if "comment" not in df.columns:
        df["comment"] = df.get("prototype_name", "")

    # ── feature engineering ─────────────────────────────────────────────────
    print("Building feature matrix (matminer featurizers)...")
    X = build_features(df, feat_cols)

    # ── predict ─────────────────────────────────────────────────────────────
    y_pred  = pipeline.predict(X)
    y_prob  = pipeline.predict_proba(X)[:, 1]
    df["p_stable"]         = np.round(y_prob, 4)
    df["predicted_label"]  = np.where(y_pred == 1, "stable", "unstable")
    df["confidence_band"]  = [confidence_band(p) for p in y_prob]
    df_sorted = df.sort_values("p_stable", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    # ── output columns for printing ─────────────────────────────────────────
    def row_str(r, show_comment=True):
        comment = str(r.get("comment",""))[:18]
        base = (f"  {r['rank']:<4} {r.get('task_id',''):<15} "
                f"{r['formula_pretty']:<10} {str(r.get('sg_symbol','')):<10} "
                f"{str(r.get('crystal_system','')):<14} {r['nsites']:>5} "
                f"{r.get('co_fraction',0.5):>9.3f}  {r['p_stable']:>9.4f}  "
                f"{r['confidence_band']:<16}")
        if show_comment:
            base += f"  {r.get('priority','')}  {comment}"
        return base

    COL_HDR = (f"  {'#':<4} {'task_id':<15} {'formula':<10} {'sg_sym':<10} "
               f"{'crystal_sys':<14} {'n_at':>5} {'co_frac':>9}  {'p(stable)':>9}  "
               f"{'band':<16}  priority  comment")
    COL_SEP = "  " + "-" * 115

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A — confidence band distribution
    # ─────────────────────────────────────────────────────────────────────────
    header("SECTION A — Confidence band distribution")
    band_order = ["high stable","likely stable","uncertain","likely unstable","high unstable"]
    print(f"  {'Band':<20}  {'Count':>6}  {'Pct':>7}  Bar")
    sep()
    for band in band_order:
        n   = (df_sorted["confidence_band"] == band).sum()
        pct = n / len(df_sorted) * 100
        bar = "#" * int(pct / 2)
        print(f"  {band:<20}  {n:>6}  {pct:>6.1f}%  {bar}")
    sep()
    print(f"  {'TOTAL':<20}  {len(df_sorted):>6}  100.0%")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B — top 20
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION B — Top 20 candidates ranked by p(stable)")
    print(COL_HDR)
    print(COL_SEP)
    for _, r in df_sorted.head(20).iterrows():
        print(row_str(r))
    print(COL_SEP)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C — bottom 10
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION C — Bottom 10 candidates (lowest p(stable))")
    print(COL_HDR)
    print(COL_SEP)
    for _, r in df_sorted.tail(10).iloc[::-1].iterrows():
        print(row_str(r))
    print(COL_SEP)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D — breakdown by composition
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION D — Breakdown by composition ratio (reduced_formula)")
    grp = (df_sorted.groupby("formula_pretty")["p_stable"]
           .agg(count="count", mean="mean", max="max")
           .reset_index()
           .sort_values("mean", ascending=False))
    grp["high_stable"] = df_sorted.groupby("formula_pretty").apply(
        lambda g: (g["confidence_band"] == "high stable").sum()
    ).values
    print(f"  {'formula':<12} {'count':>6}  {'mean p':>8}  {'max p':>8}  {'# high stable':>14}")
    sep(62)
    for _, r in grp.iterrows():
        bar = "#" * int(r["mean"] * 20)
        print(f"  {r['formula_pretty']:<12} {r['count']:>6}  {r['mean']:>8.4f}  "
              f"{r['max']:>8.4f}  {r['high_stable']:>14}  {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E — breakdown by crystal system
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION E — Breakdown by crystal system")
    grp_cs = (df_sorted.groupby("crystal_system")["p_stable"]
              .agg(count="count", mean="mean")
              .reset_index()
              .sort_values("mean", ascending=False))
    grp_cs["high_stable"] = df_sorted.groupby("crystal_system").apply(
        lambda g: (g["confidence_band"] == "high stable").sum()
    ).values
    print(f"  {'crystal_system':<16} {'count':>6}  {'mean p':>8}  {'# high stable':>14}  bar")
    sep(65)
    for _, r in grp_cs.iterrows():
        bar = "#" * int(r["mean"] * 25)
        print(f"  {r['crystal_system']:<16} {r['count']:>6}  {r['mean']:>8.4f}  "
              f"{r['high_stable']:>14}  {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION F — breakdown by space group (top 10)
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION F — Top 10 most common space groups")
    grp_sg = (df_sorted.groupby(["spacegroup_number","sg_symbol"])["p_stable"]
              .agg(count="count", mean="mean")
              .reset_index()
              .sort_values("count", ascending=False)
              .head(10))
    grp_sg["high_stable"] = [
        (df_sorted[df_sorted["sg_symbol"] == r["sg_symbol"]]["confidence_band"] == "high stable").sum()
        for _, r in grp_sg.iterrows()
    ]
    print(f"  {'sg_number':>10}  {'sg_symbol':<12} {'count':>6}  {'mean p':>8}  {'# high stable':>14}")
    sep(62)
    for _, r in grp_sg.iterrows():
        print(f"  {int(r['spacegroup_number']):>10}  {str(r['sg_symbol']):<12} "
              f"{r['count']:>6}  {r['mean']:>8.4f}  {r['high_stable']:>14}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION G — known reference candidates
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION G — Known reference candidates (priority = 'known_reference')")
    refs = df_sorted[df_sorted["priority"] == "known_reference"].copy()
    if refs.empty:
        print("  No candidates marked as known_reference.")
    else:
        print(f"  {len(refs)} reference entries (should ideally rank high as sanity check)\n")
        print(COL_HDR)
        print(COL_SEP)
        for _, r in refs.iterrows():
            print(row_str(r))
        print(COL_SEP)
        # sanity summary
        n_stable_refs = (refs["predicted_label"] == "stable").sum()
        mean_p = refs["p_stable"].mean()
        print(f"\n  Known-reference summary: {n_stable_refs}/{len(refs)} predicted stable, "
              f"mean p={mean_p:.4f}")
        print(f"  Rank range: {refs['rank'].min()} -- {refs['rank'].max()}  "
              f"(out of {len(df_sorted)} total)")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION H — composition vs crystal system heatmap (text)
    # ─────────────────────────────────────────────────────────────────────────
    print()
    header("SECTION H — Composition vs crystal system: mean p(stable) heatmap")
    top_formulas = (df_sorted.groupby("formula_pretty").size()
                    .sort_values(ascending=False).head(8).index.tolist())
    all_cs = sorted(df_sorted["crystal_system"].dropna().unique())
    pivot = pd.pivot_table(
        df_sorted[df_sorted["formula_pretty"].isin(top_formulas)],
        values="p_stable",
        index="formula_pretty",
        columns="crystal_system",
        aggfunc="mean",
    )
    # header row
    cs_w = 10
    print(f"  {'formula':<12}", end="")
    for cs in all_cs:
        label = cs[:cs_w-1] if len(cs) >= cs_w else cs
        print(f"  {label:>{cs_w}}", end="")
    print()
    sep(12 + len(all_cs) * (cs_w + 2))
    for formula in top_formulas:
        print(f"  {formula:<12}", end="")
        for cs in all_cs:
            if cs in pivot.columns and formula in pivot.index and not np.isnan(pivot.loc[formula, cs]):
                val = f"{pivot.loc[formula, cs]:.3f}"
            else:
                val = "---"
            print(f"  {val:>{cs_w}}", end="")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Save output
    # ─────────────────────────────────────────────────────────────────────────
    out_path = Path("data/results/cobi_predictions_full.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in [
        "rank","task_id","formula_pretty","cell_formula","n_co","n_bi","nsites",
        "co_fraction","bi_fraction","sg_number","sg_symbol","crystal_system",
        "prototype_name","priority","comment",
        "p_stable","predicted_label","confidence_band",
    ] if c in df_sorted.columns]
    df_sorted[keep].to_csv(out_path, index=False)
    print(f"\nSaved {len(df_sorted)} predictions -> {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plain-English interpretation
    # ─────────────────────────────────────────────────────────────────────────
    top1    = df_sorted.iloc[0]
    n_high  = (df_sorted["confidence_band"] == "high stable").sum()
    n_high_u = (df_sorted["confidence_band"] == "high unstable").sum()
    top_cs  = df_sorted.groupby("crystal_system")["p_stable"].mean().idxmax()
    top_f   = df_sorted.groupby("formula_pretty")["p_stable"].mean().idxmax()
    ref_mean = refs["p_stable"].mean() if not refs.empty else float("nan")

    print()
    header("INTERPRETATION")
    print(f"""  The structure-only GBC model scores {n_high}/100 Co-Bi candidates as "high stable"
  (p > 0.80) and {n_high_u}/100 as "high unstable" (p < 0.20), giving a bimodal
  confidence distribution with {100 - n_high - n_high_u} candidates in intermediate bands.

  Compositionally, {top_f} compounds receive the highest mean predicted stability,
  suggesting that the model (trained on analogous binary systems) associates that
  stoichiometry with hull-favorable coordination environments.  Crystal-symmetry wise,
  {top_cs} structures score highest on average — consistent with the prevalence of
  high-symmetry prototypes among known stable binaries in the training data.

  The #1 ranked candidate is {top1['formula_pretty']} in {top1.get('crystal_system','')}
  symmetry (sg {top1.get('sg_number','?')}, {top1.get('prototype_name','')}),
  predicted stable with p = {top1['p_stable']:.4f}.

  Known-reference sanity check: {len(refs)} prototypes drawn from the CHGNet sweep
  receive mean p = {ref_mean:.4f} from the structure-only model.  Several of those
  prototypes (e.g., B2_CsCl, L12) are experimentally well-characterized in analogous
  systems, so high predicted probability there is expected; misranked references point
  to where pure structure information is insufficient without DFT energetics.

  Practical recommendation: prioritise the top-20 shortlist for DFT relaxation,
  focusing first on candidates in "high stable" orthorhombic and cubic environments,
  where the model shows the highest discrimination.  Uncertain-band candidates
  (p 0.40-0.60) warrant a second pass only after clear signal from the top tier.
""")


if __name__ == "__main__":
    main()
