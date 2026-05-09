#!/usr/bin/env python3
"""
save_model.py

Re-exports the trained structure-only classifier from the training artifact
into the two files expected by screen_candidates.py:

  models/structure_only_classifier.joblib  -- bare Pipeline object
  models/feature_columns.json             -- ordered list of feature names

Usage:
    python save_model.py
    python save_model.py --src models/stability_classifier_structure_only.joblib
"""

import argparse
import json
from pathlib import Path

from joblib import dump, load as jload


def main():
    parser = argparse.ArgumentParser(description="Export trained model for screen_candidates.py")
    parser.add_argument("--src",  default="models/stability_classifier_structure_only.joblib")
    parser.add_argument("--model-out",    default="models/structure_only_classifier.joblib")
    parser.add_argument("--features-out", default="models/feature_columns.json")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Source model not found: {src}")

    artifact  = jload(src)
    pipeline  = artifact["model"]
    feat_cols = artifact["feat_cols"]

    dump(pipeline, args.model_out)
    print(f"Saved pipeline  -> {args.model_out}")

    with open(args.features_out, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"Saved feat_cols -> {args.features_out}  ({len(feat_cols)} features)")


if __name__ == "__main__":
    main()
