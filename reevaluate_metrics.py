"""
Recompute MED metrics from saved outputs only (no model inference).
Uses med_eval_lib.py (Macro-EquiDiff paper methodology).
"""

import os
import sys
import json
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from med_eval_lib import (
    load_lines,
    eval_transformer_intermediate,
    eval_pipeline_final,
)

GT = os.path.join(ROOT, "GraphTransformer")


def main():
    parser = argparse.ArgumentParser(description="Reevaluate metrics from saved outputs")
    parser.add_argument("--predictions", default=os.path.join(GT, "datasets", "predictions.txt"))
    parser.add_argument("--targets", default=os.path.join(GT, "datasets", "final_test_dataset.txt"))
    parser.add_argument(
        "--pipeline_csv",
        default=os.path.join(ROOT, "intermediates", "top_performing_per_input.csv"),
    )
    parser.add_argument("--out", default=os.path.join(ROOT, "reevaluated_metrics.json"))
    args = parser.parse_args()

    preds = load_lines(args.predictions)
    tgts = load_lines(args.targets) if os.path.exists(args.targets) else None

    seen = set()
    fp, ft = [], []
    if tgts:
        for p, t in zip(preds, tgts):
            if t not in seen:
                seen.add(t)
                fp.append(p)
                ft.append(t)
    else:
        fp = preds

    train_csv = os.path.join(GT, "datasets", "data", "train.csv")
    from med_eval_lib import load_train_tgt_raw

    tr = eval_transformer_intermediate(fp, ft, load_train_tgt_raw(train_csv))

    try:
        from EDM.src.delinker_utils.sascorer import calculateScore as sa_scorer
    except ImportError:
        sa_scorer = None

    df = pd.read_csv(args.pipeline_csv)
    sc = next((c for c in df.columns if c.upper() == "SMILES"), df.columns[-1])
    macros = df[sc].dropna().astype(str).tolist()
    linkers = df["Linkers"].astype(str).tolist() if "Linkers" in df.columns else None

    pl = eval_pipeline_final(
        macros,
        sa_scorer=sa_scorer,
        linker_smiles=linkers,
        train_csv=train_csv if linkers else None,
    )

    print("=" * 65)
    print("  TRANSFORMER (intermediate * structures, n=%d)" % tr["n_total"])
    print("=" * 65)
    print("  Validity                 : %.2f%%" % tr["validity_pct"])
    print("  Uniqueness               : %.2f%%" % tr["uniqueness_pct"])
    print("  Transformer novelty      : %.2f%%  (Sec 2.2; NOT linker novelty)" % tr["novelty_pct"])
    print("  Avg Tanimoto             : %.4f" % tr["avg_tanimoto"])
    print("  Diversity                : %.2f%%" % tr["diversity_pct"])

    print("\n" + "=" * 65)
    print("  PIPELINE (final macrocycles, n=%d)" % pl["n_total"])
    print("=" * 65)
    print("  Validity                 : %.2f%%" % pl["validity_pct"])
    print("  Uniqueness               : %.2f%%" % pl["uniqueness_pct"])
    print("  Macrocyclisation         : %.2f%%" % pl["macrocyclisation_pct"])
    if pl.get("linker_novelty_pct") is not None:
        print(
            "  Linker novelty (Table 1) : %.2f%%  (%d/%d valid+unique; NOT transformer novelty)"
            % (
                pl["linker_novelty_pct"],
                pl.get("n_novel_linkers", 0),
                pl.get("n_valid_unique_for_linker_novelty", 0),
            )
        )
    print("  Avg QED                  : %.4f" % pl["avg_qed"])
    if pl.get("avg_sa") is not None:
        print("  Avg SA                   : %.4f" % pl["avg_sa"])

    summary = {"transformer": tr, "pipeline": pl}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved {args.out}")


if __name__ == "__main__":
    main()
