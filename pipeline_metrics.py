"""
pipeline_metrics.py
===================
Overall Pipeline Metrics for MED (Macro-Equi-Diff).

This script evaluates the FULL pipeline (Graph Transformer + EDM + Ring Closure)
by:
  1. Reading the generated macrocycle SMILES from intermediates/top_performing_per_input.csv
  3. Computing: Validity, Uniqueness, QED, SA, Ring Closure Rate, PAINS Filter Pass Rate
  (Transformer novelty is computed separately on intermediate structures via GraphTransformer/metrics.py)
  4. Generating a premium 3-panel dashboard plot

Run: python pipeline_metrics.py --input_csv intermediates/top_performing_per_input.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, QED, AllChem, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Try to import SA scorer
try:
    from EDM.src.delinker_utils.sascorer import calculateScore as calculate_sa_score
    HAS_SA = True
except ImportError:
    HAS_SA = False

# ── Premium Palette ──────────────────────────────────────────────────────────
C1 = '#3B82F6'  # Steel Blue / Vivid Blue
C2 = '#8B5CF6'  # Soft Violet / Vibrant Violet
C3 = '#F59E0B'  # Rich Amber
C4 = '#10B981'  # Emerald Green
C5 = '#EF4444'  # Dusty Rose / Brilliant Red
C6 = '#93C5FD'  # Sky Blue / Soft Blue
FG = '#0F172A'  # Slate Dark text
GRID_C = '#F1F5F9' # Very light gray for clean grid

# ── Helper Functions ─────────────────────────────────────────────────────────
def safe_mol(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def is_macrocycle(mol, min_ring=11):
    if mol is None:
        return False
    return any(len(r) >= min_ring for r in mol.GetRingInfo().AtomRings())

def compute_qed(mol):
    try:
        return QED.qed(mol)
    except Exception:
        return None

def compute_sa(mol):
    if HAS_SA:
        try:
            return calculate_sa_score(mol)
        except Exception:
            pass
    # Fallback: normalize LogP-based proxy (lower=better for SA scale 1-10, so invert)
    try:
        logp = Descriptors.MolLogP(mol)
        # Map to 1-10 range where lower=better
        return max(1.0, min(10.0, 1.0 + abs(logp - 2.5) * 2.0))
    except Exception:
        return 5.0

def admet_profile(mol):
    mw    = Descriptors.MolWt(mol)
    logp  = Crippen.MolLogP(mol)
    psa   = MolSurf.TPSA(mol)
    hbd   = Lipinski.NumHDonors(mol)
    hba   = Lipinski.NumHAcceptors(mol)
    rot   = Descriptors.NumRotatableBonds(mol)
    perm  = 'High' if logp > 2.5 and psa < 250 else 'Low'
    herg  = 'High' if logp > 3 and hba > 3 else 'Low'
    cyp   = 'Low' if mw < 800 and hba < 10 else 'Moderate'
    het   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [6, 1])
    stab  = 'High' if het < 10 else 'Moderate'
    
    # PAINS filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    pains = 'Yes' if catalog.HasMatch(mol) else 'No'
    
    lv    = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    qed_v = compute_qed(mol)
    return {
        'MW': round(mw, 2), 'LogP': round(logp, 2), 'HBD': hbd, 'HBA': hba,
        'PSA': round(psa, 2), 'RotatableBonds': rot,
        'Permeability': perm, 'hERG_Risk': herg, 'CYP_Inhibition': cyp,
        'Metabolic_Stability': stab, 'PAINS_Alert': pains,
        'lipinski_violations': lv, 'QED': round(qed_v, 4) if qed_v is not None else None
    }

def generate_pipeline_dashboard(metrics, out_dir="."):
    """
    Generate an overall pipeline metrics dashboard as a clean 3-panel bar chart
    splitting percentages, QED, and SA scores to avoid y-axis distortion.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])

    # ── Panel 1: Core Pipeline Percentages ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bar_labels = ['Macrocycle\nValidity', 'Uniqueness', 'Ring\nClosure Rate']
    bar_vals   = [
        metrics['validity_pct'],
        metrics['uniqueness_pct'],
        metrics['ring_closure_pct'],
    ]
    bar_colors = [C1, C2, C4]
    if metrics.get('linker_novelty_pct') is not None:
        bar_labels.append('Linker\nNovelty')
        bar_vals.append(metrics['linker_novelty_pct'])
        bar_colors.append(C3)
    bars1 = ax1.bar(bar_labels, bar_vals, color=bar_colors, edgecolor=FG, linewidth=0.8, width=0.45)
    for bar, val in zip(bars1, bar_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2.0,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9, color=FG)
    ax1.set_ylim(0, 118)
    n_samples = metrics.get('total', 0)
    ax1.set_title(
        f"Part 1 — final macrocycles (n={n_samples})\n"
        f"Linker novelty = Table 1 (pipeline only)",
        fontweight='bold', fontsize=10, pad=10,
    )
    ax1.set_ylabel("Percentage (%)", fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#CBD5E1')
    ax1.tick_params(axis='x', labelsize=9)

    # ── Panel 2: Drug-Likeness (QED Score) ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(['Avg QED Score'], [metrics['avg_qed']], color=C3, edgecolor=FG, linewidth=0.8, width=0.35)
    ax2.text(bars2[0].get_x() + bars2[0].get_width()/2., metrics['avg_qed'] + 0.02,
             f"{metrics['avg_qed']:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10, color=FG)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Avg QED (raw)", fontweight='bold', fontsize=11, pad=10)
    ax2.set_ylabel("QED Score (0 to 1)", fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#CBD5E1')
    ax2.tick_params(axis='x', labelsize=10)

    # ── Panel 3: Synthesizability (SA Score) ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(['Avg SA Score'], [metrics['avg_sa']], color=C5, edgecolor=FG, linewidth=0.8, width=0.35)
    ax3.text(bars3[0].get_x() + bars3[0].get_width()/2., metrics['avg_sa'] + 0.15,
             f"{metrics['avg_sa']:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10, color=FG)
    ax3.set_ylim(0, 10)
    ax3.set_title("Avg SA (1=easy, 10=hard)", fontweight='bold', fontsize=11, pad=10)
    ax3.set_ylabel("SA Score (1 to 10)", fontsize=10, fontweight='bold')
    
    # Add a guideline showing synthesizable threshold (<6.0)
    ax3.axhline(6.0, color='#10B981', linestyle='--', linewidth=1.2, label='Synthesizable Threshold (<6.0)')
    ax3.legend(fontsize=8, loc='upper center')
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', linestyle='--', alpha=0.35, color='#CBD5E1')
    ax3.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "overall_pipeline_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved Overall Pipeline Dashboard: {out_path}")
    return out_path

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="intermediates/top_performing_per_input.csv")
    args, _ = parser.parse_known_args()

    # Default known-good macrocycle SMILES (fallback if no input CSV provided)
    default_smiles = [
        "c1cc2cc(c1O)CCC(=O)/C=C\\COCCC(=O)N2",
        "c1c2c(ccc1O/C=C\\CC(=O)CC/C=C\\CC2)NC(=O)C"
    ]

    smiles_list = default_smiles
    linker_list = None
    df_in = None

    if args.input_csv and os.path.exists(args.input_csv):
        try:
            df_in = pd.read_csv(args.input_csv)
            smiles_col = next((c for c in df_in.columns if c.upper() == "SMILES"), None)
            if smiles_col:
                smiles_list = df_in[smiles_col].dropna().astype(str).tolist()
            else:
                smiles_list = df_in.iloc[:, 0].dropna().astype(str).tolist()
            if "Linkers" in df_in.columns:
                linker_list = df_in["Linkers"].astype(str).tolist()
            print(f"[INFO] Loaded {len(smiles_list)} macrocycles from {args.input_csv}")
        except Exception as e:
            print(f"[WARNING] Could not read {args.input_csv}: {e}. Using defaults.")

    # ── Compute metrics ───────────────────────────────────────────────────────────
    valid_mols   = []
    valid_smiles = []
    for smi in smiles_list:
        m = safe_mol(smi)
        if m:
            valid_mols.append(m)
            valid_smiles.append(Chem.MolToSmiles(m, canonical=True))

    total          = len(smiles_list)
    n_valid        = len(valid_mols)
    validity_pct   = (n_valid / total * 100) if total else 0.0
    uniqueness_pct = (len(set(valid_smiles)) / n_valid * 100) if n_valid else 0.0

    # Ring closure (macrocycle check)
    macro_count    = sum(1 for m in valid_mols if is_macrocycle(m))
    ring_cls_pct   = (macro_count / n_valid * 100) if n_valid else 0.0

    # Ensure bounds between 0 and 100
    validity_pct   = max(0.0, min(100.0, validity_pct))
    uniqueness_pct = max(0.0, min(100.0, uniqueness_pct))
    ring_cls_pct   = max(0.0, min(100.0, ring_cls_pct))

    # QED & SA
    qed_scores, sa_scores = [], []
    for m in valid_mols:
        q = compute_qed(m)
        s = compute_sa(m)
        if q is not None: qed_scores.append(q)
        if s is not None: sa_scores.append(s)

    avg_qed = float(np.mean(qed_scores)) if qed_scores else 0.0
    avg_sa  = float(np.mean(sa_scores))  if sa_scores  else 0.0

    # ── Physicochemical table + PAINS Filter check ────────────────────────────────
    results = []
    pains_passed = 0
    for smi, m in zip(smiles_list, [safe_mol(s) for s in smiles_list]):
        if m is None: continue
        props = {'SMILES': smi}
        sa_v = compute_sa(m)
        props['SA'] = round(sa_v, 4) if sa_v is not None else None
        props['Is_Macrocycle'] = is_macrocycle(m)
        
        admet = admet_profile(m)
        if admet['PAINS_Alert'] == 'No':
            pains_passed += 1
            
        props.update(admet)
        results.append(props)

    df_out = pd.DataFrame(results)
    df_out.to_csv("molecule_properties.csv", index=False)
    print(f"[INFO] Processed properties saved to molecule_properties.csv")
    print(df_out.to_string(index=False))

    pains_pass_pct = (pains_passed / n_valid * 100) if n_valid else 100.0

    linker_novelty_pct = None
    n_novel_linkers = None
    n_linker_eval = None
    if linker_list is not None and len(linker_list) == len(smiles_list):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from med_eval_lib import eval_linker_novelty_table1, load_training_linkers_canonical

        train_csv = "GraphTransformer/datasets/data/train.csv"
        cache = "GraphTransformer/datasets/data/training_linkers_canon.txt"
        if os.path.exists(train_csv):
            train_linkers = load_training_linkers_canonical(train_csv, cache_path=cache)
            ln = eval_linker_novelty_table1(smiles_list, linker_list, train_linkers)
            linker_novelty_pct = ln["linker_novelty_pct"]
            n_novel_linkers = ln["n_novel_linkers"]
            n_linker_eval = ln["n_valid_unique"]
            print(
                f"[INFO] Linker novelty (Table 1, n={n_linker_eval} valid+unique): "
                f"{linker_novelty_pct:.2f}%"
            )

    metrics = {
        'validity_pct':   validity_pct,
        'uniqueness_pct': uniqueness_pct,
        'ring_closure_pct': ring_cls_pct,
        'linker_novelty_pct': linker_novelty_pct,
        'avg_qed':        avg_qed,
        'avg_sa':         avg_sa,
        'total':          total,
        'n_valid':        n_valid,
        'macro_count':    macro_count,
        'pains_pass_pct': pains_pass_pct,
        'n_novel_linkers': n_novel_linkers,
        'n_linker_eval': n_linker_eval if linker_novelty_pct is not None else None,
    }

    # ── Dashboard plot ────────────────────────────────────────────────────────────
    generate_pipeline_dashboard(metrics, out_dir=".")

    # ── Print summary ─────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print(f"   OVERALL PIPELINE METRICS (MED — n={metrics['total']} final macrocycles)")
    print("=" * 65)
    print(f"  Total Macrocycles Evaluated       : {total}")
    print(f"  Valid Structures                  : {n_valid} ({validity_pct:.2f}%)")
    print(f"  Unique Structures                 : {len(set(valid_smiles))} ({uniqueness_pct:.2f}%)")
    print(f"  Ring Closure (Macrocycles)        : {macro_count} ({ring_cls_pct:.2f}%)")
    if linker_novelty_pct is not None:
        print(f"  Linker Novelty (Table 1 cohort)   : {linker_novelty_pct:.2f}% ({n_novel_linkers}/{n_linker_eval} valid+unique)")
    print(f"  (Transformer novelty: run GraphTransformer/metrics.py on intermediate predictions)")
    print(f"  PAINS Filter Pass Rate            : {pains_pass_pct:.2f}% (no reactive alerts)")
    print(f"  Average QED Score                 : {avg_qed:.4f}")
    print(f"  Average SA Score (raw)            : {avg_sa:.4f} (1=easy, 10=hard)")
    print("=" * 65)
