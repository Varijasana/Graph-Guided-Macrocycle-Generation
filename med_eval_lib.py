"""
Canonical MED metric definitions (Macro-EquiDiff paper methodology).
Used by reevaluate_metrics.py and the Kaggle dashboard notebook.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MolStandardize, QED, rdMolDescriptors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_train_tgt_raw(train_csv: str) -> List[str]:
    """Training intermediate tgt SMILES (with *), as in Macro-EquiDiff metrics.py."""
    if not os.path.exists(train_csv):
        return []
    df = pd.read_csv(train_csv, usecols=["tgt"])
    return df["tgt"].dropna().astype(str).tolist()


def maccs_tanimoto(mol1, mol2) -> float:
    fp1 = rdMolDescriptors.GetMACCSKeysFingerprint(mol1)
    fp2 = rdMolDescriptors.GetMACCSKeysFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def morgan_diversity_pct(mols: Sequence, radius: int = 2, n_bits: int = 1024, max_mols: int = 1000) -> float:
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0
    if len(mols) > max_mols:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(mols), max_mols, replace=False)
        mols = [mols[i] for i in idx]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float((1.0 - np.mean(sims)) * 100.0) if sims else 0.0


def is_macrocycle(mol, min_ring: int = 11) -> bool:
    if mol is None:
        return False
    return any(len(r) >= min_ring for r in mol.GetRingInfo().AtomRings())


# ---------------------------------------------------------------------------
# Transformer / intermediate-stage metrics (*-aware linkers)
# ---------------------------------------------------------------------------

def eval_transformer_intermediate(
    predictions: Sequence[str],
    targets: Optional[Sequence[str]] = None,
    train_tgt_raw: Optional[Sequence[str]] = None,
) -> dict:
    """
    Intermediate cyclisation structures (with * markers).
    """
    preds = list(predictions)
    n_total = len(preds)

    valid_pairs = []
    for smi in preds:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_pairs.append((smi, mol))

    n_valid = len(valid_pairs)
    validity_pct = (n_valid / n_total * 100.0) if n_total else 0.0

    canon_list = [canonicalize_smiles(s) for s, _ in valid_pairs]
    canon_list = [c for c in canon_list if c]
    uniqueness_pct = (len(set(canon_list)) / len(canon_list) * 100.0) if canon_list else 0.0

    # Transformer novelty — Macro-EquiDiff MacTransformer/metrics.py ("Novality")
    # Intermediate predicted SMILES (with *) vs training tgt; NOT EDM linker novelty.
    train_set = set(train_tgt_raw) if train_tgt_raw else set()
    novel_count = sum(1 for s in preds if s not in train_set)
    novelty_pct = (novel_count / len(preds) * 100.0) if preds else 0.0

    tanimoto_vals = []
    if targets is not None:
        for p, t in zip(preds, targets):
            p_clean = p.replace("(*)", "").replace("[*]", "").replace("*", "")
            t_clean = t.replace("(*)", "").replace("[*]", "").replace("*", "")
            m1, m2 = Chem.MolFromSmiles(p_clean), Chem.MolFromSmiles(t_clean)
            if m1 and m2:
                tanimoto_vals.append(maccs_tanimoto(m1, m2))
    avg_tanimoto = float(np.mean(tanimoto_vals)) if tanimoto_vals else 0.0

    valid_mols = [m for _, m in valid_pairs]
    diversity_pct = morgan_diversity_pct(valid_mols)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_pct": round(validity_pct, 4),
        "uniqueness_pct": round(uniqueness_pct, 4),
        "novelty_pct": round(novelty_pct, 4),
        "avg_tanimoto": round(avg_tanimoto, 4),
        "diversity_pct": round(diversity_pct, 4),
        "stage": "transformer_intermediate",
    }


# ---------------------------------------------------------------------------
# Linker novelty (Table 1 / EDM compute_metrics.py — final macrocycles only)
# ---------------------------------------------------------------------------

def canonicalize_linker_smiles(linker_smi: str) -> Optional[str]:
    """Same canonicalization as Macro-EquiDiff EDM/compute_metrics.py novelty block."""
    mol = Chem.MolFromSmiles(str(linker_smi))
    if mol is None:
        return str(linker_smi).strip()
    Chem.RemoveStereochemistry(mol)
    try:
        return MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(mol))
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def load_training_linkers_canonical(
    train_csv: str,
    cache_path: Optional[str] = None,
) -> set:
    """
    Build training linker reference from train.csv tgt (with * attachment markers).
    Cached to disk on first use. Matches the training corpus used by MED.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    raw = load_train_tgt_raw(train_csv)
    canon = set()
    for smi in raw:
        c = canonicalize_linker_smiles(smi)
        if c:
            canon.add(c)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for s in sorted(canon):
                f.write(s + "\n")
    return canon


def eval_linker_novelty_table1(
    macro_smiles: Sequence[str],
    linker_smiles: Sequence[str],
    training_linkers: set,
) -> dict:
    """
    Paper Table 1 / Macro-EquiDiff:
    % of novel linkers (not in training) among valid AND unique final macrocycles.

    macro_smiles and linker_smiles must be aligned row-wise (same pipeline cohort).
    """
    pairs = list(zip(macro_smiles, linker_smiles))
    valid_pairs = []
    for macro_smi, linker_smi in pairs:
        if not macro_smi or not linker_smi or pd.isna(linker_smi):
            continue
        mol = Chem.MolFromSmiles(str(macro_smi))
        if mol is None:
            continue
        valid_pairs.append((Chem.MolToSmiles(mol, canonical=True), str(linker_smi)))

    # Unique among valid (paper: valid and unique molecules)
    seen_macro = set()
    unique_pairs = []
    for canon_macro, linker_smi in valid_pairs:
        if canon_macro in seen_macro:
            continue
        seen_macro.add(canon_macro)
        unique_pairs.append((canon_macro, linker_smi))

    novel_count = 0
    for _, linker_smi in unique_pairs:
        canon_linker = canonicalize_linker_smiles(linker_smi)
        if canon_linker not in training_linkers:
            novel_count += 1

    n_eval = len(unique_pairs)
    pct = (novel_count / n_eval * 100.0) if n_eval else 0.0
    return {
        "n_total": len(pairs),
        "n_valid": len(valid_pairs),
        "n_valid_unique": n_eval,
        "n_novel_linkers": novel_count,
        "linker_novelty_pct": round(pct, 4),
        "stage": "pipeline_linker_novelty_table1",
    }


# ---------------------------------------------------------------------------
# Full pipeline / final macrocycle metrics (ring-closed)
# ---------------------------------------------------------------------------

def eval_pipeline_final(
    smiles_list: Sequence[str],
    sa_scorer=None,
    linker_smiles: Optional[Sequence[str]] = None,
    train_csv: Optional[str] = None,
) -> dict:
    valid_mols, valid_smiles = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m:
            valid_mols.append(m)
            valid_smiles.append(Chem.MolToSmiles(m, canonical=True))

    total = len(smiles_list)
    n_valid = len(valid_mols)
    validity_pct = (n_valid / total * 100.0) if total else 0.0
    uniqueness_pct = (len(set(valid_smiles)) / n_valid * 100.0) if n_valid else 0.0
    macro_count = sum(1 for m in valid_mols if is_macrocycle(m))
    macrocyclisation_pct = (macro_count / n_valid * 100.0) if n_valid else 0.0

    qed_scores = [QED.qed(m) for m in valid_mols]
    sa_scores = []
    if sa_scorer is not None:
        try:
            for m in valid_mols:
                s = sa_scorer(m)
                if s is not None:
                    sa_scores.append(float(s))
        except Exception:
            sa_scores = []

    out = {
        "n_total": total,
        "n_valid": n_valid,
        "n_macrocycles": macro_count,
        "validity_pct": round(validity_pct, 4),
        "uniqueness_pct": round(uniqueness_pct, 4),
        "macrocyclisation_pct": round(macrocyclisation_pct, 4),
        "avg_qed": round(float(np.mean(qed_scores)), 4) if qed_scores else 0.0,
        "avg_sa": round(float(np.mean(sa_scores)), 4) if sa_scores else None,
        "stage": "pipeline_final_macrocycles",
    }

    if linker_smiles is not None and train_csv and os.path.exists(train_csv):
        cache = os.path.join(os.path.dirname(train_csv), "training_linkers_canon.txt")
        train_linkers = load_training_linkers_canonical(train_csv, cache_path=cache)
        ln = eval_linker_novelty_table1(smiles_list, linker_smiles, train_linkers)
        out["linker_novelty_pct"] = ln["linker_novelty_pct"]
        out["n_valid_unique_for_linker_novelty"] = ln["n_valid_unique"]
        out["n_novel_linkers"] = ln["n_novel_linkers"]

    return out
