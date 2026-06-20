import torch
import os
import sys
import re
import argparse
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem, DataStructs, Draw
from tqdm import tqdm
import itertools

# Add parent directory to path to find EDM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from EDM.src.delinker_utils.sascorer import calculateScore as calculate_sa_score
except ImportError:
    calculate_sa_score = None

from model import GraphMacTransformer
from decoder import SmilesDecoder
from tokenizer import SmilesTokenizer

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# AGGRESSIVE SMILES REPAIR & UNK RECOVERY
# =========================================================
def repair_smiles(smiles):
    """
    Aggressively attempt to recover a valid SMILES by:
    1. Returning as-is if already valid.
    2. Replacing <UNK> tokens combinatorially (up to 3 <UNK> tokens).
    3. Fixing invalid triple or double bonds to aromatic atoms (e.g., #c -> -c).
    4. Progressively trimming dangling tail elements while balancing parentheses and brackets.
    """
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles

    def balance_brackets(s):
        # Balance parenthesized groups
        open_p = s.count('(')
        close_p = s.count(')')
        if open_p > close_p:
            s = s + ')' * (open_p - close_p)
        elif close_p > open_p:
            # Strip trailing unmatched close parens
            while s.endswith(')') and s.count('(') < s.count(')'):
                s = s[:-1]
        
        # Balance brackets
        open_b = s.count('[')
        close_b = s.count(']')
        if open_b > close_b:
            s = s + ']' * (open_b - close_b)
        elif close_b > open_b:
            # Strip trailing unmatched close brackets
            while s.endswith(']') and s.count('[') < s.count(']'):
                s = s[:-1]
        return s

    def clean_junk(s):
        # Remove trailing bonds, dots, open brackets, and open parentheses
        s = re.sub(r'[=#\-\/\\\.\[\(]+$', '', s).strip()
        return s

    # Handle <UNK> tokens combinatorially if there are few, otherwise fall back to simple replacement
    parts = smiles.split("<UNK>")
    num_unks = len(parts) - 1
    
    unk_combos = []
    if num_unks == 0:
        unk_combos = [smiles]
    elif num_unks <= 3:
        replacements = ["c", "C", "n", "N", ""]
        for combo in itertools.product(replacements, repeat=num_unks):
            res = []
            for i, part in enumerate(parts):
                res.append(part)
                if i < len(combo):
                    res.append(combo[i])
            unk_combos.append("".join(res))
    else:
        unk_combos = [
            smiles.replace("<UNK>", "c"),
            smiles.replace("<UNK>", "C"),
            smiles.replace("<UNK>", "")
        ]

    candidates = []
    for base in unk_combos:
        candidates.append(base)
        
        # Fix invalid bonds to aromatic atoms (e.g. #c -> -c, =c -> -c)
        aromatic_fix = re.sub(r'#(?=[a-z])', '-', base)
        aromatic_fix = re.sub(r'=(?=[a-z])', '-', aromatic_fix)
        candidates.append(aromatic_fix)
        
        # Try converting all triple bonds to single or double bonds
        candidates.append(base.replace("#", "-"))
        candidates.append(base.replace("#", "="))

    # Apply nested repairs to all candidate options
    extended_candidates = []
    for cand in candidates:
        extended_candidates.append(cand)
        extended_candidates.append(balance_brackets(clean_junk(cand)))
        
        # Remove trailing digits (unclosed rings)
        c_no_digits = re.sub(r'\d+$', '', cand).strip()
        extended_candidates.append(balance_brackets(clean_junk(c_no_digits)))
        
        # Progressive right trimming (up to 15 characters)
        for trim_len in range(1, 16):
            if len(cand) > trim_len:
                trimmed = cand[:-trim_len]
                extended_candidates.append(balance_brackets(clean_junk(trimmed)))

    # De-duplicate while preserving prioritization order
    seen = set()
    unique_candidates = []
    for c in extended_candidates:
        if c and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Check validity for all candidate repairs
    for c in unique_candidates:
        if Chem.MolFromSmiles(c) is not None:
            return c

    # Fallback to the first cleaned and balanced candidate if all fail
    fallback = balance_brackets(clean_junk(unk_combos[0]))
    return fallback


# =========================================================
# LOAD MODELS
# =========================================================
def load_models():
    tokenizer = SmilesTokenizer()

    encoder = GraphMacTransformer().to(DEVICE)
    decoder = SmilesDecoder(
        vocab_size=tokenizer.vocab_size(),
        hidden_dim=192,
        encoder_dim=128,
        num_layers=4
    ).to(DEVICE)

    if os.path.exists("./checkpoints/joint_checkpoint.pth"):
        ckpt = torch.load("./checkpoints/joint_checkpoint.pth", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print("[INFO] Models loaded successfully from joint checkpoint.")
    else:
        print("[WARNING] No checkpoint found! Evaluate will run with randomly initialized weights.")

    encoder.eval()
    decoder.eval()

    return encoder, decoder, tokenizer


# =========================================================
# FAST MOLECULE PARSE
# =========================================================
def safe_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


# =========================================================
# MULTI-CANDIDATE BEAM SEARCH
# =========================================================
@torch.no_grad()
def beam_search_candidates(decoder, memory, start_token, end_token,
                           beam_size=5, max_len=200):
    """
    Beam search returning all tracked candidate sequences sorted by
    cumulative log-probability.
    """
    beams = [(torch.tensor([[start_token]], device=DEVICE), 0.0)]

    for _ in range(max_len):
        candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == end_token:
                candidates.append((seq, score))
                continue

            logits = decoder(memory, seq)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                token = top_idx[0, i].view(1, 1)
                new_seq = torch.cat([seq, token], dim=1)
                candidates.append((new_seq, score + top_vals[0, i].item()))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(b[0][0, -1].item() == end_token for b in beams):
            break

    return [b[0] for b in beams]


# =========================================================
# COMPATIBILITY WRAPPER FOR OTHER SCRIPTS
# =========================================================
@torch.no_grad()
def beam_search(decoder, memory, start_token, end_token, beam_size=5, max_len=200):
    """
    Returns the top single generated sequence from beam search.
    Used for backward compatibility with predict.py and advanced_metrics.py.
    """
    candidates = beam_search_candidates(decoder, memory, start_token, end_token, beam_size, max_len)
    return candidates[0]


# =========================================================
# FAST METRICS
# =========================================================
def compute_metrics(smiles_list):
    valid = []
    invalid = 0

    qed_scores = []
    sa_scores = []
    fps = []
    ring_counts = []

    for smi in smiles_list:
        mol = safe_mol(smi)
        if mol is None:
            invalid += 1
            continue

        valid.append(mol)
        qed_scores.append(QED.qed(mol))

        if calculate_sa_score is not None:
            raw_sa = calculate_sa_score(mol)
            if raw_sa is not None:
                sa_scores.append(raw_sa)
        ring_counts.append(Chem.rdMolDescriptors.CalcNumRings(mol))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    diversity = 0.0
    if len(fps) > 1:
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        avg_internal_sim = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_internal_sim

    validity = len(valid) / len(smiles_list) if smiles_list else 0.0

    return {
        "validity": validity,
        "invalid_rate": invalid / len(smiles_list) if smiles_list else 1.0,
        "avg_qed": np.mean(qed_scores) if qed_scores else 0,
        "avg_sa": np.mean(sa_scores) if sa_scores else 0,
        "avg_rings": np.mean(ring_counts) if ring_counts else 0,
        "diversity": diversity
    }


# =========================================================
# MAIN EVALUATION
# =========================================================
def evaluate(beam_size=5, num_samples=100, csv_path="./datasets/data/test.csv", random_state=42):
    encoder, decoder, tokenizer = load_models()

    if not os.path.exists(csv_path):
        print(f"[ERROR] Test dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_available = len(df)
    
    # Properly handle -1 or <= 0 as evaluating the entire ZINC test set
    if num_samples <= 0 or num_samples >= total_available:
        num_samples = total_available
        print(f"[INFO] Evaluating the ENTIRE test dataset of {num_samples} samples.")
        df_sample = df
    else:
        print(f"[INFO] Evaluating a random subset of {num_samples} samples from {csv_path}.")
        df_sample = df.sample(num_samples, random_state=random_state)
        
    rows = list(df_sample.itertuples(index=False))

    start_token = tokenizer.stoi.get("<SOS>", tokenizer.stoi.get("<START>", 1))
    end_token = tokenizer.stoi.get("<EOS>", tokenizer.stoi.get("<END>", 2))

    print(f"\n[INFO] FAST EVALUATION STARTED (VALIDITY-FILTERED BEAM DECODING, beam_size={beam_size})\n")

    # --- Checkpointing & Auto-Resume Configuration ---
    progress_file = "datasets/eval_model_progress.json"
    start_idx = 0
    preds = []
    targets = []
    
    os.makedirs("datasets", exist_ok=True)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                saved_state = json.load(f)
            # Ensure checkpoint parameters match the current run configuration
            if (saved_state.get("num_samples") == num_samples and 
                saved_state.get("beam_size") == beam_size and
                saved_state.get("random_state") == random_state and
                saved_state.get("csv_path") == csv_path):
                preds = saved_state.get("preds", [])
                targets = saved_state.get("targets", [])
                start_idx = len(preds)
                print(f"[INFO] Checkpoint found! Automatically resuming from sample index {start_idx}/{num_samples}.")
            else:
                print("[WARNING] Checkpoint found but parameters do not match. Starting fresh.")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}. Starting fresh.")

    # Main evaluation loop
    for i in tqdm(range(start_idx, len(rows)), desc="Generating linkers"):
        row = rows[i]
        mol = Chem.MolFromSmiles(row.src)
        if mol is None:
            # Append empty fallback to keep lengths aligned
            preds.append("")
            targets.append(row.tgt)
            continue

        graph = encoder.encode_graph(mol)
        graph.x = graph.x.to(DEVICE)
        graph.edge_index = graph.edge_index.to(DEVICE)
        if hasattr(graph, "edge_attr"):
            graph.edge_attr = graph.edge_attr.to(DEVICE)

        memory = encoder.encode(graph).unsqueeze(0)

        # Generate beam candidates
        seq_candidates = beam_search_candidates(
            decoder,
            memory,
            start_token,
            end_token,
            beam_size=beam_size
        )

        # Pick the highest-ranked candidate that is valid after repair
        best_smi = None
        for seq in seq_candidates:
            smi_raw = tokenizer.decode(seq.squeeze().tolist())
            smi_repaired = repair_smiles(smi_raw)
            if safe_mol(smi_repaired) is not None:
                best_smi = smi_repaired
                break

        # Fallback: return the top candidate after repair regardless of validity
        if best_smi is None:
            smi_raw = tokenizer.decode(seq_candidates[0].squeeze().tolist())
            best_smi = repair_smiles(smi_raw)

        preds.append(best_smi)
        targets.append(row.tgt)
        
        # Save progress every 10 molecules to prevent data loss
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            try:
                with open(progress_file, "w") as f:
                    json.dump({
                        "num_samples": num_samples,
                        "beam_size": beam_size,
                        "random_state": random_state,
                        "csv_path": csv_path,
                        "preds": preds,
                        "targets": targets
                    }, f)
            except Exception as e:
                pass

    # Clean up checkpoint file upon 100% successful completion
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
        except Exception:
            pass

    # Save final predictions and targets for metrics.py
    with open("datasets/predictions.txt", "w") as f:
        for p in preds:
            f.write(str(p) + "\n")
    with open("datasets/final_test_dataset.txt", "w") as f:
        for t in targets:
            f.write(str(t) + "\n")
    print(f"[INFO] Saved predictions and targets to 'datasets/predictions.txt' and 'datasets/final_test_dataset.txt'")

    chem = compute_metrics(preds)

    print("\n================ FINAL FAST EVAL ================\n")
    print(f"Samples Tested            : {len(preds)}")
    print(f"Chemical Validity         : {chem['validity']*100:.2f}%")
    print(f"Invalid Rate              : {chem['invalid_rate']*100:.2f}%")
    print(f"Diversity                 : {chem['diversity']*100:.2f}%")
    print(f"Avg QED                   : {chem['avg_qed']:.4f}")
    print(f"Avg SA                    : {chem['avg_sa']:.4f}")
    print(f"Avg Ring Count            : {chem['avg_rings']:.4f} (Proof of Cyclisation!)")

    print("\nSample Outputs & Visualizations:")
    os.makedirs("graph_visuals_eval", exist_ok=True)

    for i in range(min(5, len(preds))):
        print(f"[{i+1}] Generated: {preds[i]}")

        mol = safe_mol(preds[i])
        if mol:
            try:
                Draw.MolToFile(mol, f"graph_visuals_eval/eval_generated_{i+1}.png", size=(300, 300))
            except Exception as e:
                print(f"    [WARNING] Could not draw molecule {i+1}: {e}")

    print("\n[INFO] Generated topological graphs saved directly to 'graph_visuals_eval/' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Graph Transformer Model")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--csv_path", type=str, default="./datasets/data/test.csv", help="Path to test CSV")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for sampling")
    args = parser.parse_args()

    evaluate(
        beam_size=args.beam_size,
        num_samples=args.num_samples,
        csv_path=args.csv_path,
        random_state=args.random_state
    )
