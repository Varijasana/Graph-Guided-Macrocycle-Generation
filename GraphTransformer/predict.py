import argparse
import torch
import os
from rdkit import Chem
from rdkit.Chem import Draw

from model import GraphMacTransformer
from decoder import SmilesDecoder
from tokenizer import SmilesTokenizer
from evaluate_model import beam_search, safe_mol, repair_smiles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_predicted_smiles(smiles):
    return repair_smiles(smiles)

def predict_cyclisation(smiles, output_file=None, beam_width=5):
    print(f"\n[INFO] Input Acyclic SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("[ERROR] Invalid SMILES inputted. Please check your chemistry strings.")
        return

    # Load architecture
    tokenizer = SmilesTokenizer()
    encoder = GraphMacTransformer().to(DEVICE)
    decoder = SmilesDecoder(
        vocab_size=tokenizer.vocab_size(),
        hidden_dim=192,
        encoder_dim=128,
        num_layers=4
    ).to(DEVICE)

    # Load Joint Checkpoint using absolute path
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "joint_checkpoint.pth")
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print("[OK] Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"[WARNING] No trained checkpoint found at {checkpoint_path}. Running with random weights...")

    encoder.eval()
    decoder.eval()

    # Encode acyclic graph
    graph = encoder.encode_graph(mol)
    graph.x = graph.x.to(DEVICE)
    graph.edge_index = graph.edge_index.to(DEVICE)
    if hasattr(graph, "edge_attr"):
        graph.edge_attr = graph.edge_attr.to(DEVICE)

    with torch.no_grad():
        memory = encoder.encode(graph).unsqueeze(0)

        start_token = tokenizer.stoi.get("<SOS>", tokenizer.stoi.get("<START>", 1))
        end_token = tokenizer.stoi.get("<EOS>", tokenizer.stoi.get("<END>", 2))

        # Autoregressive generation
        seq = beam_search(
            decoder=decoder,
            memory=memory,
            start_token=start_token,
            end_token=end_token,
            beam_size=beam_width,
            max_len=256
        )

        pred_smiles = tokenizer.decode(seq.squeeze().tolist())
    
    # Post-process UNK tokens to prevent RDKit parse failures
    pred_smiles = clean_predicted_smiles(pred_smiles)
    
    print(f"[OK] Predicted Cyclized Output: {pred_smiles}\n")
    
    # Save graph visuals
    os.makedirs("graph_visuals", exist_ok=True)
    Draw.MolToFile(mol, f"graph_visuals/input_acyclic.png", size=(500, 500))
    
    pred_mol = safe_mol(pred_smiles)
    if pred_mol:
        Draw.MolToFile(pred_mol, f"graph_visuals/predicted_cyclized.png", size=(500, 500))
        print("[SAVED] Saved graph visuals directly: 'graph_visuals/predicted_cyclized.png'")
    else:
        print("[WARNING] Predicted SMILES has syntax errors, could not draw predicted structure.")

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(pred_smiles + "\n")

    return pred_smiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cyclisation Rings from Acyclic Molecules")
    parser.add_argument("--smiles", type=str, required=True, help="Your acyclic input SMILES string")
    parser.add_argument("--output_file", type=str, default=None, help="Path to write the predicted SMILES to")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for search")
    args = parser.parse_args()

    predict_cyclisation(args.smiles, args.output_file, args.beam_width)
