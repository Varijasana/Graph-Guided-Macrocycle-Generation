
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

GRAPH_DIR = "graph_dataset"
os.makedirs(GRAPH_DIR, exist_ok=True)

_graph_counter = 0


class GraphDataset:
    def __init__(self, csv_file, mode="encoder"):
        """
        mode:
        - encoder -> current training (UNCHANGED BEHAVIOR)
        - decoder -> future SMILES generation training
        """
        self.df = pd.read_csv(csv_file)
        self.mode = mode

        print("Loaded columns:", self.df.columns)
        print("Dataset size:", len(self.df))
        print("Dataset mode:", self.mode)

    def __len__(self):
        return len(self.df)

    def smiles_to_graph(self, smiles, save_graph=False, tag="train"):
        global _graph_counter

        try:
            if not isinstance(smiles, str) or smiles.strip() == "":
                return None

            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return None

            # -------- NODE FEATURES --------
            x = []
            for atom in mol.GetAtoms():
                h_count = atom.GetTotalNumHs() if atom.GetAtomicNum() > 0 else 0

                x.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    h_count
                ])

            # -------- EDGE FEATURES --------
            edge_index = []
            edge_attr = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_index.append([i, j])
                edge_index.append([j, i])

                bt = bond.GetBondType()

                bf = [
                    int(bt == Chem.rdchem.BondType.SINGLE),
                    int(bt == Chem.rdchem.BondType.DOUBLE),
                    int(bt == Chem.rdchem.BondType.TRIPLE),
                    int(bt == Chem.rdchem.BondType.AROMATIC)
                ]

                edge_attr.append(bf)
                edge_attr.append(bf)

            if len(edge_index) == 0:
                return None

            x = torch.tensor(x, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            if save_graph:
                save_path = os.path.join(
                    GRAPH_DIR,
                    f"graph_{_graph_counter}_{tag}.pt"
                )

                torch.save(graph, save_path)
                print(f"[GRAPH SAVED] {save_path}")

                _graph_counter += 1

            return graph

        except Exception:
            return None

    def __getitem__(self, idx):
        max_tries = 10

        for i in range(max_tries):
            new_idx = (idx + i) % len(self.df)
            row = self.df.iloc[new_idx]

            smiles = row["src"]

            graph = self.smiles_to_graph(smiles, save_graph=True)

            if graph is None:
                continue

            # =========================
            # NEW: decoder target
            # =========================
            tgt = None
            if "tgt" in self.df.columns:
                tgt = row["tgt"]

            # =========================
            # MODE HANDLING
            # =========================

            if self.mode == "decoder":
                return {
                    "graph": graph,
                    "tgt": tgt
                }

            # encoder mode (UNCHANGED)
            return graph

        # fallback (encoder-safe)
        x = torch.zeros((1, 6), dtype=torch.float)
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr = torch.zeros((1, 4), dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if self.mode == "decoder":
            return {"graph": graph, "tgt": ""}

        return graph