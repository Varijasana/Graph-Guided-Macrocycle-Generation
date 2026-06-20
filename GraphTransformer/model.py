
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem

# =========================================================
# GRAPH BUILDER (UNCHANGED)
# =========================================================
class GraphBuilder:
    def atom_features(self, atom):
        return np.array([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs()
        ], dtype=np.float32)

    def bond_features(self, bond):
        bt = bond.GetBondType()
        return np.array([
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC)
        ], dtype=np.float32)

# =========================================================
# ATTENTION LAYER
# =========================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_encoder = nn.Linear(4, out_dim)
        self.attn = nn.Linear(3 * out_dim, 1)

    def forward(self, x, edge_index, edge_attr):

        h = self.linear(x)

        row, col = edge_index
        h_row, h_col = h[row], h[col]

        e = self.edge_encoder(edge_attr)

        attn_input = torch.cat([h_row, h_col, e], dim=-1)
        attn_scores = self.attn(attn_input)
        attn_scores = F.leaky_relu(attn_scores, 0.2)

        # Per node softmax (Scatter Softmax)
        exp_attn = torch.exp(attn_scores - attn_scores.max()) # prevent overflow
        
        sum_exp = torch.zeros(h.size(0), 1, dtype=exp_attn.dtype, device=h.device)
        sum_exp.index_add_(0, row, exp_attn)
        sum_exp_gathered = sum_exp[row]
        
        attn_weights = exp_attn / (sum_exp_gathered + 1e-9)

        out = torch.zeros_like(h)
        out.index_add_(0, row, attn_weights * h_col)

        return out

# =========================================================
# TRANSFORMER BLOCK
# =========================================================
class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = GraphAttentionLayer(hidden_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.norm1(x + self.attn(x, edge_index, edge_attr))
        x = self.norm2(x + self.ff(x))
        return x

# =========================================================
# MAIN ENCODER (UNCHANGED BEHAVIOR)
# =========================================================
class GraphMacTransformer(nn.Module):
    def __init__(self, node_dim=6, hidden_dim=128, num_layers=4, out_dim=6, use_decoder=False):

        super().__init__()

        self.use_decoder = use_decoder

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Linear(hidden_dim, out_dim)

    # =====================================================
    # FORWARD (UNCHANGED LOGIC)
    # =====================================================
    def forward(self, *args):

        if len(args) == 1:
            data = args[0]
            x, edge_index = data.x, data.edge_index
            edge_attr = getattr(data, "edge_attr", None)

        elif len(args) == 2:
            x, edge_index = args
            edge_attr = None
        else:
            raise ValueError("Invalid forward call")

        x = self.node_encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return self.decoder(x)

    # =====================================================
    # NEW: CLEAN ENCODER OUTPUT (FOR DECODER)
    # =====================================================
    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        x = self.node_encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x

    # =====================================================
    # GRAPH ENCODING (UNCHANGED)
    # =====================================================
    def encode_graph(self, mol):

        x = []
        edge_index = []
        edge_attr = []

        for atom in mol.GetAtoms():
            x.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs()
            ])

        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feat = [
                int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC)
            ]

            edge_index.append([i, j])
            edge_index.append([j, i])

            edge_attr.append(feat)
            edge_attr.append(feat)

        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        class G:
            def __init__(self):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr

        return G()