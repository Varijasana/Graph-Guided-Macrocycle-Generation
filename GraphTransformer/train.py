import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader

from model import GraphMacTransformer
from decoder import SmilesDecoder
from tokenizer import SmilesTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = "./checkpoints"
ENC_PATH = f"{CKPT_DIR}/latest_checkpoint.pth"
DEC_PATH = f"{CKPT_DIR}/best_decoder_model.pth"
JOINT_PATH = f"{CKPT_DIR}/joint_checkpoint.pth"


# =========================================================
# DATASET
# =========================================================
class SmilesDataset(Dataset):

    def __init__(self, df, encoder, tokenizer):
        self.data = df.reset_index(drop=True)
        self.encoder = encoder
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        mol = Chem.MolFromSmiles(row["src"])
        if mol is None:
            return None

        graph = self.encoder.encode_graph(mol)

        tgt = self.tokenizer.encode(row["tgt"], max_len=256)
        tgt = torch.tensor(tgt, dtype=torch.long)

        return graph, tgt


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    graphs, tgts = zip(*batch)
    return list(graphs), torch.stack(tgts)


# =========================================================
# LOAD MODELS
# =========================================================
def load_models():

    tok = SmilesTokenizer()

    enc = GraphMacTransformer().to(DEVICE)

    dec = SmilesDecoder(
        vocab_size=tok.vocab_size(),
        hidden_dim=192,
        encoder_dim=128,
        num_layers=4
    ).to(DEVICE)

    return enc, dec, tok


# =========================================================
# CHECKPOINT
# =========================================================
def save_ckpt(epoch, enc, dec, opt, best):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "optimizer": opt.state_dict(),
        "best": best
    }, JOINT_PATH)


def load_ckpt(enc, dec, opt):
    if not os.path.exists(JOINT_PATH):
        print("\n[INFO] Starting fresh end-to-end joint training!")
        return 0, 999

    ckpt = torch.load(JOINT_PATH, map_location=DEVICE)

    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    opt.load_state_dict(ckpt["optimizer"])

    print(f"\n[INFO] Resumed safely from epoch {ckpt['epoch']+1}\n")

    return ckpt["epoch"] + 1, ckpt["best"]


# =========================================================
# TRAIN STEP (IMPROVED JOINT LEARNING)
# =========================================================
def train_one(enc, dec, opt, loss_fn, graphs, tgts, tok, ss_prob=0.2):

    tgts = tgts.to(DEVICE)
    opt.zero_grad()

    memory_list = []

    for g in graphs:

        g.x = g.x.to(DEVICE)
        g.edge_index = g.edge_index.to(DEVICE)

        if hasattr(g, "edge_attr"):
            g.edge_attr = g.edge_attr.to(DEVICE)

        # =====================================================
        # encoder forward (joint learning)
        # =====================================================
        node_emb = enc.encode(g)

        # Light regularization (helps generalization)
        node_emb = F.dropout(node_emb, p=0.1, training=enc.training)

        memory_list.append(node_emb)

    memory = torch.nn.utils.rnn.pad_sequence(memory_list, batch_first=True)
    
    max_len = memory.size(1)
    memory_key_padding_mask = torch.zeros(len(memory_list), max_len, dtype=torch.bool, device=DEVICE)
    for i, mem_item in enumerate(memory_list):
        memory_key_padding_mask[i, mem_item.size(0):] = True

    inp = tgts[:, :-1]
    target = tgts[:, 1:]

    # teacher forcing noise (kept same)
    if random.random() < ss_prob:
        mask = torch.rand_like(inp.float()) < 0.1
        rand_tokens = torch.randint(0, tok.vocab_size(), inp.shape).to(DEVICE)
        inp = torch.where(mask, rand_tokens, inp)

    logits = dec(memory, inp, memory_key_padding_mask=memory_key_padding_mask)

    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1)
    )

    loss.backward()

    # stabilize training
    torch.nn.utils.clip_grad_norm_(
        list(enc.parameters()) + list(dec.parameters()),
        1.0
    )

    opt.step()

    return loss.item()


# =========================================================
# TRAIN LOOP
# =========================================================
def train():

    enc, dec, tok = load_models()

    opt = optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tok.stoi["<PAD>"],
        label_smoothing=0.1
    )

    start_epoch, best = load_ckpt(enc, dec, opt)

    df = pd.read_csv("./datasets/data/train.csv")

    dataset = SmilesDataset(df, enc, tok)

    loader = DataLoader(
        dataset,
        batch_size=32,  # Speed up training
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    EPOCHS = 35 # Train a very strong professional model

    print("\n[INFO] STRONG JOINT MULTI-EPOCH TRAINING STARTED\n")

    for epoch in range(start_epoch, EPOCHS):

        enc.train()
        dec.train()

        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:

            if batch is None:
                continue

            graphs, tgts = batch

            loss = train_one(enc, dec, opt, loss_fn, graphs, tgts, tok)

            total_loss += loss
            loop.set_postfix(loss=loss)

        avg_loss = total_loss / len(loader)

        print(f"\n[INFO] Epoch {epoch+1} Completed!")
        print(f"[OK] Avg Train Loss: {avg_loss:.4f}\n")

        save_ckpt(epoch, enc, dec, opt, best)


if __name__ == "__main__":
    train()