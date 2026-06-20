import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmilesDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim=192,
        encoder_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_len=256
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.memory_proj = nn.Linear(encoder_dim, hidden_dim)

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=0
        )

        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device),
            diagonal=1
        )

    def forward(self, encoder_memory, tgt_tokens, memory_key_padding_mask=None):

        memory = self.memory_proj(encoder_memory)

        x = self.token_embedding(tgt_tokens)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt_tokens.size(1),
            tgt_tokens.device
        )

        tgt_padding_mask = (tgt_tokens == 0)

        out = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        out = self.norm(out)
        return self.output_layer(out)

    # ================= FIXED GENERATION =================
    @torch.no_grad()
    def generate(
        self,
        encoder_memory,
        start_token,
        end_token,
        max_len=256,
        temperature=0.9,
        top_k=5,
        memory_key_padding_mask=None
    ):
        self.eval()

        generated = torch.tensor(
            [[start_token]],
            device=encoder_memory.device,
            dtype=torch.long
        )

        for _ in range(max_len):

            logits = self.forward(encoder_memory, generated, memory_key_padding_mask=memory_key_padding_mask)

            logits = logits[:, -1, :] / temperature

            # top-k filtering
            if top_k is not None:
                vals, idxs = torch.topk(logits, top_k)
                probs = torch.softmax(vals, dim=-1)
                choice = torch.multinomial(probs, 1)
                next_token = idxs.gather(-1, choice)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == end_token:
                break

        return generated