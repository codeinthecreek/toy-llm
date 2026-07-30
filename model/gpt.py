import torch
import torch.nn as nn

from model.block import Block


class GPT(nn.Module):
    """A small GPT-style decoder-only transformer.

    Every component (embeddings, attention, MLP, LayerNorm wiring) is wired
    up explicitly rather than via nn.TransformerEncoder, so the forward pass
    stays inspectable end to end.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, return_intermediates=False):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)

        block_intermediates = [] if return_intermediates else None
        for block in self.blocks:
            x, intermediates = block(x, return_intermediates=return_intermediates)
            if return_intermediates:
                block_intermediates.append(intermediates)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if not return_intermediates:
            return logits, None

        return logits, block_intermediates
