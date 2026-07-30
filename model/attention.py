import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit Q/K/V projections.

    Every step (projection, head split, scaled dot-product, masking, softmax,
    weighted sum, output projection) is written out rather than delegated to
    nn.MultiheadAttention, so the forward pass can be inspected head-by-head.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask, cached as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_attn_weights=False):
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention, computed explicitly
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_head, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        out = self.resid_dropout(self.out_proj(out))

        if return_attn_weights:
            return out, attn_weights
        return out, None
