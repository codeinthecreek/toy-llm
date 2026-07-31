import math

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

        self.apply(self._init_weights)
        self._scale_residual_projections()

        # weight tying: the output head reuses the token embedding matrix
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        """GPT-2-style init: N(0, 0.02) for Linear/Embedding weights, zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        """Scale down each residual branch's final projection by 1/sqrt(2*n_layer),
        the standard GPT-2 trick to keep residual-stream variance from growing with depth."""
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        matched = 0
        for name, param in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.fc_out.weight"):
                nn.init.normal_(param, mean=0.0, std=std)
                matched += 1

        expected = 2 * self.config.n_layer
        assert matched == expected, (
            f"expected to scale {expected} residual-branch projections "
            f"(2 per layer x {self.config.n_layer} layers), matched {matched} — "
            "check for a layer/attribute rename"
        )

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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample `max_new_tokens` tokens following `idx` (B, T)."""
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        self.train(was_training)
        return idx
