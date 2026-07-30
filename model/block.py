import torch.nn as nn

from model.attention import CausalSelfAttention


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU, written out explicitly."""

    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm transformer block: LN -> attention -> residual, LN -> MLP -> residual."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_intermediates=False):
        pre_ln_1 = x
        post_ln_1 = self.ln_1(pre_ln_1)
        attn_out, attn_weights = self.attn(post_ln_1, return_attn_weights=return_intermediates)
        x = pre_ln_1 + attn_out

        pre_ln_2 = x
        post_ln_2 = self.ln_2(pre_ln_2)
        mlp_out = self.mlp(post_ln_2)
        x = pre_ln_2 + mlp_out

        if not return_intermediates:
            return x, None

        intermediates = {
            "pre_ln_1": pre_ln_1.detach(),
            "post_ln_1": post_ln_1.detach(),
            "attn_weights": attn_weights.detach() if attn_weights is not None else None,
            "pre_ln_2": pre_ln_2.detach(),
            "post_ln_2": post_ln_2.detach(),
            "block_output": x.detach(),
        }
        return x, intermediates
