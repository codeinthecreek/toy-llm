import torch

from model.config import GPTConfig
from model.gpt import GPT


def make_model():
    config = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=8)
    return config, GPT(config)


def test_forward_output_shape():
    config, model = make_model()
    batch_size, seq_len = 4, 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, intermediates = model(idx)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert intermediates is None


def test_forward_with_intermediates():
    config, model = make_model()
    batch_size, seq_len = 2, 6
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, block_intermediates = model(idx, return_intermediates=True)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert len(block_intermediates) == config.n_layer

    head_dim = config.n_embd // config.n_head
    for layer in block_intermediates:
        assert layer["pre_ln_1"].shape == (batch_size, seq_len, config.n_embd)
        assert layer["post_ln_1"].shape == (batch_size, seq_len, config.n_embd)
        assert layer["pre_ln_2"].shape == (batch_size, seq_len, config.n_embd)
        assert layer["post_ln_2"].shape == (batch_size, seq_len, config.n_embd)
        assert layer["block_output"].shape == (batch_size, seq_len, config.n_embd)
        attn_weights = layer["attn_weights"]
        assert attn_weights.shape == (batch_size, config.n_head, seq_len, seq_len)
        assert head_dim * config.n_head == config.n_embd

        # causal: each row sums to 1, and no weight leaks to future positions
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, config.n_head, seq_len))
        upper_triangle = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        assert torch.all(attn_weights[:, :, upper_triangle] == 0)
