import json
from pathlib import Path

import pytest
import torch

from model.config import GPTConfig
from train import TrainConfig, tensor_stats, train


def make_synthetic_data(pattern="abc", repeats=400):
    """A trivially learnable periodic sequence — next-char prediction should converge fast."""
    text = pattern * repeats
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    vocab_meta = {"vocab_size": len(chars), "itos": itos}
    return train_data, val_data, vocab_meta


def test_training_loop_writes_snapshots_and_loss_decreases(tmp_path):
    torch.manual_seed(0)
    train_data, val_data, vocab_meta = make_synthetic_data()

    model_config = GPTConfig(
        vocab_size=vocab_meta["vocab_size"], block_size=16, n_layer=2, n_head=2, n_embd=16
    )
    train_config = TrainConfig(
        out_dir=str(tmp_path / "runs"),
        learning_rate=1e-2,
        batch_size=16,
        max_steps=200,
        log_every=20,
        checkpoint_every=50,
        device="cpu",
        seed=0,
    )

    run_dir, losses = train(train_data, val_data, model_config, train_config, vocab_meta=vocab_meta)

    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg

    snapshot_path = Path(run_dir) / "snapshots.jsonl"
    assert snapshot_path.exists()

    events = []
    with open(snapshot_path) as f:
        for line in f:
            line = line.strip()
            assert line, "snapshot file should not contain blank lines"
            events.append(json.loads(line))  # raises if any line isn't valid JSON

    event_types = {e["type"] for e in events}
    assert event_types == {"loss", "histogram", "attention_sample", "embedding_projection"}

    loss_events = [e for e in events if e["type"] == "loss"]
    assert len(loss_events) == train_config.max_steps
    assert [e["step"] for e in loss_events] == list(range(train_config.max_steps))

    histogram_events = [e for e in events if e["type"] == "histogram"]
    assert len(histogram_events) == train_config.max_steps // train_config.log_every
    sample_histogram = histogram_events[0]
    assert "token_emb.weight" in sample_histogram["weights"]
    assert "token_emb.weight" in sample_histogram["grads"]
    for stats in sample_histogram["weights"].values():
        assert set(stats.keys()) == {"mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"}

    attention_events = [e for e in events if e["type"] == "attention_sample"]
    assert len(attention_events) == len(histogram_events)
    first_layer = attention_events[0]["layers"][0]
    attn = first_layer["attn_weights"]
    assert len(attn) == model_config.n_head
    assert len(attn[0]) == model_config.block_size

    projection_events = [e for e in events if e["type"] == "embedding_projection"]
    assert len(projection_events) == train_config.max_steps // train_config.checkpoint_every
    assert len(projection_events[0]["projection"]) == vocab_meta["vocab_size"]
    assert len(projection_events[0]["projection"][0]) == 2
    assert projection_events[0]["tokens"] == [vocab_meta["itos"][i] for i in range(vocab_meta["vocab_size"])]

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) == train_config.max_steps // train_config.checkpoint_every


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_tensor_stats_on_cuda():
    # torch.quantile requires its q tensor to live on the same device as the input —
    # regression check for a device-mismatch bug only reproducible on GPU.
    t = torch.randn(100, device="cuda")
    stats = tensor_stats(t)
    assert set(stats.keys()) == {"mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"}
