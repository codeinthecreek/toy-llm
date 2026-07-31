"""Training loop for the toy GPT model, with instrumentation streamed to
runs/<run_id>/snapshots.jsonl as JSON lines for later dashboard consumption.

Kept as a separate process from the dashboard: this script only writes files,
it doesn't know anything about how they get displayed.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from data.prepare import fetch_tiny_shakespeare, prepare_dataset
from model.config import GPTConfig
from model.gpt import GPT


@dataclass
class TrainConfig:
    data_dir: str = "data/shakespeare"
    out_dir: str = "runs"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    max_steps: int = 2000
    log_every: int = 50  # N: weight/grad histograms + attention sample
    checkpoint_every: int = 500  # M: checkpoint + embedding PCA projection (M >> N)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


def make_run_id():
    return time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"


class SnapshotLogger:
    """Appends one JSON object per line to runs/<run_id>/snapshots.jsonl."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")

    def log(self, event):
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def get_batch(data, block_size, batch_size, device):
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def tensor_stats(t):
    """Summary stats for a tensor — mean/std/min/max/percentiles — not the full tensor."""
    flat = t.detach().float().flatten()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=flat.device)
    quantiles = torch.quantile(flat, q)
    return {
        "mean": flat.mean().item(),
        "std": flat.std().item() if flat.numel() > 1 else 0.0,
        "min": flat.min().item(),
        "max": flat.max().item(),
        "p10": quantiles[0].item(),
        "p25": quantiles[1].item(),
        "p50": quantiles[2].item(),
        "p75": quantiles[3].item(),
        "p90": quantiles[4].item(),
    }


def histogram_event(step, model):
    weights, grads = {}, {}
    for name, param in model.named_parameters():
        weights[name] = tensor_stats(param.data)
        if param.grad is not None:
            grads[name] = tensor_stats(param.grad)
    return {"type": "histogram", "step": step, "weights": weights, "grads": grads}


def attention_sample_event(step, model, fixed_input):
    """Attention weights for a single fixed validation input, so patterns are comparable across steps."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, block_intermediates = model(fixed_input, return_intermediates=True)
    model.train(was_training)

    layers = [
        {"layer": i, "attn_weights": layer["attn_weights"][0].tolist()}
        for i, layer in enumerate(block_intermediates)
    ]
    return {"type": "attention_sample", "step": step, "layers": layers}


def embedding_projection_event(step, model, vocab_meta=None):
    weight = model.token_emb.weight.detach().float().cpu()
    centered = weight - weight.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    projection = (centered @ v[:, :2]).tolist()

    event = {"type": "embedding_projection", "step": step, "projection": projection}
    if vocab_meta is not None and "itos" in vocab_meta:
        itos = vocab_meta["itos"]
        event["tokens"] = [itos[str(i)] if str(i) in itos else itos[i] for i in range(len(projection))]
    return event


def save_checkpoint(checkpoint_dir, step, model, optimizer, model_config, train_config):
    path = Path(checkpoint_dir) / f"step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        path,
    )
    return path


def train(train_data, val_data, model_config, train_config, vocab_meta=None):
    """Run the training loop, writing instrumentation to runs/<run_id>/snapshots.jsonl.

    Returns (run_dir, losses) — losses is the per-step loss history, useful for tests.
    """
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    run_dir = Path(train_config.out_dir) / make_run_id()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = SnapshotLogger(run_dir / "snapshots.jsonl")

    # fixed input so the attention pattern is comparable across logging steps
    fixed_val_x = val_data[: model_config.block_size].unsqueeze(0).to(device)

    losses = []
    try:
        for step in range(train_config.max_steps):
            x, y = get_batch(train_data, model_config.block_size, train_config.batch_size, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if step % train_config.log_every == 0:
                logger.log(histogram_event(step, model))
                logger.log(attention_sample_event(step, model, fixed_val_x))

            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            logger.log({"type": "loss", "step": step, "value": loss_value})

            if step % train_config.checkpoint_every == 0:
                save_checkpoint(checkpoint_dir, step, model, optimizer, model_config, train_config)
                logger.log(embedding_projection_event(step, model, vocab_meta))
    finally:
        logger.close()

    return run_dir, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/shakespeare")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train.pt").exists():
        text = fetch_tiny_shakespeare()
        prepare_dataset(text, data_dir)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")
    with open(data_dir / "meta.json") as f:
        vocab_meta = json.load(f)

    model_config = GPTConfig(
        vocab_size=vocab_meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    train_config = TrainConfig(
        data_dir=str(data_dir),
        out_dir=args.out_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )

    run_dir, losses = train(train_data, val_data, model_config, train_config, vocab_meta=vocab_meta)
    print(f"run dir: {run_dir}")
    print(f"final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
