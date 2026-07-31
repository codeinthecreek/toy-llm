"""Checkpoint loading and generation for the dashboard's inference/playground view.

Deliberately stateless: each request reloads the checkpoint from disk rather than caching a
loaded model, since this is a toy-scale model and simplicity matters more than latency here.
"""

import json
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT


def find_checkpoint(run_dir, step=None):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    if step is None:
        return checkpoints[-1]

    target = checkpoint_dir / f"step_{step:07d}.pt"
    if not target.exists():
        raise FileNotFoundError(f"no checkpoint for step {step} in {checkpoint_dir}")
    return target


def load_checkpoint_for_inference(run_dir, step=None, device=None):
    """Returns (model, stoi, itos, device) ready for generation."""
    checkpoint_path = find_checkpoint(run_dir, step)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_dir = Path(checkpoint["train_config"]["data_dir"])
    with open(data_dir / "meta.json") as f:
        vocab_meta = json.load(f)
    stoi = vocab_meta["stoi"]
    itos = {int(k): v for k, v in vocab_meta["itos"].items()}

    return model, stoi, itos, device


def encode_prompt(prompt, stoi, device):
    if not prompt:
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    unknown = sorted({ch for ch in prompt if ch not in stoi})
    if unknown:
        raise ValueError(f"prompt contains characters outside the training vocabulary: {unknown!r}")

    return torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)


def generate_stream(model, idx, itos, max_new_tokens=200, temperature=1.0, top_k=None):
    """Yields one generated character at a time as plain-text chunks."""
    for _ in range(max_new_tokens):
        idx = model.generate(idx, max_new_tokens=1, temperature=temperature, top_k=top_k)
        yield itos[idx[0, -1].item()]
