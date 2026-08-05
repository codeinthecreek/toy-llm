"""Checkpoint loading and generation for the dashboard's inference/playground view.

Deliberately stateless: each request reloads the checkpoint from disk rather than caching a
loaded model, since this is a toy-scale model and simplicity matters more than latency here.
"""

import json
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT
from train import tensor_stats


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


def architecture_info(run_dir, step=None):
    """Model config, parameter shapes, and per-parameter weight stats, for the architecture
    view's fetch-once-per-run-selection load. Computed straight from a checkpoint's own
    parameters, not the training log, so shapes are available even though the log only ever
    carries summary stats (see DESIGN.md "Instrumentation cadence")."""
    checkpoint_path = find_checkpoint(run_dir, step)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    parameters = []
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        parameters.append(
            {
                "name": name,
                "shape": list(param.shape),
                "num_params": num_params,
                "stats": tensor_stats(param.data),
            }
        )

    return {
        "step": checkpoint["step"],
        "config": checkpoint["model_config"],
        "parameters": parameters,
        "total_params": total_params,
    }


def attention_for_text(run_dir, text, step=None):
    """Attention weights from one forward pass over `text`, for the inference/playground
    view's post-generation heatmap. Mirrors train.py's attention_sample_event (same
    return_intermediates=True forward pass) but over arbitrary text instead of a fixed
    validation input, and is deliberately kept off the plain-text /generate stream — see
    DESIGN.md "Inference/playground view is a separate, non-streaming-log concern"."""
    model, stoi, itos, device = load_checkpoint_for_inference(run_dir, step)
    idx = encode_prompt(text, stoi, device)
    idx = idx[:, -model.config.block_size :]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, block_intermediates = model(idx, return_intermediates=True)
    model.train(was_training)

    tokens = [itos[i] for i in idx[0].tolist()]
    layers = [
        {"layer": i, "attn_weights": layer["attn_weights"][0].tolist()}
        for i, layer in enumerate(block_intermediates)
    ]
    return {"tokens": tokens, "layers": layers}
