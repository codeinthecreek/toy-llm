# toy-llm

A small hand-written transformer built for learning — the goal is to visualize
architecture, training, and inference, not to produce a usable model.

## Environment
- Python 3.12.12 via pyenv virtualenv `toy-llm` (already active before Claude Code runs) — do not create a new venv or use system pip.
- torch: CUDA 13.0 build, installed with `--extra-index-url` (not `--index-url`, due to a known cu130 packaging bug).
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## Design constraints
- No `nn.MultiheadAttention` or `nn.TransformerEncoder` — every component (attention, MLP, LayerNorm wiring) written explicitly so internals are inspectable.
- Model must stay small (config-driven n_layer/n_head/n_embd) — legibility of visualizations matters more than capability.
- Forward pass should optionally return intermediate tensors (attention weights, activations) for later instrumentation.

## Structure
- `model/` — transformer implementation
- `data/` — tokenizer + dataset prep
- `train.py` — training loop (separate process from dashboard)
- `dashboard/` — FastAPI backend + frontend, consumes state streamed from train.py
- `scripts/` — standalone inference/generation
