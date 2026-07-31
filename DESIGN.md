# Design decisions

This document captures *why* the project is built the way it is. `CLAUDE.md`
stays operational (conventions, commands, constraints); this is where the
reasoning behind those constraints lives, for future sessions and future me.

## Purpose and scope

The goal is to visualize how an LLM works — architecture, training, and
inference — not to produce a usable model. "Toy" is a design constraint, not
a limitation: the model must stay small enough that its internals are
legible in a visualization, even at the cost of capability.

Three areas get roughly equal weight: architecture (data flow through the
model), training dynamics (how it learns over time), and inference (token-
by-token generation behavior).

## Model implementation

**Hand-written components, PyTorch for autograd only.** No
`nn.MultiheadAttention` or `nn.TransformerEncoder` — every component
(attention, MLP, LayerNorm wiring) is written explicitly so the forward pass
is inspectable end to end and can optionally return intermediate tensors
(attention weights per head, pre/post-LayerNorm activations) for
instrumentation. This was chosen over a pure-NumPy/manual-backprop
implementation: the priority is conceptual clarity of *structure*, not
re-deriving backpropagation from scratch.

**Weight tying** (`lm_head.weight` aliased to `token_emb.weight`) and
**GPT-2-style scaled initialization** (std=0.02, residual-branch output
projections scaled by `1/sqrt(2*n_layer)`) were added after the initial
model core, matching standard GPT implementations. Both are cosmetic at this
model's scale (~32K params) but keep the implementation representative of
how real GPT-style models are built.

**Model size stays tiny by default** (config-driven `n_layer`, `n_head`,
`n_embd`, `block_size`) — legibility of the resulting visualizations matters
more than capability.

## Data and tokenization

**tiny-shakespeare, character-level tokenization.** Vocab is just the
unique characters in the corpus — no tokenizer library needed, keeps the
vocab small enough that embedding-space visualizations (PCA projections,
per-token attention) stay legible rather than becoming an unreadable wall of
subword tokens.

## Process architecture: training and dashboard are separate processes

Two options were considered: run training and the dashboard backend in the
same process (simpler, but the dashboard freezes during training unless
threaded/async), or as separate processes streaming to each other. Separate
processes were chosen so the dashboard stays responsive during training,
and so training doesn't depend on the dashboard being up at all.

**Training writes; it does not push.** `train.py` appends JSON lines to
`runs/<run_id>/snapshots.jsonl` rather than streaming directly over a socket
to the dashboard backend. This was chosen over direct socket streaming
because it decouples the two processes completely: training never blocks or
errors if the dashboard isn't running, and a finished run's history can be
replayed by the dashboard at any time by reading the log file, not just
watched live. The dashboard backend is expected to tail this file for live
updates.

## Instrumentation cadence

Different snapshot types have different costs, so they're logged at
different frequencies:

- **Loss**: every step. Cheap (a scalar).
- **Weight/gradient histograms and a fixed-input attention sample**: every N
  steps (default 50). These require moving tensors off the GPU
  (`.detach().cpu()`) to serialize, which has real overhead — logged as
  summary statistics (mean/std/min/max/percentiles), not full tensors, to
  keep both the compute cost and the log file size down.
- **Embedding PCA projection and checkpoints**: every M steps (default 500,
  M >> N). The most expensive to compute, and the slowest-changing
  signal — the embedding space doesn't reorganize meaningfully step to step.

Each JSON line has a `type` field (`loss`, `histogram`, `attention_sample`,
`embedding_projection`) and a `step` field, so the dashboard can filter and
render each stream independently.

## Environment

**Dedicated pyenv virtualenv (`toy-llm`, Python 3.12.12), not the existing
inference venv.** The existing venv (used for ollama, chromadb,
sentence-transformers, CPU torch) would have pulled in unrelated
dependencies and locked this project to a CPU-only torch build. A fresh venv
keeps `pip freeze` meaningful and keeps the CUDA torch build isolated from
the CPU build used elsewhere.

**`requirements.txt`, not `pyproject.toml`.** The project will never be
packaged or distributed, so `pyproject.toml`'s build metadata and packaging
config buys nothing. `requirements.txt` also handles the CUDA wheel index
directive natively (`--extra-index-url`), whereas `pyproject.toml` has no
pip-native equivalent (only `uv`-specific extensions).

**CUDA 13.0 wheel via `--extra-index-url`, not `--index-url`.** The local
driver reports CUDA 13.3 support; PyTorch's newest published wheel target is
`cu130`. Plain `--index-url` hits a known packaging bug where a required
dependency (`cuda-bindings==13.0.3`) is missing from the `cu130` index;
`--extra-index-url` works around it by letting pip fall back to PyPI for
that dependency.

**GPU is not a constraint for this project.** Local hardware is a GTX 1650
(TU117, 4GB GDDR5, no tensor cores) — limiting for the real local-inference
work this machine is also used for, but a nanoGPT-scale toy model is a
rounding error against 4GB VRAM. Device is a config flag (`cuda` if
available, else `cpu`) rather than hardcoded, mainly so CPU-vs-GPU training
time can be compared directly as a side experiment.

## Development environment: native, not Docker

Claude Code runs natively on the host for this project, not inside the
Docker sandbox used for systems-level work. The two are different risk
profiles: systems work has a wide blast radius (OS config, package
management, mounted drives) and benefits from containment; project dev work
is scoped to a version-controlled directory tree, where the worst case is
recoverable via git. Containerizing this project would also reintroduce GPU
passthrough friction (`nvidia-container-toolkit`, driver/CUDA version
matching inside the container) for no real safety benefit at this risk
level.

**Guardrails instead of containment**: `.claude/settings.local.json` denies
destructive commands outright (`sudo`, `dd`, `mkfs`, `rm -rf /*`, force-push,
curl-pipe-bash) and requires confirmation for risky-but-sometimes-legitimate
ones (`rm -rf` scoped, `git reset --hard`, `git clean`). This is a policy
enforced by Claude Code itself, not OS-level sandboxing — git discipline
(commit before letting Claude Code make non-trivial changes) is the actual
fallback if something gets through.

## Commit conventions

Commits with substantial Claude Code-authored content carry a
`Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer, adopted
from the training-loop commit onward. Not applied retroactively to earlier
commits — not worth rewriting history for a cosmetic trailer.

## Open questions / not yet decided

- Dashboard frontend framework/library — not yet chosen.
- Exact PCA vs. t-SNE tradeoff for the embedding-projection view (PCA is
  cheap enough to compute at every M-step snapshot; t-SNE was discussed as
  better for periodic, less-frequent snapshots given its cost, but this
  hasn't been implemented yet).
- Whether the dashboard needs authentication/access control if ever exposed
  beyond localhost on the homelab network.
