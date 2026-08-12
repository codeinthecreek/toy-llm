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

## Dashboard architecture

**FastAPI backend, polling-based file tail.** The backend reads
`runs/<run_id>/snapshots.jsonl` from the last known offset on a short
interval (starting around 200-500ms) rather than watching the filesystem
with something like `watchdog`/inotify. At this project's snapshot cadence,
polling is very likely indistinguishable from a filesystem watch to a human
eye, and it avoids an extra dependency and moving part. Revisit only if
polling is observed to feel laggy in practice.

**Server-Sent Events (SSE) for pushing updates to the browser, not
WebSocket.** The training and architecture views only ever need the server
to push new data to the browser — the browser never needs to push data
back. SSE is a one-directional protocol that matches this shape exactly and
is simpler than WebSocket (plain HTTP, no handshake/connection-state
management). WebSocket's bidirectional capability would be unused overhead
here.

**Inference/playground view is a separate, non-streaming-log concern.**
Typing a prompt and generating a response is a genuine request/response
interaction, not a push scenario — it's served by a plain HTTP endpoint that
streams the generated tokens back as they're produced, independent of the
SSE/polling mechanism used for the other two views. The three dashboard
views intentionally do not share a single update mechanism, since they have
different data-freshness needs (architecture is near-static, training is
continuously live, inference is on-demand).

**Architecture and attention-for-text endpoints read checkpoints directly,
not the snapshot log.** `GET /api/runs/{run_id}/architecture` and
`POST /api/runs/{run_id}/attention` were added alongside the frontend
because two pieces of data it needs were never in `snapshots.jsonl`: full
parameter shapes and model config (the log's `histogram` events only carry
summary stats, logged that way deliberately — see "Instrumentation
cadence") and attention weights for arbitrary playground text (the log's
`attention_sample` events only cover one fixed validation input, and
`/generate` is intentionally plain-text with no side channel — see below).
Both new endpoints load a checkpoint straight off disk and run inference
against it, matching the existing `/generate` endpoint's "reload each
request, no caching" approach rather than introducing a new one.

**Run selector built in from the start**, not deferred to a later
iteration. Each run lives in its own `runs/<run_id>/` directory; the
dashboard lets you pick which run's data to view rather than always
assuming "the current run." This was chosen over a v1-only-shows-latest-run
approach because comparing a new run's training curve against a previous
one seems likely to be genuinely useful given the project's exploratory
nature, and retrofitting a run selector later is more work than building it
in from the start.

**Frontend: plain HTML/JS + Plotly, no framework, no build step.** Chosen
specifically because there's no existing frontend background to build on —
the deciding factor was which option requires the least new skill just to
get a chart on screen, not which is most capable in the abstract. Plotly
was chosen over D3 or Chart.js because it has heatmaps as a first-class,
built-in chart type (`type: 'heatmap'`), which matters because the
attention-weight visualization was identified as the most bespoke part of
the dashboard; D3 could do this too but requires learning its whole
data-binding programming model first, which is a much larger investment. A
framework (React) was ruled out because the dashboard's three views are
loosely coupled panels, not an app with complex coordinated interactive
state — the shape that would justify a framework's overhead. Practically,
the dashboard is a single HTML file that pulls in Plotly from a CDN
`<script>` tag, plus vanilla JS to receive data (via SSE or polling) and
call `Plotly.newPlot()` / `Plotly.extendTraces()`.

## Commit conventions

Commits with substantial Claude Code-authored content carry a
`Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer, adopted
from the training-loop commit onward. Not applied retroactively to earlier
commits — not worth rewriting history for a cosmetic trailer.

## Open questions / not yet decided

- Exact PCA vs. t-SNE tradeoff for the embedding-projection view (PCA is
  cheap enough to compute at every M-step snapshot; t-SNE was discussed as
  better for periodic, less-frequent snapshots given its cost, but this
  hasn't been implemented yet).
- Whether the dashboard needs authentication/access control if ever exposed
  beyond localhost on the homelab network.
