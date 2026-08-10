# toy-llm

A small hand-written transformer for visualizing architecture, training, and inference.
See [DESIGN.md](DESIGN.md) for the reasoning behind the choices below.

## Setup

1. Create the pyenv virtualenv (Python 3.12.12):
```bash
   pyenv virtualenv 3.12.12 toy-llm
```

2. Pin this project to it (creates `.python-version`, auto-activates on `cd`):
```bash
   cd toy-llm
   pyenv local toy-llm
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Verify GPU is visible to torch:
```bash
   python -c "import torch; print(torch.cuda.is_available())"
```
   Should print `True`.

## Training

Training and the dashboard are separate processes — run them side by side to
watch a run in progress.

```bash
python train.py
```

Each invocation starts a new run, writing checkpoints and a streamed
`snapshots.jsonl` log to a fresh `runs/<timestamp>_<id>/` directory. Useful
flags:

- `--max-steps 2000` — total training steps
- `--log-every 50` — weight/gradient histogram + attention-sample cadence
- `--checkpoint-every 500` — checkpoint + embedding-PCA cadence
- `--n-layer` / `--n-head` / `--n-embd` / `--block-size` — model size
- `--batch-size 32` — samples per gradient step

### Choosing parameters

This project's priority is legibility over capability (see [DESIGN.md](DESIGN.md)), so
pick these with "what will this look like in the dashboard" in mind rather
than optimizing for loss:

- **Model size** (`n_layer`, `n_head`, `n_embd`, `block_size`): keep small.
  `n_layer × n_head` is how many attention heatmaps you browse in the
  selector, and `n_embd` is what gets PCA'd to 2D — both get hard to read
  past a handful. `2/2/64/64` (~108K params) is a good default: enough
  structure to compare early vs. late layers, small enough that the
  parameter table and bar chart in the Architecture tab don't need
  scrolling. The GPU (GTX 1650, 4GB) is not the constraint here.
- **`max_steps`**: 300 is a quick smoke test — enough to confirm the
  pipeline and dashboard work end to end, but the loss curve barely
  flattens. 2000 still shows visible descent rather than a plateau
  (loss dropped ~0.22 in the last 1000 of those steps); on this
  model/dataset it takes ~8000–10000 steps before the curve actually
  flattens out (in one measured run, the last 5000 steps only bought
  another ~0.12) — that's the length worth using if the point is to
  see a full convergence curve, not just a downward trend.
- **`log_every` ("N")**: trades a smoother-looking histogram/attention
  animation against per-snapshot GPU→CPU overhead. Aim for ~15–30
  snapshots total, i.e. `max_steps / 20`.
- **`checkpoint_every` ("M", must be `M >> N`)**: controls how many
  points you get in the Architecture tab's checkpoint selector. Each
  checkpoint is a full state dict (~1.3MB at this model size) plus an
  embedding PCA pass — cheap, so pick based on how many points-in-time
  you want to compare (3–5 is usually plenty).
- **`batch_size`**: smaller batches (e.g. 16) give a noisier, more
  visually "alive" loss curve; memory isn't a real constraint at this
  scale either way.

Two presets:

```bash
# Quick smoke test — verify the pipeline and dashboard render correctly
python train.py --max-steps 300 --log-every 20 --checkpoint-every 100 \
  --n-layer 2 --n-head 2 --n-embd 64 --block-size 64 --batch-size 16

# Longer run — full convergence curve (plateaus around step ~8000-10000
# on this model/dataset), several checkpoints to browse
python train.py --max-steps 10000 --log-every 250 --checkpoint-every 2000 \
  --n-layer 2 --n-head 2 --n-embd 64 --block-size 64 --batch-size 32
```

## Intuitive views of how the LLM works

The model trains on a tiny-shakespeare corpus with character-level
tokenization (see [DESIGN.md](DESIGN.md) for why) — so "characters" below means literal
letters/punctuation, and "ROMEO:" is an actual prompt this project's model
was given. The three illustrations below simplify the core ideas, but the
specific behaviors they call out (the layer 0/layer 1 attention split, the
loss curve's shape) aren't hypothetical — they're what was actually observed
on a real run of this model, documented in detail in
[`docs/findings.md`](docs/findings.md).

### How attention lets the model read text

![How attention reads text](docs/images/how-attention-reads-text.svg)

The bottom row is a sequence of characters ("ROMEO:"). Each layer above is
a full pass over the whole sequence, and the lines show attention: how
much each token "looks at" every other token when deciding what comes
next. Layer 0's lines cluster on nearby characters — a recency bias.
Layer 1's dashed line reaches further back for one specific character —
sparser, more selective linking, consistent with a deeper layer building
on the previous layer's output rather than raw token embeddings. This
split is exactly what the trained model's real attention weights show —
see "Attention patterns" in `docs/findings.md`.

### Training as descending a loss landscape

![Training as descending a landscape](docs/images/training-as-descending-a-landscape.svg)

The vertical axis is loss (how wrong the model's predictions are); the
ball is the model's current state, always rolling toward lower loss.
Early on the surface is steep, so a single training step buys a big drop.
Later, the surface flattens, so each step only nudges the ball a little —
matching this project's actual loss curve, which falls sharply in the
first ~500 steps, then decays slowly out to 8000-10,000 steps.

### Inference: generating one character at a time

![Inference autoregressive loop](docs/images/inference-autoregressive-loop.svg)

This is what happens *after* training, using the model's learned weights
(a saved checkpoint) rather than the random weights it started with: the
context goes through one forward pass of the model, which outputs a
probability for every character in the vocabulary rather than a single
answer. The most likely (or sampled) character is chosen, appended onto
the context, and the loop runs again with one more character in view —
this is why generation is inherently sequential, one pass at a time.

## How training works

The illustrations above are the intuitive picture; this is the literal one
— the same single training step, but with every stage and tensor shape as
actually implemented in `model/`, from the raw batch to the AdamW update.

![Training pipeline with tensor shapes](docs/images/training-pipeline.svg)

A single training step: a batch of characters is embedded, passed through
the transformer blocks (attention, then MLP, each wrapped in a residual
connection), projected to logits over the vocabulary, scored against the
true next character via cross-entropy loss, and used to update every
parameter via backward pass and AdamW. The tensor shape is annotated at
each handoff — notably, the residual stream stays a constant (B, 64, 64)
through every block; only the LM head expands it to (B, 64, 65) to score
against the vocabulary, and only the backward/AdamW steps operate in
per-parameter shape space rather than per-token shape space.

## Dashboard

The dashboard backend serves the frontend too, so one command runs both:

```bash
uvicorn dashboard.backend.app:app --reload
```

Then open `http://127.0.0.1:8000/` in a browser.

- Run it from the repo root, so the default `runs_dir="runs"` resolves correctly.
- `--reload` restarts the server on backend code changes; not needed for
  frontend-only edits, since `index.html` is served fresh from disk each
  request.
- Add `--port` if 8000 is already taken.
- The dashboard can be started before or after `train.py` — it just reads
  whatever's in `runs/` and tails the file. Pick the run from the "Run"
  dropdown and open the **Training** tab to watch loss, weight/gradient
  distributions, embedding PCA, and attention samples update live as
  training progresses.

![Dashboard Training tab](docs/images/dashboard-training-tab.png)

## Findings

[`docs/findings.md`](docs/findings.md) documents observations from an actual
training run — parameter distribution, loss/weight/gradient dynamics,
embedding PCA across checkpoints, and attention patterns from both training
and generation.

`docs/embedding_pca_compare.html` is a self-contained interactive chart
referenced from those findings; GitHub only shows HTML files as source, so
view it rendered via
[htmlpreview.github.io](https://htmlpreview.github.io/?https://raw.githubusercontent.com/codeinthecreek/toy-llm/main/docs/embedding_pca_compare.html)
instead.

## Notes
- `pyenv local toy-llm` writes `.python-version` — as long as you `cd` into the repo with pyenv's shell hook active, the venv activates automatically. No manual `source .venv/bin/activate` needed.
- To list existing pyenv virtualenvs: `pyenv virtualenvs`
- To remove this one if you need to rebuild it: `pyenv uninstall toy-llm`

## License

BSD 3-Clause. See [LICENSE](LICENSE).
