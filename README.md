# toy-llm

A small hand-written transformer for visualizing architecture, training, and inference.

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

## Notes
- `pyenv local toy-llm` writes `.python-version` — as long as you `cd` into the repo with pyenv's shell hook active, the venv activates automatically. No manual `source .venv/bin/activate` needed.
- To list existing pyenv virtualenvs: `pyenv virtualenvs`
- To remove this one if you need to rebuild it: `pyenv uninstall toy-llm`
