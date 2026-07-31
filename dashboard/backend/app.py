"""FastAPI dashboard backend. Reads what train.py writes; never talks back to it.

- GET  /api/runs                     list available runs
- GET  /api/runs/{run_id}/stream     SSE: replays a run's snapshot log, then tails it live
- POST /api/runs/{run_id}/generate   plain-text streamed generation from a run's checkpoint
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dashboard.backend.inference import encode_prompt, generate_stream, load_checkpoint_for_inference
from dashboard.backend.runs import POLL_INTERVAL_SECONDS, list_runs, tail_snapshots


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: Optional[int] = None
    checkpoint_step: Optional[int] = None  # None = latest checkpoint


def create_app(runs_dir="runs", poll_interval=POLL_INTERVAL_SECONDS):
    app = FastAPI(title="toy-llm dashboard backend")
    runs_dir = Path(runs_dir)

    @app.get("/api/runs")
    def get_runs():
        return list_runs(runs_dir)

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        snapshot_path = runs_dir / run_id / "snapshots.jsonl"
        if not snapshot_path.exists():
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

        async def event_source():
            async for event in tail_snapshots(snapshot_path, poll_interval=poll_interval):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/generate")
    def generate(run_id: str, body: GenerateRequest):
        run_dir = runs_dir / run_id
        try:
            model, stoi, itos, device = load_checkpoint_for_inference(run_dir, body.checkpoint_step)
            idx = encode_prompt(body.prompt, stoi, device)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return StreamingResponse(
            generate_stream(model, idx, itos, body.max_new_tokens, body.temperature, body.top_k),
            media_type="text/plain",
        )

    return app


app = create_app()
