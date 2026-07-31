"""Run discovery and log-tailing.

Polls runs/<run_id>/snapshots.jsonl from the last known byte offset on a
short interval, rather than watching the filesystem — see DESIGN.md
"Dashboard architecture" for why.
"""

import asyncio
import json
from pathlib import Path

POLL_INTERVAL_SECONDS = 0.3


def list_runs(runs_dir):
    """List available runs, newest first, with their logged checkpoint steps."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []

    runs = []
    for run_path in sorted(runs_dir.iterdir(), reverse=True):
        if not run_path.is_dir():
            continue
        snapshot_path = run_path / "snapshots.jsonl"
        if not snapshot_path.exists():
            continue

        checkpoint_dir = run_path / "checkpoints"
        checkpoint_steps = sorted(
            int(p.stem.split("_")[1]) for p in checkpoint_dir.glob("step_*.pt")
        ) if checkpoint_dir.exists() else []

        runs.append(
            {
                "run_id": run_path.name,
                "last_updated": snapshot_path.stat().st_mtime,
                "checkpoint_steps": checkpoint_steps,
            }
        )
    return runs


def _read_new_events(path, offset):
    """Read complete JSON lines appended after `offset`, returning (events, new_offset).

    A trailing line without a newline yet (still being written) is left for the next read.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()

    if not chunk:
        return [], offset

    text = chunk.decode("utf-8")
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]
    lines = [line for line in lines if line]

    events = [json.loads(line) for line in lines]
    consumed = sum(len(line.encode("utf-8")) + 1 for line in lines)
    return events, offset + consumed


async def tail_snapshots(snapshot_path, poll_interval=POLL_INTERVAL_SECONDS, stop_event=None):
    """Yield JSON events from snapshot_path: first whatever's already there, then new lines as
    they're appended. Runs until `stop_event` is set (or forever, if not given)."""
    snapshot_path = Path(snapshot_path)

    offset = 0
    if snapshot_path.exists():
        events, offset = _read_new_events(snapshot_path, offset)
        for event in events:
            yield event

    while stop_event is None or not stop_event.is_set():
        await asyncio.sleep(poll_interval)
        if not snapshot_path.exists():
            continue
        events, offset = _read_new_events(snapshot_path, offset)
        for event in events:
            yield event
