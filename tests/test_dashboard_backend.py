import asyncio
import contextlib
import json
import threading
import time

import httpx
import torch
import uvicorn
from fastapi.testclient import TestClient

from dashboard.backend.app import create_app
from dashboard.backend.inference import encode_prompt, find_checkpoint
from dashboard.backend.runs import _read_new_events, list_runs, tail_snapshots
from model.config import GPTConfig
from tests.test_train import make_synthetic_data
from train import TrainConfig, train


@contextlib.contextmanager
def running_server(app):
    """Serve `app` on a real loopback socket.

    httpx's in-process ASGITransport (what TestClient uses under the hood) fully drains an
    endpoint's response before returning it, which deadlocks against our intentionally
    never-ending SSE generator. A real socket doesn't have that problem, so live-tailing is
    tested against one instead of TestClient.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def write_jsonl(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


# --- run discovery ---------------------------------------------------------


def test_list_runs_orders_newest_first_and_skips_runs_without_a_snapshot_file(tmp_path):
    runs_dir = tmp_path / "runs"
    write_jsonl(runs_dir / "20260101_000000" / "snapshots.jsonl", [{"type": "loss", "step": 0, "value": 1.0}])
    write_jsonl(runs_dir / "20260102_000000" / "snapshots.jsonl", [{"type": "loss", "step": 0, "value": 1.0}])
    (runs_dir / "20260103_000000_not_started").mkdir(parents=True)

    checkpoint_dir = runs_dir / "20260102_000000" / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step_0000000.pt").touch()
    (checkpoint_dir / "step_0000050.pt").touch()

    runs = list_runs(runs_dir)

    assert [r["run_id"] for r in runs] == ["20260102_000000", "20260101_000000"]
    assert runs[0]["checkpoint_steps"] == [0, 50]
    assert runs[1]["checkpoint_steps"] == []


def test_list_runs_missing_dir_returns_empty_list(tmp_path):
    assert list_runs(tmp_path / "does-not-exist") == []


# --- log tailing -------------------------------------------------------------


def test_read_new_events_leaves_a_partial_trailing_line_for_next_read(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    path.write_text('{"type": "loss", "step": 0, "value": 1.0}\n{"type": "loss", "step": 1')

    events, offset = _read_new_events(path, 0)
    assert events == [{"type": "loss", "step": 0, "value": 1.0}]

    with open(path, "a") as f:
        f.write(', "value": 0.9}\n')

    more_events, _ = _read_new_events(path, offset)
    assert more_events == [{"type": "loss", "step": 1, "value": 0.9}]


def test_tail_snapshots_replays_existing_lines_then_streams_new_ones(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(path, [{"type": "loss", "step": 0, "value": 1.0}])

    async def body():
        stop_event = asyncio.Event()
        gen = tail_snapshots(path, poll_interval=0.01, stop_event=stop_event)
        try:
            assert await gen.__anext__() == {"type": "loss", "step": 0, "value": 1.0}

            with open(path, "a") as f:
                f.write(json.dumps({"type": "loss", "step": 1, "value": 0.9}) + "\n")

            assert await gen.__anext__() == {"type": "loss", "step": 1, "value": 0.9}
        finally:
            stop_event.set()
            await gen.aclose()

    asyncio.run(body())


# --- HTTP API ----------------------------------------------------------------


def test_get_runs_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    write_jsonl(runs_dir / "run1" / "snapshots.jsonl", [{"type": "loss", "step": 0, "value": 1.0}])

    client = TestClient(create_app(runs_dir=runs_dir))
    response = client.get("/api/runs")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "run1"


def test_stream_endpoint_404_for_unknown_run(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.get("/api/runs/does-not-exist/stream")
    assert response.status_code == 404


def test_stream_endpoint_replays_and_tails_live(tmp_path):
    runs_dir = tmp_path / "runs"
    snapshot_path = runs_dir / "run1" / "snapshots.jsonl"
    write_jsonl(snapshot_path, [{"type": "loss", "step": 0, "value": 1.0}])

    app = create_app(runs_dir=runs_dir, poll_interval=0.02)

    async def body(base_url):
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            async with client.stream("GET", "/api/runs/run1/stream") as response:
                assert response.status_code == 200
                lines = response.aiter_lines()

                async def next_event():
                    async for line in lines:
                        if line.startswith("data: "):
                            return json.loads(line[len("data: ") :])

                assert await next_event() == {"type": "loss", "step": 0, "value": 1.0}

                with open(snapshot_path, "a") as f:
                    f.write(json.dumps({"type": "loss", "step": 1, "value": 0.9}) + "\n")

                assert await next_event() == {"type": "loss", "step": 1, "value": 0.9}

    with running_server(app) as base_url:
        asyncio.run(body(base_url))


def _train_tiny_run(tmp_path, max_steps=20, checkpoint_every=10):
    torch.manual_seed(0)
    train_data, val_data, vocab_meta = make_synthetic_data()
    stoi = {ch: i for i, ch in vocab_meta["itos"].items()}

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "meta.json", "w") as f:
        json.dump({"vocab_size": vocab_meta["vocab_size"], "stoi": stoi, "itos": vocab_meta["itos"]}, f)

    model_config = GPTConfig(
        vocab_size=vocab_meta["vocab_size"], block_size=16, n_layer=2, n_head=2, n_embd=16
    )
    train_config = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(tmp_path / "runs"),
        learning_rate=1e-2,
        batch_size=16,
        max_steps=max_steps,
        log_every=10,
        checkpoint_every=checkpoint_every,
        device="cpu",
        seed=0,
    )
    run_dir, _ = train(train_data, val_data, model_config, train_config, vocab_meta=vocab_meta)
    return run_dir, vocab_meta


def test_find_checkpoint_selects_latest_or_a_specific_step(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)

    assert find_checkpoint(run_dir).name == "step_0000010.pt"
    assert find_checkpoint(run_dir, step=0).name == "step_0000000.pt"


def test_find_checkpoint_raises_for_missing_step(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)
    try:
        find_checkpoint(run_dir, step=5)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_encode_prompt_rejects_out_of_vocab_characters():
    stoi = {"a": 0, "b": 1, "c": 2}
    try:
        encode_prompt("abz", stoi, device=torch.device("cpu"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generate_endpoint_streams_text_from_a_checkpoint(tmp_path):
    run_dir, vocab_meta = _train_tiny_run(tmp_path)

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(
        f"/api/runs/{run_dir.name}/generate", json={"prompt": "a", "max_new_tokens": 5}
    )

    assert response.status_code == 200
    assert len(response.text) == 5
    assert all(ch in vocab_meta["itos"].values() for ch in response.text)


def test_generate_endpoint_404_for_unknown_run(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post("/api/runs/no-such-run/generate", json={"prompt": ""})
    assert response.status_code == 404


def test_generate_endpoint_400_for_out_of_vocab_prompt(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(f"/api/runs/{run_dir.name}/generate", json={"prompt": "$$$"})
    assert response.status_code == 400


# --- architecture and attention endpoints -------------------------------------


def test_architecture_endpoint_returns_config_and_parameter_shapes(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.get(f"/api/runs/{run_dir.name}/architecture")

    assert response.status_code == 200
    body = response.json()
    assert body["step"] == 10  # latest checkpoint from _train_tiny_run
    assert body["config"]["n_layer"] == 2
    assert body["total_params"] == sum(p["num_params"] for p in body["parameters"])

    by_name = {p["name"]: p for p in body["parameters"]}
    assert by_name["token_emb.weight"]["shape"] == [body["config"]["vocab_size"], body["config"]["n_embd"]]
    assert "mean" in by_name["token_emb.weight"]["stats"]


def test_architecture_endpoint_respects_step_query_param(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.get(f"/api/runs/{run_dir.name}/architecture", params={"step": 0})

    assert response.status_code == 200
    assert response.json()["step"] == 0


def test_architecture_endpoint_404_for_unknown_run(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.get("/api/runs/no-such-run/architecture")
    assert response.status_code == 404


def test_attention_endpoint_returns_per_layer_weights_for_given_text(tmp_path):
    run_dir, vocab_meta = _train_tiny_run(tmp_path)
    text = "".join(list(vocab_meta["itos"].values())[:3])

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(f"/api/runs/{run_dir.name}/attention", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["tokens"] == list(text)
    assert len(body["layers"]) == 2  # n_layer from _train_tiny_run
    # attn_weights: (n_head, T, T)
    attn = body["layers"][0]["attn_weights"]
    assert len(attn[0]) == len(text)
    assert len(attn[0][0]) == len(text)


def test_attention_endpoint_404_for_unknown_run(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post("/api/runs/no-such-run/attention", json={"text": "a"})
    assert response.status_code == 404


def test_attention_endpoint_400_for_out_of_vocab_text(tmp_path):
    run_dir, _ = _train_tiny_run(tmp_path)
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(f"/api/runs/{run_dir.name}/attention", json={"text": "$$$"})
    assert response.status_code == 400
