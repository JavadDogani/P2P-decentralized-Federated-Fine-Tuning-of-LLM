"""End-to-end runs of the simulator on a tiny random BERT and synthetic sentences (no downloads)."""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest
import torch

from p2p_peft_fl import cli
from p2p_peft_fl.config import ExperimentConfig


def tiny_config(tmp_path, model_dir, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig(
        model=model_dir,
        nodes=3,
        rounds=2,
        local_steps=2,
        batch_size=8,
        lr=1e-2,
        max_length=32,
        max_train_samples=48,
        max_eval_samples=16,
        device="cpu",
        seed=0,
        non_iid_alpha=100.0,  # near-IID so every node gets data
        run_dir=str(tmp_path / "run"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.agg = cfg.peft  # as parse_args does
    return cfg


def read_events(run_dir):
    with open(f"{run_dir}/events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_fully_connected_epidemic_mean_reaches_consensus(tmp_path, tiny_model_dir, synthetic_loader):
    cfg = tiny_config(tmp_path, tiny_model_dir, topology="full", protocol="epidemic", peft="lora")
    result = cli.run_experiment(cfg, data_loader=synthetic_loader)
    n, rounds = cfg.nodes, cfg.rounds

    assert len(result.nodes) == n

    events = read_events(result.run_dir)
    counts = Counter(e["type"] for e in events)
    assert counts == {
        "run_start": 1, "node_init": n, "topology": 1, "round_start": rounds, "local_train": rounds * n,
        "messages": rounds, "eval": rounds * n, "round_end": rounds, "run_end": 1,
    }
    # epidemic on a full graph: every node pushes to every other node
    assert [e["count"] for e in events if e["type"] == "messages"] == [n * (n - 1)] * rounds
    assert sum(e["train_size"] for e in events if e["type"] == "node_init") == cfg.max_train_samples
    for e in events:
        if e["type"] == "local_train":
            assert math.isfinite(e["train_loss"])

    # every inbox holds all n adapters, so after aggregation all peers hold the same adapter ...
    states = [nd.export_adapter() for nd in result.nodes]
    for other in states[1:]:
        assert other.keys() == states[0].keys()
        for k in states[0]:
            torch.testing.assert_close(other[k], states[0][k])
    # ... and therefore report the same metrics on the shared test set
    assert len(result.evals) == rounds * n
    for r in range(rounds):
        evs = [e for e in result.evals if e["round"] == r]
        assert len({e["acc"] for e in evs}) == 1
        for e in evs:
            assert e["loss"] == pytest.approx(evs[0]["loss"], rel=1e-5)


@pytest.mark.parametrize(
    "topology,protocol,peft,comm_topk",
    [
        ("ring", "gossip", "dlora", 1.0),
        ("grid", "epidemic", "wlora", 1.0),
        ("random", "gossip", "lora", 0.5),
    ],
)
def test_topology_protocol_aggregation_combinations_run(tmp_path, tiny_model_dir, synthetic_loader,
                                                        topology, protocol, peft, comm_topk):
    cfg = tiny_config(tmp_path, tiny_model_dir, nodes=4, topology=topology, protocol=protocol, peft=peft, comm_topk=comm_topk)
    result = cli.run_experiment(cfg, data_loader=synthetic_loader)

    events = read_events(result.run_dir)
    adj = {int(k): v for k, v in next(e["adj"] for e in events if e["type"] == "topology").items()}
    expected = cfg.nodes if protocol == "gossip" else sum(len(v) for v in adj.values())
    assert [e["count"] for e in events if e["type"] == "messages"] == [expected] * cfg.rounds

    assert len(result.evals) == cfg.rounds * cfg.nodes
    for e in result.evals:
        assert 0.0 <= e["acc"] <= 1.0
        assert math.isfinite(e["loss"])


def test_runs_are_reproducible_for_a_fixed_seed(tmp_path, tiny_model_dir, synthetic_loader):
    a = cli.run_experiment(tiny_config(tmp_path / "a", tiny_model_dir, peft="dlora"), data_loader=synthetic_loader)
    b = cli.run_experiment(tiny_config(tmp_path / "b", tiny_model_dir, peft="dlora"), data_loader=synthetic_loader)
    assert a.evals == b.evals
    for na, nb in zip(a.nodes, b.nodes):
        sa, sb = na.export_adapter(), nb.export_adapter()
        for k in sa:
            assert torch.equal(sa[k], sb[k])


def test_main_parses_argv_and_runs(tmp_path, monkeypatch, tiny_model_dir, synthetic_loader):
    # main() always uses the real dataset loader; swap it for the offline stand-in
    monkeypatch.setattr(cli, "load_text_dataset", synthetic_loader)
    run_dir = tmp_path / "cli_run"
    cli.main([
        "--model", tiny_model_dir,
        "--nodes", "2", "--topology", "full", "--protocol", "gossip", "--peft", "wlora",
        "--rounds", "1", "--local_steps", "1", "--max_train_samples", "16", "--max_eval_samples", "8",
        "--max_length", "32", "--device", "cpu", "--non_iid_alpha", "100", "--run_dir", str(run_dir),
    ])
    events = read_events(run_dir)
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    assert events[0]["config"]["agg"] == "wlora"
    assert [e["count"] for e in events if e["type"] == "messages"] == [2]
    assert sum(e["type"] == "eval" for e in events) == 2
