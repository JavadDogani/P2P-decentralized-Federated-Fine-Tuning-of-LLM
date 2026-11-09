"""Unit tests for data loading, model/PEFT helpers, the Node wrapper, logging and argument parsing."""

from __future__ import annotations

import json
import math
from dataclasses import fields

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding

from p2p_peft_fl.cli import parse_args, pick_device
from p2p_peft_fl.config import ExperimentConfig
from p2p_peft_fl.data.datasets import SPECS, load_text_dataset, tokenize_splits
from p2p_peft_fl.models.factory import build_model, resolve_model_id
from p2p_peft_fl.peft.factory import PeftSpec, attach_lora, get_adapter_state, set_adapter_state
from p2p_peft_fl.training.evaluate import eval_loop
from p2p_peft_fl.training.node import Node
from p2p_peft_fl.utils.logging import RunLogger


# --- data ---------------------------------------------------------------------------------

def test_tokenize_splits_renames_labels_and_counts_classes(tokenizer):
    train = Dataset.from_dict({"text": ["the movie was good", "bad plot", "great acting", "dull"], "label": [1, 0, 2, 0]})
    test = Dataset.from_dict({"text": ["very good"], "label": [1]})
    tok_train, tok_test, num_labels = tokenize_splits(SPECS["ag_news"], train, test, tokenizer, max_length=16)
    assert num_labels == 3
    assert "labels" in tok_train.column_names and "label" not in tok_train.column_names
    assert "text" not in tok_train.column_names
    assert {"input_ids", "attention_mask"} <= set(tok_train.column_names)
    assert tok_train["labels"] == [1, 0, 2, 0]
    assert len(tok_test) == 1


def test_tokenize_splits_truncates_to_max_length(tokenizer):
    long = " ".join(["movie"] * 50)
    ds = Dataset.from_dict({"sentence": [long], "label": [0]})
    tok_train, _, _ = tokenize_splits(SPECS["sst2"], ds, ds, tokenizer, max_length=8)
    assert len(tok_train[0]["input_ids"]) == 8


def test_unknown_dataset_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_text_dataset("mnist", model_id="unused", max_length=8, max_train_samples=1, max_eval_samples=1)


def test_dataset_specs_cover_documented_datasets():
    assert set(SPECS) == {"ag_news", "sst2", "imdb"}
    assert SPECS["sst2"].text_cols == ("sentence",)


# --- models / PEFT ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "alias,model_id",
    [("bert", "bert-base-uncased"), ("ALBERT", "albert-base-v2"), ("roberta", "roberta-base"),
     ("distilbert", "distilbert-base-uncased"), ("prajjwal1/bert-tiny", "prajjwal1/bert-tiny")],
)
def test_resolve_model_id(alias, model_id):
    assert resolve_model_id(alias) == model_id


def test_build_model_loads_a_local_checkpoint(tiny_model_dir):
    model, model_id = build_model(tiny_model_dir, num_labels=3)
    assert model_id == tiny_model_dir
    assert model.config.num_labels == 3
    assert model.classifier.out_features == 3


def test_lora_adapter_contains_only_lora_and_head(tiny_model_dir):
    model, _ = build_model(tiny_model_dir, num_labels=2)
    peft_model = attach_lora(model, PeftSpec())
    state = get_adapter_state(peft_model)
    assert state, "adapter state must not be empty"
    for k in state:
        assert "lora_" in k or "classifier" in k, k
    # PEFT infers query/value as LoRA targets for BERT
    assert any(".query.lora_A" in k for k in state)
    assert any(".value.lora_B" in k for k in state)
    # exported tensors are detached CPU copies
    for v in state.values():
        assert v.device.type == "cpu" and not v.requires_grad
    n_adapter = sum(v.numel() for v in state.values())
    n_total = sum(p.numel() for p in peft_model.parameters())
    assert n_adapter < 0.5 * n_total


TINY = dict(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=64, num_labels=2)


def tiny_classifier(model_type: str):
    if model_type == "distilbert":
        cfg = AutoConfig.for_model("distilbert", vocab_size=64, dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                                   max_position_embeddings=64, num_labels=2)
    elif model_type == "albert":
        cfg = AutoConfig.for_model("albert", embedding_size=8, **TINY)
    else:
        cfg = AutoConfig.for_model(model_type, **TINY)
    return AutoModelForSequenceClassification.from_config(cfg)


@pytest.mark.parametrize(
    "model_type",
    [
        "bert",
        "roberta",
        pytest.param("albert", marks=pytest.mark.xfail(
            strict=True, raises=ValueError,
            reason="Known issue: PEFT has no default LoRA target modules for ALBERT and PeftSpec.target_modules "
            "is None, so `--model albert` fails in attach_lora.")),
        pytest.param("distilbert", marks=pytest.mark.xfail(
            strict=True, raises=ValueError,
            reason="Known issue: PEFT has no default LoRA target modules for DistilBERT and "
            "PeftSpec.target_modules is None, so `--model distilbert` fails in attach_lora.")),
    ],
)
def test_attach_lora_with_default_targets(model_type):
    peft_model = attach_lora(tiny_classifier(model_type), PeftSpec())
    assert any("lora_A" in name for name, _ in peft_model.named_parameters())


def test_adapter_state_roundtrip(tiny_model_dir):
    a, _ = build_model(tiny_model_dir, num_labels=2)
    b, _ = build_model(tiny_model_dir, num_labels=2)
    pa, pb = attach_lora(a, PeftSpec()), attach_lora(b, PeftSpec())
    sa = {k: v + 0.5 for k, v in get_adapter_state(pa).items()}
    set_adapter_state(pb, sa)
    sb = get_adapter_state(pb)
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k])


# --- node / evaluation --------------------------------------------------------------------

def make_node(model_dir, loader, node_id=0, fp16=False):
    tok_train, tok_test, tokenizer, num_labels = loader("sst2", model_dir, 32, 1000, 1000)
    model, _ = build_model(model_dir, num_labels)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return Node(
        node_id=node_id,
        model=attach_lora(model, PeftSpec()),
        tokenizer=tokenizer,
        train_loader=DataLoader(tok_train, batch_size=8, shuffle=True, collate_fn=collator),
        eval_loader=DataLoader(tok_test, batch_size=8, shuffle=False, collate_fn=collator),
        device=torch.device("cpu"),
        lr=1e-2,
        weight_decay=0.0,
        fp16=fp16,
    )


def test_node_local_train_updates_only_the_adapter(tiny_model_dir, synthetic_loader):
    torch.manual_seed(0)
    node = make_node(tiny_model_dir, synthetic_loader, fp16=True)  # fp16 is ignored on CPU
    frozen_before = {n: p.detach().clone() for n, p in node.model.named_parameters() if not p.requires_grad}
    adapter_before = node.export_adapter()

    # more steps than batches per epoch, so the loader is re-iterated
    metrics = node.local_train(steps=8)
    assert set(metrics) == {"train_loss"} and math.isfinite(metrics["train_loss"])

    adapter_after = node.export_adapter()
    assert any(not torch.equal(adapter_before[k], adapter_after[k]) for k in adapter_before)
    for n, p in node.model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(frozen_before[n], p), f"frozen parameter {n} changed"


def test_node_import_then_export_roundtrip(tiny_model_dir, synthetic_loader):
    node = make_node(tiny_model_dir, synthetic_loader)
    state = {k: torch.full_like(v, 0.25) for k, v in node.export_adapter().items()}
    node.import_adapter(state)
    out = node.export_adapter()
    for k in state:
        assert torch.equal(out[k], state[k])


def test_eval_loop_reports_accuracy_and_loss(tiny_model_dir, synthetic_loader):
    node = make_node(tiny_model_dir, synthetic_loader)
    ev = eval_loop(node.model, node.eval_loader, device=torch.device("cpu"))
    assert set(ev) == {"acc", "loss"}
    assert 0.0 <= ev["acc"] <= 1.0
    assert ev["loss"] > 0 and math.isfinite(ev["loss"])


# --- logging / config / CLI ---------------------------------------------------------------

def test_run_logger_writes_jsonl(tmp_path):
    cfg = ExperimentConfig()
    log = RunLogger(str(tmp_path / "run"), cfg)
    log.event("eval", round=0, node=1, acc=0.5)
    log.close()
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["run_start", "eval", "run_end"]
    assert events[0]["config"]["nodes"] == cfg.nodes
    assert events[1]["acc"] == 0.5 and events[1]["t"] >= 0


def test_parse_args_defaults_match_config():
    cfg = parse_args([])
    defaults = ExperimentConfig()
    for f in fields(ExperimentConfig):
        if f.name != "agg":
            assert getattr(cfg, f.name) == getattr(defaults, f.name), f.name
    # the aggregation policy follows --peft ("lora" -> plain mean)
    assert cfg.agg == cfg.peft == "lora"


def test_parse_args_overrides():
    cfg = parse_args(["--peft", "dlora", "--nodes", "5", "--topology", "grid", "--protocol", "epidemic",
                      "--non_iid_alpha", "0.1", "--comm_topk", "0.2", "--fp16"])
    assert (cfg.peft, cfg.agg, cfg.nodes, cfg.topology, cfg.protocol) == ("dlora", "dlora", 5, "grid", "epidemic")
    assert cfg.non_iid_alpha == 0.1 and cfg.comm_topk == 0.2 and cfg.fp16 is True


@pytest.mark.parametrize("argv", [["--peft", "adalora"], ["--topology", "star"], ["--protocol", "flood"]])
def test_parse_args_rejects_unknown_choices(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_pick_device_cpu():
    assert pick_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert pick_device("cuda").type == "cpu"
        assert pick_device("auto").type == "cpu"
