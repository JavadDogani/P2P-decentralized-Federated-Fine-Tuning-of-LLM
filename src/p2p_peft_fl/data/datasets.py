from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    text_cols: Tuple[str, ...]
    label_col: str

SPECS: Dict[str, DatasetSpec] = {
    "ag_news": DatasetSpec(
        name="ag_news", hf_path="ag_news", hf_config=None,
        text_cols=("text",), label_col="label"
    ),
    "sst2": DatasetSpec(
        name="sst2", hf_path="glue", hf_config="sst2",
        text_cols=("sentence",), label_col="label"
    ),
    "imdb": DatasetSpec(
        name="imdb", hf_path="imdb", hf_config=None,
        text_cols=("text",), label_col="label"
    ),
}

def load_text_dataset(dataset: str, model_id: str, max_length: int,
                      max_train_samples: int, max_eval_samples: int):
    if dataset not in SPECS:
        raise ValueError(f"Unknown dataset={dataset}. Options: {list(SPECS)}")

    spec = SPECS[dataset]
    raw: DatasetDict = load_dataset(spec.hf_path, spec.hf_config) if spec.hf_config else load_dataset(spec.hf_path)

    # normalize splits
    if dataset == "ag_news":
        train_ds = raw["train"]
        test_ds = raw["test"]
    elif dataset == "imdb":
        train_ds = raw["train"]
        test_ds = raw["test"]
    elif dataset == "sst2":
        train_ds = raw["train"]
        test_ds = raw["validation"]  # GLUE uses validation
    else:
        raise RuntimeError("unreachable")

    train_ds = train_ds.shuffle(seed=0).select(range(min(max_train_samples, len(train_ds))))
    test_ds = test_ds.shuffle(seed=1).select(range(min(max_eval_samples, len(test_ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def _join_text(ex):
        if len(spec.text_cols) == 1:
            return ex[spec.text_cols[0]]
        # For future multi-field datasets
        return " ".join(str(ex[c]) for c in spec.text_cols)

    def tokenize_fn(batch):
        texts = [_join_text(ex) for ex in batch]
        return tokenizer(texts, truncation=True, max_length=max_length)

    # datasets.map can batch dicts; we convert to list-of-examples pattern:
    def map_tokenize(ds):
        # create list-of-examples view for speed
        return ds.map(
            lambda ex: tokenizer(_join_text(ex), truncation=True, max_length=max_length),
            batched=False,
            remove_columns=[c for c in ds.column_names if c not in (spec.label_col,)]
        )

    tok_train = map_tokenize(train_ds)
    tok_test = map_tokenize(test_ds)

    # Ensure labels are ints
    tok_train = tok_train.rename_column(spec.label_col, "labels") if spec.label_col != "labels" else tok_train
    tok_test = tok_test.rename_column(spec.label_col, "labels") if spec.label_col != "labels" else tok_test

    num_labels = int(max(tok_train["labels"])) + 1

    return tok_train, tok_test, tokenizer, num_labels
