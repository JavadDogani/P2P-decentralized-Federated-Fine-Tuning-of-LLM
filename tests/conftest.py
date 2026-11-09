"""Shared fixtures: a tiny, randomly initialised BERT checkpoint and a synthetic SST-2-like dataset.

Nothing here touches the network. A small ``BertModel`` and a word-level tokenizer are
saved to a temporary directory once per session; passing that directory as the model id
makes the real ``build_model`` / ``AutoTokenizer.from_pretrained`` code paths load them
exactly like a Hub checkpoint. Only the raw dataset download is replaced.
"""

from __future__ import annotations

import os

# Fail fast instead of downloading if anything tries to reach the Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import random
from pathlib import Path
from typing import Dict, List

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

from p2p_peft_fl.data.datasets import SPECS, tokenize_splits

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
POSITIVE = ["good", "great", "fun", "lovely", "brilliant", "enjoyable"]
NEGATIVE = ["bad", "awful", "boring", "dull", "terrible", "painful"]
FILLER = ["the", "movie", "film", "was", "really", "quite", "plot", "acting", "and", "very"]
VOCAB: List[str] = SPECIAL + FILLER + POSITIVE + NEGATIVE


def make_tokenizer() -> PreTrainedTokenizerFast:
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def make_sentences(n: int, seed: int) -> Dict[str, list]:
    """Balanced binary sentiment data in the SST-2 schema (``sentence``, ``label``)."""
    rng = random.Random(seed)
    sentences, labels = [], []
    for i in range(n):
        label = i % 2
        words = rng.sample(FILLER, 4) + rng.sample(POSITIVE if label else NEGATIVE, 2)
        rng.shuffle(words)
        sentences.append(" ".join(words))
        labels.append(label)
    return {"sentence": sentences, "label": labels}


class SyntheticLoader:
    """Stands in for ``data.datasets.load_text_dataset``: same signature and return value,
    but the raw splits are generated locally instead of downloaded."""

    def __init__(self, n_train: int = 48, n_test: int = 16) -> None:
        self.n_train = n_train
        self.n_test = n_test

    def __call__(self, dataset: str, model_id: str, max_length: int, max_train_samples: int, max_eval_samples: int):
        train = Dataset.from_dict(make_sentences(self.n_train, seed=0))
        test = Dataset.from_dict(make_sentences(self.n_test, seed=1))
        train = train.select(range(min(max_train_samples, len(train))))
        test = test.select(range(min(max_eval_samples, len(test))))
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tok_train, tok_test, num_labels = tokenize_splits(SPECS["sst2"], train, test, tokenizer, max_length)
        return tok_train, tok_test, tokenizer, num_labels


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    """A local 'checkpoint' directory holding a tiny random BERT encoder and its tokenizer."""
    path: Path = tmp_path_factory.mktemp("tiny-bert")
    cfg = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        BertModel(cfg).save_pretrained(path)
    make_tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture
def synthetic_loader() -> SyntheticLoader:
    return SyntheticLoader()
