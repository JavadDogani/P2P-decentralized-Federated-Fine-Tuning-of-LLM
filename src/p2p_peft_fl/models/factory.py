from __future__ import annotations
from typing import Tuple
from transformers import AutoModelForSequenceClassification

MODEL_ALIASES = {
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2",
    "roberta": "roberta-base",
    # convenient lightweight option
    "distilbert": "distilbert-base-uncased",
}

def resolve_model_id(name_or_id: str) -> str:
    return MODEL_ALIASES.get(name_or_id.lower(), name_or_id)

def build_model(name_or_id: str, num_labels: int):
    model_id = resolve_model_id(name_or_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model, model_id
