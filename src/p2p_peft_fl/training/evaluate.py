from __future__ import annotations
from typing import Dict, Tuple
import torch

@torch.no_grad()
def eval_loop(model, dataloader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        total += labels.numel()
        correct += (preds == labels).sum().item()
        total_loss += float(loss.item()) * labels.size(0)
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return {"acc": acc, "loss": avg_loss}
