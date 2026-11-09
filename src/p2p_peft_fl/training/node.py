from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from p2p_peft_fl.peft.factory import get_adapter_state, set_adapter_state

@dataclass
class Node:
    node_id: int
    model: torch.nn.Module
    tokenizer: any
    train_loader: DataLoader
    eval_loader: DataLoader
    device: torch.device
    lr: float
    weight_decay: float
    fp16: bool = False

    def __post_init__(self):
        self.model.to(self.device)
        self.model.train()
        # Only trainable params (adapters)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.fp16 and self.device.type == "cuda"))

    def local_train(self, steps: int, log_every: int = 10) -> Dict[str, float]:
        self.model.train()
        it = iter(self.train_loader)
        total_loss = 0.0
        n = 0
        for step in range(1, steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss = out.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            total_loss += float(loss.item())
            n += 1

        return {"train_loss": total_loss / max(1, n)}

    def export_adapter(self) -> Dict[str, torch.Tensor]:
        return get_adapter_state(self.model)

    def import_adapter(self, state: Dict[str, torch.Tensor]) -> None:
        set_adapter_state(self.model, state)
