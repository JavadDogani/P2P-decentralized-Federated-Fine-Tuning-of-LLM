from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict

@dataclass(frozen=True)
class PeftSpec:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"  # none | all | lora_only
    target_modules: Optional[list[str]] = None  # if None: PEFT will infer common modules

def attach_lora(model, spec: PeftSpec):
    cfg = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias=spec.bias,
        task_type="SEQ_CLS",
        target_modules=spec.target_modules,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model

def get_adapter_state(model) -> Dict[str, torch.Tensor]:
    sd = get_peft_model_state_dict(model)
    # make a detached CPU copy for "communication"
    return {k: v.detach().cpu().clone() for k, v in sd.items()}

def set_adapter_state(model, state: Dict[str, torch.Tensor]) -> None:
    # PEFT expects tensors on current device; it will move as needed if we give CPU tensors,
    # but we explicitly map to model's device to be safe.
    device = next(model.parameters()).device
    mapped = {k: v.to(device) for k, v in state.items()}
    set_peft_model_state_dict(model, mapped)
