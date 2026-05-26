from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs


@dataclass
class DynaPromptConfig:
    batch_size: int = 512
    num_classes: int = 2
    temperature: float = 1.0
    bias_lr: float = 0.2
    temperature_lr: float = 0.02
    max_temperature: float = 3.0
    min_temperature: float = 0.25


class DynaPrompt(TTAMethod):
    """Prompt-free proxy for dynamic prompt tuning via online logit calibration."""

    def __init__(self, config: DynaPromptConfig | None = None):
        super().__init__(name="dynaprompt")
        self.config = config or DynaPromptConfig()
        self.class_bias: torch.Tensor | None = None
        self.temperature: float = self.config.temperature

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.class_bias = torch.zeros(self.config.num_classes)
        self.temperature = self.config.temperature

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.class_bias is None:
            raise RuntimeError("DynaPrompt.fit() must be called before predict_proba().")

        model.eval()
        bias = self.class_bias.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = test_feats[start : start + self.config.batch_size].float().to(device)
            base_probs = binary_model_probs(model, feats)
            logits = (base_probs.clamp_min(1e-8).log() + bias) / self.temperature
            out = torch.softmax(logits, dim=1)
            probs.append(out.cpu())

            entropy = -(out * out.clamp_min(1e-8).log()).sum(dim=1).mean()
            target = out.mean(dim=0)
            bias = bias + self.config.bias_lr * (target - target.mean())
            if entropy > 0.5:
                self.temperature = min(self.config.max_temperature, self.temperature + self.config.temperature_lr)
            else:
                self.temperature = max(self.config.min_temperature, self.temperature - self.config.temperature_lr)

        return torch.cat(probs, dim=0)
