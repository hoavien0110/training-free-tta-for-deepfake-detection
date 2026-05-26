from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from deepfake_tta.modeling import evaluate_scores


@dataclass
class TTAMethod(ABC):
    name: str

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        """Optional preparation step using source/cache data."""

    @abstractmethod
    def predict_proba(
        self,
        model: Any,
        test_feats: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Return class probabilities with shape [N, C]."""

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        test_feats: torch.Tensor,
        test_labels: torch.Tensor,
        device: str,
        name: str = "test",
        show_report: bool = True,
    ) -> dict[str, float]:
        probs = self.predict_proba(model, test_feats, device)
        return evaluate_scores(
            y_true=test_labels.numpy(),
            y_score=probs[:, 1].numpy(),
            y_pred=probs.argmax(dim=1).numpy(),
            name=name,
            show_report=show_report,
        )


class MethodNotImplemented(TTAMethod):
    paper_name: str
    reason: str

    def __init__(self, name: str, paper_name: str, reason: str):
        super().__init__(name=name)
        self.paper_name = paper_name
        self.reason = reason

    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name} ({self.paper_name}) is scaffolded but not implemented yet. "
            f"{self.reason}"
        )
