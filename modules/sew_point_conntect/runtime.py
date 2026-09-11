"""Imperative torch adapter for connector inference.

``TorchConnectorRuntime`` owns one checkpoint/model/device for a run.  Callers
should construct one runtime per batch, sequence, or evaluator and reuse it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .inference_core import (
    resolve_postprocess_params,
    resolve_threshold,
    score_to_edge_results,
)
from .model_registry import DEFAULT_MODEL, get_model


class TorchConnectorRuntime:
    """Load a connector checkpoint once and execute it on graph samples."""

    def __init__(self, model_path: str | Path, device: str | torch.device | None = None):
        self.model_path = str(model_path)
        self.device = self._resolve_device(device)
        # Keep strict checkpoint loading and map_location behavior compatible
        # with the original entry points.
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        args = self.checkpoint.get("args", {})
        self.args = args
        self.threshold = resolve_threshold(self.checkpoint)
        self.model = get_model(
            str(self.checkpoint.get("model_key") or args.get("model_name") or DEFAULT_MODEL),
            node_dim=int(self.checkpoint["node_dim"]),
            edge_dim=int(self.checkpoint["edge_dim"]),
            hidden_dim=int(args.get("hidden_dim", 128)),
            num_layers=int(args.get("num_layers", 3)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def predict_sample(self, sample) -> Any:
        """Return sigmoid probabilities for a ``GraphSample``."""

        with torch.no_grad():
            logits = self.model(
                sample.node_x.to(self.device),
                sample.edge_index.to(self.device),
                sample.edge_attr.to(self.device),
                sample.edge_patch.to(self.device),
            )
            return torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()

    def predict_edges(
        self,
        sample,
        annotation: dict,
        threshold: float | None = None,
        postprocess_preset: str = "balanced",
        max_degree: int | None = None,
        max_small_cycle_length: int | None = None,
        continuity_weight: float | None = None,
        cycle_penalty: float | None = None,
        point_xy: dict[int, tuple[float, float]] | None = None,
    ) -> list[dict]:
        """Score one graph sample and return the stable public edge schema."""

        if sample is None:
            return []
        probabilities = self.predict_sample(sample)
        edge_index = sample.edge_index.detach().cpu().t().tolist()
        if point_xy is None:
            point_xy = {
                int(point.get("id", idx)): (float(point["x"]), float(point["y"]))
                for idx, point in enumerate(annotation.get("points", []))
            }
        params = resolve_postprocess_params(
            preset=postprocess_preset,
            max_degree=max_degree,
            max_small_cycle_length=max_small_cycle_length,
            continuity_weight=continuity_weight,
            cycle_penalty=cycle_penalty,
        )
        return score_to_edge_results(
            probabilities,
            edge_index,
            sample.point_ids,
            threshold=resolve_threshold(self.checkpoint, threshold),
            point_xy=point_xy,
            postprocess_params=params,
        )
