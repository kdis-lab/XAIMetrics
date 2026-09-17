# xai_metrics\metrics\sensitivity\random_logit.py
import copy
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc

from typing import Mapping, Any

_EPS = 1e-8

@register_metric
class ModelRandomization(BaseMetric):
    NAME = "ModelRandomization"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
    ):
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("ModelRandomization requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func


    @staticmethod
    def _spearman(original: np.ndarray, randomized: np.ndarray) -> float:
        original = np.asarray(original, dtype=np.float64)
        randomized = np.asarray(randomized, dtype=np.float64)

        if original.shape != randomized.shape:
            raise ValueError("Original and randomized explanations must have the same shape.")

        if original.ndim == 3:
            original = np.mean(original, axis=-1)
            randomized = np.mean(randomized, axis=-1)

        original = original.reshape(-1)
        randomized = randomized.reshape(-1)

        correlation = spearmanr(original, randomized).statistic # pyright: ignore[reportAttributeAccessIssue]

        return 0.0 if np.isnan(correlation) else float(correlation)


    def run(self):
        ctx = self.context
        p = self.params

        if not isinstance(ctx.model, torch.nn.Module):
            raise MetricSkipped("ModelRandomization currently requires a PyTorch model.")

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        targets = None if ctx.y_test is None else np.asarray(ctx.y_test.loc[ctx.observations])

        original_explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(original_explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(original_explanations)} vs {len(inputs)}."
            )

        batch_size = p.get("batch_size", 64) or len(inputs)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")

        # Equivalent to ProgrssiveLayerRandomization(0.25)
        fraction = float(p.get("fraction", 0.25))
        reverse = bool(p.get("reverse", True))

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in the interval [0, 1].")

        random_state = p.get("random_state", 42)
        rng = np.random.default_rng(random_state)

        # Deep copy so ctx.model is never mutated
        randomized_model = copy.deepcopy(ctx.model)

        layers = [
            module
            for module in randomized_model.modules()
            if list(module.parameters(recurse=False))
        ]

        if not layers:
            raise MetricSkipped("The model contains no parameterised layers.")

        if reverse:
            layers.reverse()

        number_layers = int(len(layers) * fraction)

        with torch.no_grad():
            for layer in layers[:number_layers]:
                for parameter in layer.parameters(recurse=False):
                    parameter.copy_(
                        torch.as_tensor(
                            rng.uniform(
                                low=0.0,
                                high=1.0,
                                size=tuple(parameter.shape)
                            ),
                            dtype=parameter.dtype,
                            device=parameter.device
                        )
                    )

        randomized_model.eval()
        
        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size

            inputs_batch = inputs[start:stop]
            targets_batch = None if targets is None else targets[start:stop]
            original_batch = original_explanations[start:stop]

            randomized_explanations = np.asarray(
                self.explain_func(
                    randomized_model,
                    inputs_batch,
                    targets_batch
                ),
                dtype=np.float32
            )

            if randomized_explanations.shape != original_batch.shape:
                raise ValueError("explain_func must return explanations with the same shape as ctx.attributions.")

            scores.extend(
                self._spearman(original, randomized)
                for original, randomized in zip(original_batch, randomized_explanations)
            )

        return scores