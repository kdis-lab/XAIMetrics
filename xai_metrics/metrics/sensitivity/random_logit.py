# xai_metrics\metrics\sensitivity\random_logit.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc

from typing import Mapping, Any, Tuple, List

_EPS = 1e-8

@register_metric
class RandomLogit(BaseMetric):
    NAME = "RandomLogit"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] = None,
    ):
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("RandomLogit requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func


    def _check_number_classes(self, inputs: np.ndarray) -> int:
        configured_classes = self.params.get("num_classes")

        if configured_classes is not None:
            number_classes = int(configured_classes)

            if number_classes < 2:
                raise ValueError("num_classes must be at least 2.")

            return number_classes

        model = self.context.model

        if isinstance(model, torch.nn.Module):
            model_device = self.context.device

            if model_device is None:
                try:
                    model_device = str(next(model.parameters()).device)
                except StopIteration:
                    model_device = "cpu"

            model.eval()

            with torch.no_grad():
                output = model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))

            if isinstance(output, (Tuple, List)):
                output = output[0]

            predictions = output.detach().cpu().numpy()

        elif hasattr(model, "predict_proba"):
            predictions = np.asarray(model.predict_proba(inputs))

        else:
            raise MetricSkipped("RandomLogit requires 'num_classes' or a model with predict_proba / a PyTorch classification output.")

        if predictions.ndim != 2 or predictions.shape[1] < 2:
            raise MetricSkipped("RandomLogit requires a classifier with at least two output classes.")

        return int(predictions.shape[1])


    @staticmethod
    def _ssim(first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float64).reshape(-1)
        second = np.asarray(second, dtype=np.float64).reshape(-1)

        if first.shape != second.shape:
            raise ValueError("Original and randomized explanations must have the same shape.")

        minimum = min(float(np.min(first)), float(np.min(second)))
        maximum = max(float(np.max(first)), float(np.max(second)))
        data_range = maximum - minimum

        if data_range <= _EPS:
            return 1.0

        mean_first = float(np.mean(first))
        mean_second = float(np.mean(second))

        variance_first = float(np.mean((first - mean_first) ** 2))
        variance_second = float(np.mean((second - mean_second) ** 2))
        covariance = float(np.mean((first - mean_first) * (second - mean_second)))

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        numerator = (2.0 * mean_first * mean_second + c1) * (2.0 * covariance + c2)
        denominator = (mean_first ** 2 + mean_second ** 2 + c1) * (variance_first + variance_second + c2)

        return float(numerator / (denominator + _EPS))


    def run(self):
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        if ctx.y_test is None:
            raise MetricSkipped("RandomLogit requires classification targets.")

        targets = np.asarray(ctx.y_test.loc[ctx.observations])

        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        batch_size = p.get("batch_size", 64) or len(inputs)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")

        number_classes = self._check_number_classes(inputs)

        if targets.ndim == 2:
            target_classes = np.argmax(targets, axis=-1)
            one_hot_targets = True
        else:
            target_classes = targets.astype(int)
            one_hot_targets = False

        if np.any(target_classes < 0) or np.any(target_classes >= number_classes):
            raise ValueError("A target class is outside the model output range.")

        random_state = p.get("random_state", 42)
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size

            inputs_batch = inputs[start:stop]
            targets_batch = target_classes[start:stop]
            explanation_batch = explanations[start:stop]

            sampled_classes = rng.integers(low=0, high=number_classes - 1, size=len(inputs_batch))
            alternative_classes = np.where(sampled_classes >= targets_batch, sampled_classes + 1, sampled_classes)

            if one_hot_targets:
                alternative_classes = np.eye(number_classes, dtype=np.float32)[alternative_classes]
            else:
                alternative_targets = alternative_classes

            alternative_explanations = np.asarray(
                self.explain_func(
                    ctx.model,
                    inputs_batch,
                    alternative_targets
                ),
                dtype=np.float32
            )

            if alternative_explanations.shape != explanation_batch.shape:
                raise ValueError("explain_func must return explanations with the same shape as ctx.attributions.")

            scores.extend(
                self._ssim(original, alternative)
                for original, alternative in zip(explanation_batch, alternative_explanations)
            )

        return scores