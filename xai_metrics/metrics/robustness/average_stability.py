# xai_metrics\metrics\robustness\average_stability.py
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, MetricSkipped, register_metric
from xai_metrics.base.types import ExplainFunc

from collections.abc import Callable
from typing import Mapping, Any, cast

@register_metric
class AverageStability(BaseMetric):
    NAME = "AverageStability"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("AverageStability requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

    def run(self):
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )
        targets = None if ctx.y_test is None else np.asarray(ctx.y_test.loc[ctx.observations])

        explanations = np.asarray(ctx.attributions, dtype=np.float32)

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        if len(explanations) != len(inputs):
            raise ValueError("The number of explanations must match the number of inputs.")

        radius = float(p.get("radius", 0.1))
        nb_samples = int(p.get("nb_samples", 20))
        distance = p.get("distance", "l2")
        random_state = p.get("random_state")

        if radius < 0.0:
            raise ValueError("radius must be non-negative.")

        if nb_samples <= 0:
            raise ValueError("nb_samples must be positive.")

        if isinstance(distance, str):
            if distance not in ('l1', 'l2'):
                raise ValueError("distance must be 'l1', 'l2', or a callable.")
        elif callable(distance):
            distance = cast(Callable[[np.ndarray, np.ndarray], float], distance)
        else:
            raise ValueError("distance must be 'l1', 'l2', or a callable.")

        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

        # These noise masks are created ones and reused for every observation
        noisy_masks = rng.uniform(low=0.0, high=radius, size=(nb_samples, *inputs.shape[1:])).astype(np.float32)

        scores = []

        for index, (input_sample, explanation) in enumerate(zip(inputs, explanations)):
            neighbours = input_sample[None, ...] + noisy_masks

            neighbour_targets = (
                None
                if targets is None
                else np.repeat(
                    targets[index:index + 1],
                    nb_samples,
                    axis=0
                )
            )

            neighbour_explanations = np.asarray(
                self.explain_func(
                    ctx.model,
                    neighbours,
                    neighbour_targets
                ),
                dtype=np.float32
            )

            if len(neighbour_explanations) != nb_samples:
                raise ValueError("explain_func must return one explanation per neighbouring input.")

            distances = []

            for neighbour_explanation in neighbour_explanations:
                if callable(distance):
                    value = distance(neighbour_explanation, explanation)
                elif distance == 'l1':
                    value = np.sum(np.abs(neighbour_explanation - explanation))
                else:
                    value = np.sqrt(np.sum((neighbour_explanation - explanation) ** 2))

                distances.append(float(value))

            scores.append(float(np.mean(distances)))

        return scores