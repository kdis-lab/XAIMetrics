# xai_metrics\metrics\robustness\mege.py
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc, TrainingFunc

from typing import Mapping, Any, Callable, Tuple, List

@register_metric
class MeGe(BaseMetric):
    NAME = "MeGe"

    def __init__(
        self,
        context: MetricContext,
        training_func: TrainingFunc,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        if training_func is None:
            raise ValueError("MeGe requires 'training_func' to be provided via dependencies.")

        if explain_func is None:
            raise ValueError("AverageStability requires 'explain_func' to be provided via dependencies.")

        self.training_func = training_func
        self.explain_func = explain_func


    @staticmethod
    def _predict_classes(model: Any, inputs: np.ndarray, device: str | None) -> np.ndarray:
        if isinstance(model, torch.nn.Module):
            model_device = device

            if model_device is None:
                try:
                    model_device = str(next(model.parameters()).device)
                except StopIteration:
                    model_device = 'cpu'

            model.eval()

            with torch.no_grad():
                output = model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))

            if isinstance(output, (Tuple, List)):
                output = output[0]

            predictions = output.detach().cpu().numpy()

            if predictions.ndim == 1:
                return predictions.astype(int)

            return np.argmax(predictions, axis=-1)

        if hasattr(model, "predict"):
            predictions = np.asarray(model.predict(inputs))

            if predictions.ndim == 1:
                return predictions

            indices = np.argmax(predictions, axis=-1)

            if hasattr(model, "classes_") and len(model.classes_) == predictions.shape[-1]:
                return np.asarray(model.classes_)[indices]

            return indices

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(inputs))
            indices = np.argmax(probabilities, axis=-1)

            if hasattr(model, "classes_") and len(model.classes_) == probabilities.shape[-1]:
                return np.asarray(model.classes_)[indices]

            return indices

        raise MetricSkipped("MeGe requires models that implement predict(), predict_proba(), or are PyTorch modules.")

    @staticmethod
    def _spearman_distance(first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)

        if first.ndim > 2:
            first = np.mean(first, axis=-1)
        if second.ndim > 2:
            second = np.mean(second, axis=-1)

        first = np.nan_to_num(first, nan=0.0).reshape(-1)
        second = np.nan_to_num(second, nan=0.0).reshape(-1)

        correlation = spearmanr(first, second).statistic # pyright: ignore[reportAttributeAccessIssue]

        if np.isnan(correlation):
            return np.nan

        return float(np.sqrt(max(0.0, 1.0 - abs(correlation))))


    def run(self):
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )

        if ctx.y_test is None:
            raise MetricSkipped("MeGe requires classification targets.")

        targets = np.asarray(ctx.y_test.loc[ctx.observations])

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        k_splits = int(p.get("k_splits", 4))

        if k_splits < 2:
            raise ValueError("k_splits must be at least 2.")

        if len(inputs) % k_splits != 0:
            raise ValueError("The number of selected observations must be divisible by k_splits")

        split_len = len(inputs) // k_splits

        x_splits = inputs.reshape(k_splits, split_len, *inputs.shape[1:])
        y_splits = targets.reshape(k_splits, split_len, *targets.shape[1:])

        models = []

        for split_index in range(k_splits):
            train_splits = [
                index
                for index in range(k_splits)
                if index != split_index
            ]

            X_train = x_splits[train_splits].reshape(-1, *inputs.shape[1:])
            y_train = x_splits[train_splits].reshape(-1, *targets.shape[1:])

            model = self.training_func(X_train, y_train, x_splits[split_index], y_splits[split_index])
            models.append(model)

        predictions = []
        explanations = []

        for model in models:
            predictions.append(self._predict_classes(model, inputs, ctx.device))
            explanations.append(np.asarray(self.explain_func(model, inputs, targets), dtype=np.float32))

        predictions = np.asarray(predictions)
        explanations = np.asarray(explanations)

        if explanations.shape[1] != len(inputs):
            raise ValueError("explain_func must return one explanation for every selected observation.")

        target_classes = np.argmax(targets, axis=-1) if targets.ndim == 2 else targets

        equal_prediction_distances = [[] for _ in range(len(inputs))]

        for first_model_index in range(k_splits):
            for second_model_index in range(first_model_index + 1, k_splits):
                for observation_index in range(len(inputs)):
                    split_index = observation_index // split_len

                    # Compare a sample only when one of both models did not train on it
                    if split_index not in (first_model_index, second_model_index):
                        continue

                    first_prediction = predictions[first_model_index, observation_index]
                    second_prediction = predictions[second_model_index, observation_index]
                    target = target_classes[observation_index]

                    # At least one model must predict the true class
                    if first_prediction != target and second_prediction != target:
                        continue

                    if first_prediction != second_prediction:
                        continue

                    distance = self._spearman_distance(
                        explanations[first_model_index, observation_index],
                        explanations[second_model_index, observation_index]
                    )

                    if not np.isnan(distance):
                        equal_prediction_distances[observation_index].append(distance)

        scores = []

        for distances in equal_prediction_distances:
            if distances:
                scores.append(float(1.0 / (1.0 + np.mean(distances))))
            else:
                # No model pair produced a valid comparison for this observation.
                scores.append(float("nan"))

        if np.all(np.isnan(scores)):
            raise MetricSkipped("MeGe skipped: no valid equal-prediction explanation pairs were produced.")

        return scores