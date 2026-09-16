# xai_metrics\metrics\fidelity\completeness\average_gain.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, register_metric, MetricContext

from typing import Mapping, Any, Tuple, List

_EPS = 1e-8


@register_metric
class AverageGain(BaseMetric):
    NAME = "AverageGain"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] = None):
        super().__init__(context, params)


    def _score(self, inputs: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
            if self.operator is not None:
                result = self.operator(self.context.model, inputs, targets)
                return np.asarray(result, dtype=np.float32).reshape(-1)
    
            # get prediction
            if isinstance(self.context.model, torch.nn.Module):
                model_device = self.context.device
                if model_device is None:
                    try:
                        model_device = str(next(self.context.model.parameters()).device)
                    except StopIteration:
                        model_device = "cpu"
                self.context.model.eval()
                with torch.no_grad():
                    output = self.context.model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))
                if isinstance(output, (Tuple, List)):
                    output = output[0]
                prediction = output.detach().cpu().numpy()
            elif hasattr(self.context.model, "predict_proba"):
                prediction = np.asarray(self.context.model.predict_proba(inputs)) # pyright: ignore[reportCallIssue]
            elif hasattr(self.context.model, "predict"):
                prediction = np.asarray(self.context.model.predict(inputs)) # pyright: ignore[reportCallIssue]
            else:
                prediction = np.asarray(self.context.model(inputs))
    
            if self.activation == 'sigmoid':
                prediction = 1.0 / (1.0 + np.exp(-prediction))
            elif self.activation == 'softmax':
                shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
                exp_prediction = np.exp(shifted)
                prediction = exp_prediction / np.sum(exp_prediction, axis=-1, keepdims=True)
    
            if prediction.ndim == 1:
                return prediction.astype(np.float32)
            if prediction.ndim != 2:
                raise ValueError("The default AverageGain operator requires predictions shaped (N, C).")
            if targets is None:
                return np.max(prediction, axis=-1).astype(np.float32)
            if targets.ndim == 1:
                return prediction[np.arange(len(prediction)), targets.astype(int)].astype(np.float32)
            return np.sum(prediction * targets, axis=-1, dtype=np.float32)
    

    def _score_batched(self, inputs: np.ndarray, targets: np.ndarray | None, batch_size: int) -> np.ndarray:
        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size
            target_batch = None if targets is None else targets[start:stop]
            scores.append(self._score(inputs[start:stop], target_batch))

        return np.concatenate(scores)
    

    @staticmethod
    def _perturb_with_mask(inputs: np.ndarray, explanations: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        explanations = np.asarray(explanations, dtype=np.float32)

        if len(inputs) != len(explanations):
            raise ValueError("The number of explanations must match the number of inputs.")

        mask = np.abs(explanations)

        if mask.ndim == 4:
            mask = np.mean(mask, axis=-1)

        axes = tuple(range(1, mask.ndim))

        if not axes:
            raise ValueError("Explanations must include at least one feature axis.")

        mask_min = np.min(mask, axis=axes, keepdims=True)
        mask_max = np.max(mask, axis=axes, keepdims=True)

        mask = (mask - mask_min) / (mask_max - mask_min + _EPS)

        # Image attribution maps: (N, H, W) -> (N, H, W, 1).
        if inputs.ndim == 4 and mask.ndim == 3:
            mask = mask[..., None]

        # Time-series attribution maps: (N, T) -> (N, T, 1).
        if inputs.ndim == 3 and mask.ndim == 2:
            mask = mask[..., None]

        try:
            mask = np.broadcast_to(mask, inputs.shape)
        except ValueError as exc:
            raise ValueError("The explanation shape is incompatible with the input shape.") from exc

        return (inputs * mask).astype(np.float32)


    def run(self):
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )
        targets = None if ctx.y_test is None else np.asarray(ctx.y_test.loc[ctx.observations]).reshape(-1)

        batch_size = p.get("batch_size", 64) or len(inputs)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        
        self.operator = p.get("operator")

        self.activation = p.get("activation")
        if self.activation not in (None, 'sigmoid', 'softmax'):
            raise ValueError("activation must be None, 'sigmoid', or 'softmax'.")

        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        if len(inputs) == 0:
            return []
        
        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size

            targets_batch = None if targets is None else targets[start:stop]
            inputs_batch = inputs[start:stop]
            explanations_batch = explanations[start:stop]

            base = self._score_batched(inputs_batch, targets_batch, len(inputs_batch))
            perturbed_inputs = self._perturb_with_mask(inputs_batch, explanations_batch)
            after = self._score_batched(perturbed_inputs, targets_batch, len(inputs_batch))

            batch_scores = np.maximum(after - base, 0.0) / (1.0 - base - _EPS)

            scores.append(batch_scores)

        return np.concatenate(scores).astype(float).tolist()