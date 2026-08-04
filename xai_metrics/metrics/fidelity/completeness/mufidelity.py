# xai_metrics/metrics/fidelity/completeness/mufidelity.py
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, register_metric, MetricContext

from typing import Any, Mapping, Tuple, List

@register_metric
class MuFidelity(BaseMetric):
    NAME = "MuFidelity"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
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
            raise ValueError("The default MuFidelity operator requires predictions shaped (N, C).")
        if targets is None:
            return np.max(prediction, axis=-1).astype(np.float32)
        if targets.ndim == 1:
            return prediction[np.arange(len(prediction)), targets.astype(int)].astype(np.float32)
        return np.sum(prediction * targets, axis=-1, dtype=np.float32)


    def _perturb_samples(self, inputs: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Create Xplique-compatible Bernoulli masks and apply the baseline."""
        if inputs.ndim == 2:
            masks = self.rng.uniform(size=(count, inputs.shape[1])) > self.subset_percent
        elif inputs.ndim == 3:
            # Nearest-neighbour resize of (grid_size, channels) to (time, channels).
            coarse = self.rng.uniform(size=(count, self.grid_size, inputs.shape[2])) > self.subset_percent
            time_indices = np.floor(np.arange(inputs.shape[1]) * self.grid_size / inputs.shape[1]).astype(int)
            masks = coarse[:, time_indices, :]
        else:
            # Nearest-neighbour resize of a grid, shared by every image channel.
            coarse = self.rng.uniform(size=(count, self.grid_size, self.grid_size)) > self.subset_percent
            height_indices = np.floor(np.arange(inputs.shape[1]) * self.grid_size / inputs.shape[1]).astype(int)
            width_indices = np.floor(np.arange(inputs.shape[2]) * self.grid_size / inputs.shape[2]).astype(int)
            masks = coarse[:, height_indices][:, :, width_indices, None]

        masks = np.broadcast_to(masks.astype(np.float32), (len(inputs), *masks.shape)).copy()
        repeated_inputs = np.repeat(inputs[:, None, ...], count, axis=1)
        baseline_mode = self.params.get("baseline_mode", 0.0)
        baseline = baseline_mode(repeated_inputs) if callable(baseline_mode) else baseline_mode
        degraded = repeated_inputs * masks + (1.0 - masks) * np.asarray(baseline, dtype=np.float32)
        return degraded.reshape((-1, *inputs.shape[1:])), masks


    def run(self):
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )
        targets = np.asarray(ctx.y_test.loc[ctx.observations]).reshape(-1)

        nb_samples = int(p.get("nb_samples", p.get("n_masks", 200)))
        batch_size = p.get("batch_size", 64) or (len(inputs) * nb_samples)
        self.grid_size = p.get("grid_size", None) or inputs.shape[1]
        self.subset_percent = float(p.get("subset_percent", p.get("subset_probability", 0.2)))
        self.operator = p.get("operator")
        self.activation = p.get("activation")

        if self.activation not in (None, 'sigmoid', 'softmax'):
            raise ValueError("activation must be None, 'sigmoid', or 'softmax'.")

        perturbation_batch_size = min(batch_size, nb_samples)
        inputs_batch_size = max(1, batch_size // perturbation_batch_size)

        random_state=p.get("random_state")
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        base_predictions = self._score(inputs, targets)
        correlations = []
        for start in range(0, len(inputs), inputs_batch_size):
            stop = start + inputs_batch_size
            inputs_slice = inputs[start:stop]
            targets_slice= None if targets is None else targets[start:stop]
            phi = explanations[start:stop]
            base = base_predictions[start:stop, None]

            if inputs_slice.ndim > phi.ndim:
                phi = phi[..., None]
            phi = phi[:, None, ...]

            prediction_drops = []
            attribution_sums = []
            generated = 0
            while generated < nb_samples:
                count = min(perturbation_batch_size, nb_samples - generated)
                generated += count
                degraded, masks = self._perturb_samples(inputs, count)
                repeated_targets = None if targets is None else np.repeat(targets, count, axis=0)
                perturbed = self._score(degraded, repeated_targets).reshape(len(inputs), count)
                prediction_drops.append(base - perturbed)
                attribution_sums.append(np.sum(phi * (1.0 - masks), axis=tuple(range(2, masks.ndim))))

            predictions = np.concatenate(prediction_drops, axis=1)
            attributes = np.concatenate(attribution_sums, axis=1)
            for prediction, attribute in zip(predictions, attributes):
                correlation = spearmanr(prediction, attribute).statistic
                correlations.append(0.0 if np.isnan(correlation) else float(correlation))

        return float(np.mean(correlations))