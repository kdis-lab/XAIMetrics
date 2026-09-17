# xai_metrics\metrics\fidelity\completeness\insertion.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, register_metric, MetricContext, MetricSkipped

from typing import Mapping, Any, Tuple, List

@register_metric
class Insertion(BaseMetric):
    NAME = "Insertion"

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
                raise ValueError("The default Insertion operator requires predictions shaped (N, C).")
            if targets is None:
                return np.asarray(np.max(prediction, axis=-1), dtype=np.float32)
            if targets.ndim == 1:
                return np.asarray(prediction[np.arange(len(prediction)), targets.astype(int)], dtype=np.float32)
            return np.asarray(np.sum(prediction * targets, axis=-1, dtype=np.float32), dtype=np.float32)

    def run(self) -> float:
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )
        targets = None if ctx.y_test is None else np.asarray(ctx.y_test.loc[ctx.observations]).reshape(-1)

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")
        
        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        batch_size = int(p.get("batch_size", 64))

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        max_percentage_perturbed = float(p.get("max_percentage_perturbed", 1.0))
        if not 0.0 < max_percentage_perturbed <= 1.0:
            raise ValueError("max_percentage_perturbed must be in (0, 1].")

        flattened_inputs = inputs.reshape(len(inputs), -1)
        flattened_explanations = explanations.reshape(len(explanations), -1)

        if flattened_inputs.shape != flattened_explanations.shape:
            raise ValueError(
                "Inputs and explanations must contain the same number "
                "of features per observation."
            )

        nb_features = flattened_inputs.shape[1]
        max_nb_perturbed = int(np.floor(nb_features * max_percentage_perturbed))
        max_nb_perturbed = max(1, max_nb_perturbed)

        steps = int(p.get("steps", 10))
        if steps == -1:
            steps = max_nb_perturbed
        if steps <= 0:
            raise ValueError("steps must be positive or equal to -1.")
        
        self.operator = p.get("operator")

        self.activation = p.get("activation")
        if self.activation not in (None, 'sigmoid', 'softmax'):
            raise ValueError("activation must be None, 'sigmoid', or 'softmax'.")

        baseline_mode = self.params.get("baseline_mode", 0.0)
        baseline = baseline_mode(inputs) if callable(baseline_mode) else baseline_mode

        try:
            baselines = np.broadcast_to(np.asarray(baseline, dtype=np.float32), inputs.shape).copy()
        except ValueError as exc:
            raise ValueError("baseline_mode must be a scalar or return an array broadcastable to the selected inputs.") from exc

        flattened_baselines = baselines.reshape(len(inputs), -1)

        start_state = flattened_baselines
        end_state = flattened_inputs

        steps = np.linspace(0, max_nb_perturbed, steps + 1, dtype=int)
        most_important_features = np.argsort(flattened_explanations, axis=-1)[:, ::-1]

        scores_dict = {}
        for step in steps:
            ids_to_flip = most_important_features[:, :step]
            batch_inputs = start_state.copy()

            for observation_index, feature_ids in enumerate(ids_to_flip):
                batch_inputs[observation_index, feature_ids] = end_state[observation_index, feature_ids]

            batch_inputs = batch_inputs.reshape(inputs.shape)
            scores = []
            
            for start in range(0, len(batch_inputs), batch_size):
                stop = start + batch_size
                targets_batch = None if targets is None else targets[start:stop]
    
                scores.append(self._score(batch_inputs[start:stop], targets_batch))
    
            predictions = np.concatenate(scores)

            scores_dict[int(step)] = np.asarray(predictions, dtype=np.float64)

        scores_by_step = np.stack(list(scores_dict.values()), axis=0)

        if len(scores_by_step) == 1:
            return scores_by_step[0].tolist()

        observation_scores = np.mean(scores_by_step[:-1] + scores_by_step[1:], axis=0) * 0.5

        return observation_scores.tolist()