# xai_metrics\metrics\fidelity\completeness\insertion.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, register_metric, MetricContext, MetricSkipped

from typing import Mapping, Any, Tuple, List

@register_metric
class Insertion(BaseMetric):
    """
    Insertion fidelity metric.

    This metric evaluates whether the features identified as important by an
    explanation are relevant to the model prediction. Starting from a baseline
    input, features are progressively restored to their original values following
    their attribution-based importance ranking.

    For each observation, features with larger absolute attribution values are
    inserted first. After each insertion step, the model score is recomputed,
    producing an insertion curve that describes how the score evolves as
    increasingly important features are restored.

    A faithful explanation is expected to identify features that strongly
    influence the model output. Consequently, inserting highly important features
    should produce a rapid increase in the model score and therefore a large area
    under the insertion curve. Higher values are better.

    For each observation ``i``, the procedure can be described as:

    .. math::
        x_i^(0) = b_i

    .. math::
        x_i^(k) = insert(b_i, x_i, M_i, k)

    .. math::
        score_i^(k) = g(f, x_i^(k), y_i)

    where ``f`` is the model, ``g`` is the scoring operator, ``b_i`` is the
    baseline input, ``M_i`` defines the feature ranking induced by the
    explanation and ``k`` is the number of restored features.

    The final score for each observation is computed as the average trapezoidal
    value of the model scores along the insertion trajectory. A larger value
    indicates that inserting the features considered important by the explanation
    increases the model score more rapidly.

    The implementation is based on the Insertion metric introduced by Petsiuk
    et al. (2018) and follows the implementation provided by Xplique, with
    support for PyTorch models, models exposing ``predict`` or
    ``predict_proba``, custom scoring operators and configurable baseline values.

    Petsiuk, V., Das, A., & Saenko, K. (2018).
    RISE: Randomized Input Sampling for Explanation of Black-box Models.
    In British Machine Vision Conference (BMVC).
    """
    NAME = "Insertion"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, selected observations and attribution values. ``y_test``
            may be ``None`` when the model score can be determined without
            explicit targets.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``batch_size`` : int, optional
              Maximum number of perturbed observations processed at once when
              computing model scores. The default value is ``64``.

            - ``baseline_mode`` : float, array-like or callable, optional
              Baseline used as the initial state of the insertion process. A
              scalar value is broadcast to the input shape. If callable, it
              receives the selected input observations and must return a value
              or array broadcastable to their shape. The default value is
              ``0.0``.

            - ``steps`` : int, optional
              Number of insertion intervals between the baseline and the maximum
              perturbation level. A value of ``-1`` evaluates every possible
              number of inserted features up to the configured maximum. The
              default value is ``10``.

            - ``max_percentage_perturbed`` : float, optional
              Maximum fraction of input features that may be restored from the
              original input. Must belong to the interval ``(0, 1]``. The
              default value is ``1.0``.

            - ``operator`` : callable or None, optional
              Custom scoring function ``g(model, inputs, targets)`` used to
              obtain one scalar score per observation. If ``None``, the metric
              performs inference directly using the model and selects the
              corresponding target score. The default value is ``None``.

            - ``activation`` : {None, "sigmoid", "softmax"}, optional
              Activation function applied to model predictions before selecting
              the score. ``None`` leaves predictions unchanged. The default
              value is ``None``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        When no custom ``operator`` is provided, model inference is performed
        using, in order, a PyTorch ``torch.nn.Module``, ``predict_proba``,
        ``predict`` or direct model invocation.

        For two-dimensional model outputs, targets are interpreted as class
        indices. If no targets are provided, the maximum model output for each
        observation is used.

        Inputs and explanations are flattened before the insertion process.
        Therefore, each explanation must contain exactly the same number of
        feature values as its corresponding input observation.

        Features are ranked by the absolute value of their attributions in
        descending order. Consequently, features with the largest attribution
        magnitude are restored first, independently of the attribution sign.
        """
        super().__init__(context, params)

    
    def _score(self, inputs: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
        """
        Compute one scalar model score per observation.

        If a custom ``operator`` is configured, it is used directly. Otherwise,
        the method performs model inference and optionally applies the configured
        activation function.

        For two-dimensional predictions ``(N, C)``, the score corresponding to
        each target class is selected. If ``targets`` is ``None``, the maximum
        prediction over the class dimension is returned.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input observations.
        targets : np.ndarray or None
            Target class indices associated with the observations. If ``None``,
            the maximum model output is used for two-dimensional predictions.

        Returns
        -------
        np.ndarray
            One scalar score per observation, with shape ``(N,)``.

        Raises
        ------
        ValueError
            If the default scoring procedure receives model predictions with a
            shape other than ``(N,)`` or ``(N, C)``.
        """
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

    def run(self):
        """
        Compute the Insertion metric.

        The method selects the observations defined in the metric context and
        progressively restores their features from a configured baseline to their
        original values. Features are inserted in descending order of their
        absolute attribution values, so that the features considered most
        important by the explanation are restored first.

        Inputs and explanations are flattened before ranking and perturbation.
        The insertion process starts from the baseline input. For each insertion
        step, the selected number of features is replaced by their corresponding
        values from the original input and the model score is recomputed.

        The sequence of model scores obtained at the different insertion levels
        forms an insertion curve for each observation. The final score is
        computed by averaging the trapezoidal values between consecutive points
        of this curve.

        Higher scores are better. A large value indicates that restoring features
        identified as important by the explanation rapidly increases the model
        score, suggesting that the explanation successfully captures features
        relevant to the prediction.

        Returns
        -------
        List[float]
            Insertion score for each evaluated observation. Higher values indicate
            better explanations.

        Raises
        ------
        MetricSkipped
            If no observations are selected.
        ValueError
            If the number of explanations differs from the number of selected
            observations, if inputs and explanations do not contain the same
            number of features per observation, if ``batch_size`` is not
            positive, if ``max_percentage_perturbed`` does not belong to
            ``(0, 1]``, if ``steps`` is neither positive nor ``-1``, if
            ``activation`` is not one of ``None``, ``"sigmoid"`` or
            ``"softmax"``, or if the configured baseline cannot be broadcast to
            the selected input shape.

        Notes
        -----
        ``max_percentage_perturbed`` determines the maximum portion of the
        original input that can be restored. For example, a value of ``0.5``
        limits the insertion process to half of the available features.

        Setting ``steps=-1`` evaluates every possible number of inserted features
        between zero and the maximum number determined by
        ``max_percentage_perturbed``.

        The returned values are computed independently for each observation.
        They are not automatically aggregated into a dataset-level mean.

        Applying ``activation="softmax"`` or ``activation="sigmoid"`` is useful
        when the desired interpretation is the increase in class probability
        rather than the increase in an arbitrary model score or logit.
        """
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
        most_important_features = np.argsort(np.abs(flattened_explanations), axis=-1)[:, ::-1]

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