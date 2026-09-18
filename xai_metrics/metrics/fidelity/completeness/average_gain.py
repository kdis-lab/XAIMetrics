# xai_metrics\metrics\fidelity\completeness\average_gain.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, register_metric, MetricContext, MetricSkipped

from typing import Mapping, Any, Tuple, List

_EPS = 1e-8


@register_metric
class AverageGain(BaseMetric):
    r"""
    Average Gain fidelity metric.

    This metric evaluates how much the model score increases when the original
    input is masked according to the importance values provided by the
    explanation. Features with larger absolute attribution values are preserved
    to a greater extent, while features with smaller attribution values are
    attenuated.

    For each observation :math:`i`, the metric is computed as:

    .. math::

        \begin{aligned}
        \mathrm{base}_i &= 
        g(f, \mathbf{x}_i, y_i), \\
        \mathrm{after}_i &= 
        g(f, \mathbf{x}_i \odot M_i, y_i), \\
        \operatorname{AG}_i &= 
        \frac{
            \max(\mathrm{after}_i - \mathrm{base}_i, 0)
        }{
            1 - \mathrm{base}_i + \varepsilon
        }
        \end{aligned}

    where :math:`f` is the model, :math:`g` is the scoring operator and :math:`M_i`
    is a normalised mask derived from the attribution values.

    Attribution values are first converted to absolute values and independently
    min-max normalised to the interval :math:`[0, 1]` for each observation. The
    resulting mask is then broadcast to the input shape when necessary and
    multiplied element-wise by the original input.

    The metric returns one Average Gain value per observation. Higher values are
    better: a large score indicates that retaining the features considered
    important by the explanation increases the model score relative to the
    original input. A value of zero indicates that masking the input does not
    increase the score.

    Average Gain is complementary to Average Drop. While Average Drop measures
    the relative decrease in the model score after explanation-based masking,
    Average Gain measures the relative increase with respect to the remaining
    score range up to one.

    The metric is intended for model scores in the interval :math:`[0, 1]`. If
    the model produces logits, ``activation="softmax"`` or ``activation="sigmoid"``
    should be used, or a custom operator returning probability-like scores should
    be provided.

    The implementation is based on the Average Gain metric described by
    Zhang et al. (2024) and follows the implementation provided by Xplique,
    with support for PyTorch models, models exposing ``predict`` or
    ``predict_proba``, and custom scoring operators.

    Zhang, H., Torres, F., Sicre, R., Avrithis, Y., & Ayache, S. (2024).
    Opti-CAM: Optimizing Saliency Maps for Interpretability.
    Computer Vision and Image Understanding, 248, 104101.
    """
    NAME = "AverageGain"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model, ``X_test``,
            selected observations and attribution values. ``y_test`` may be ``None``
            when the model score can be determined without explicit targets.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``batch_size`` : int or None, optional
              Maximum number of observations processed at once. If ``None``, all
              selected observations are processed together. The default value is
              ``64``.

            - ``operator`` : callable or None, optional
              Custom scoring function ``g(model, inputs, targets)`` used to obtain
              one scalar score per observation. The function should return scores
              in the interval [0, 1]. If ``None``, the metric performs inference
              directly using the model and selects the corresponding target score.
              The default value is ``None``.

            - ``activation`` : {None, "sigmoid", "softmax"}, optional
              Activation function applied to model predictions before selecting the
              score. ``None`` leaves predictions unchanged. Since Average Gain
              requires scores in [0, 1], ``"sigmoid"`` or ``"softmax"`` should be
              used when the model returns logits. The default value is ``None``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        When no custom ``operator`` is provided, model inference is performed using,
        in order, a PyTorch ``torch.nn.Module``, ``predict_proba``, ``predict`` or
        direct model invocation.

        For two-dimensional model outputs, targets are interpreted as class indices.
        If no targets are provided, the maximum model output for each observation is
        used.

        For image explanations with shape ``(N, H, W, C)``, attribution values are
        averaged over the last dimension before constructing the mask. For
        time-series explanations with shape ``(N, T)``, the resulting mask is
        broadcast over the feature dimension of inputs shaped ``(N, T, F)``.
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
            raise ValueError("The default AverageGain operator requires predictions shaped (N, C).")
        if targets is None:
            return np.asarray(np.max(prediction, axis=-1), dtype=np.float32)
        if targets.ndim == 1:
            return np.asarray(prediction[np.arange(len(prediction)), targets.astype(int)], dtype=np.float32)
        return np.asarray(np.sum(prediction * targets, axis=-1, dtype=np.float32), dtype=np.float32)
    

    def _score_batched(self, inputs: np.ndarray, targets: np.ndarray | None, batch_size: int) -> np.ndarray:
        """
        Compute model scores in batches.

        Parameters
        ----------
        inputs : np.ndarray
            Input observations.
        targets : np.ndarray or None
            Target class indices, or ``None`` when targets are not required.
        batch_size : int
            Number of observations processed in each batch.

        Returns
        -------
        np.ndarray
            Concatenated scalar scores for all observations.
        """
        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size
            targets_batch = None if targets is None else targets[start:stop]
            scores.append(self._score(inputs[start:stop], targets_batch))

        return np.concatenate(scores)
    

    @staticmethod
    def _perturb_with_mask(inputs: np.ndarray, explanations: np.ndarray) -> np.ndarray:
        """
        Mask inputs according to their attribution values.

        The mask is constructed from the absolute attribution values. For each
        observation, attribution values are min-max normalised to the interval
        [0, 1]. Image explanations with four dimensions are averaged over their
        last dimension before normalisation.

        When necessary, the mask is expanded and broadcast to match the input
        shape. The perturbed input is then obtained by element-wise
        multiplication between the original input and the normalised mask.

        Parameters
        ----------
        inputs : np.ndarray
            Input observations.
        explanations : np.ndarray
            Attribution values associated with the input observations.

        Returns
        -------
        np.ndarray
            Masked inputs with the same shape as ``inputs``.

        Raises
        ------
        ValueError
            If the number of explanations differs from the number of inputs, if
            explanations contain no feature axis, or if the explanation shape
            cannot be broadcast to the input shape.
        """
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
        r"""
        Compute the Average Gain metric.

        The method selects the observations defined in the metric context,
        computes their original model scores and creates perturbed inputs by
        multiplying each observation by a normalised attribution mask. Model
        scores are then recomputed using the perturbed inputs.

        For each observation, Average Gain is computed as:

        .. math::

            \operatorname{AG}_i =
            \frac{
                \max(\mathrm{after}_i - \mathrm{base}_i, 0)
            }{
                1 - \mathrm{base}_i + \varepsilon
            }

        where :math:`\mathrm{base}_i` is the model score for the original input and
        :math:`\mathrm{after}_i` is the score obtained after retaining the input
        according to the explanation mask.

        The numerator measures the positive increase in model score produced by
        explanation-based masking. The denominator normalises this increase by
        the remaining possible increase from the original score to one.

        The ReLU operation in the numerator prevents decreases in the model
        score after masking from producing negative Average Gain values.

        Returns
        -------
        List[float]
            Average Gain score for each evaluated observation. Higher values are
            better. A value of ``0`` indicates that the explanation-based mask
            does not increase the original model score.

        Raises
        ------
        MetricSkipped
            If no observations are selected, or if any original model score lies
            outside the interval [0, 1].
        ValueError
            If ``batch_size`` is not positive, if ``activation`` is not one of
            ``None``, ``"sigmoid"`` or ``"softmax"``, if the number of
            explanations differs from the number of selected observations, or
            if explanation and input shapes are incompatible.

        Notes
        -----
        Average Gain is designed for model scores in the interval [0, 1].
        When the model returns logits, ``activation="softmax"`` or
        ``activation="sigmoid"`` should be used. Alternatively, a custom
        ``operator`` can be provided as long as it returns probability-like
        scores.

        Unlike Average Drop, which normalises a decrease by the original model
        score, Average Gain normalises an increase by ``1 - base_score``. This
        represents the remaining score range available above the original
        prediction.
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
        
        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size

            targets_batch = None if targets is None else targets[start:stop]
            inputs_batch = inputs[start:stop]
            explanations_batch = explanations[start:stop]

            base = self._score_batched(inputs_batch, targets_batch, len(inputs_batch))
            perturbed_inputs = self._perturb_with_mask(inputs_batch, explanations_batch)
            after = self._score_batched(perturbed_inputs, targets_batch, len(inputs_batch))

            if np.any(base < 0.0) or np.any(base > 1.0) or np.any(after < 0.0) or np.any(after > 1.0):
                raise MetricSkipped(
                    "AverageGain skipped: it requires scores in [0, 1]. "
                    "Use activation='softmax', activation='sigmoid', or provide an operator that returns probabilities."
                )

            batch_scores = np.maximum(after - base, 0.0) / (1.0 - base + _EPS)

            scores.append(batch_scores)

        return np.concatenate(scores).astype(float).tolist()