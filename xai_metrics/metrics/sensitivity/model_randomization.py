# xai_metrics\metrics\sensitivity\random_logit.py
import copy
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc

from typing import Mapping, Any

@register_metric
class ModelRandomization(BaseMetric):
    """
    Model Randomization sensitivity metric.

    This metric evaluates whether an explanation depends on the parameters
    learned by the model. A copy of the original model is created, a fraction
    of its parameterised layers is randomized, and explanations are recomputed
    using the randomized model.

    For each observation ``i``, the metric compares the original explanation
    with the explanation produced by the randomized model:

        MR_i = rho(phi(f, x_i), phi(f_random, x_i))

    where ``f`` is the original model, ``f_random`` is the randomized model,
    ``phi`` is the explanation function and ``rho`` is the Spearman rank
    correlation.

    Explanations that remain highly correlated after model parameters have been
    randomized may indicate that the explanation method is insufficiently
    dependent on the learned model. Conversely, a low correlation indicates
    greater sensitivity to the model parameters.

    The metric returns one Spearman correlation value per observation. Values
    normally lie in ``[-1, 1]``. Values close to ``1`` indicate very similar
    attribution rankings, values close to ``0`` indicate weak rank association,
    and negative values indicate reversed attribution rankings.

    Model parameters are randomized progressively over a configurable fraction
    of parameterised layers. By default, layers are processed in reverse module
    order and the last 25 percent of parameterised layers are randomized.
    Parameter values are independently sampled from a uniform distribution over
    ``[0, 1]``.

    The original model stored in the metric context is never modified. A deep
    copy is created before randomization.

    The implementation is based on the Model Randomization sanity check proposed
    by Adebayo et al. (2018) and follows the implementation provided by Xplique,
    adapted to PyTorch models and per-observation outputs.

    Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B.
    (2018). Sanity Checks for Saliency Maps.
    Advances in Neural Information Processing Systems, 31.
    """
    NAME = "ModelRandomization"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain a PyTorch model,
            ``X_test``, selected observations and attribution values. ``y_test``
            may be ``None`` when the explanation function does not require
            explicit targets.
        explain_func : ExplainFunc
            Explanation function used to recompute attribution values with the
            randomized model. It must accept ``(model, inputs, targets)`` and
            return one explanation per input observation.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``batch_size`` : int or None, optional
              Maximum number of observations whose explanations are recomputed
              at once. If ``None``, all selected observations are processed
              together. The default value is ``64``.

            - ``fraction`` : float, optional
              Fraction of parameterised model layers to randomize. Must belong
              to the interval ``[0, 1]``. The number of randomized layers is
              computed as ``int(number_of_layers * fraction)``. The default
              value is ``0.25``.

            - ``reverse`` : bool, optional
              Whether to reverse the order of parameterised layers before
              selecting those to randomize. If ``True``, randomization starts
              from the last parameterised layers encountered in the model.
              The default value is ``True``.

            - ``random_state`` : int or None, optional
              Random seed used to generate randomized parameter values. The
              default value is ``42``.

            If ``None``, an empty dictionary is used.

        Raises
        ------
        ValueError
            If ``explain_func`` is ``None``.

        Notes
        -----
        This implementation currently supports PyTorch models only.

        The original attribution values stored in the metric context are used
        as reference explanations. Only explanations for the randomized model
        are recomputed using ``explain_func``.

        Randomization is applied to a deep copy of the model, so the model
        stored in the metric context is not modified.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("ModelRandomization requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func


    @staticmethod
    def _spearman(original: np.ndarray, randomized: np.ndarray) -> float:
        """
        Compute the Spearman correlation between two explanations.

        The original and randomized explanations must have identical shapes.
        Three-dimensional explanation maps are averaged over their last
        dimension before the correlation is computed. The resulting arrays are
        flattened and compared using Spearman rank correlation.

        Parameters
        ----------
        original : np.ndarray
            Explanation produced by the original model.
        randomized : np.ndarray
            Explanation produced by the randomized model.

        Returns
        -------
        float
            Spearman rank correlation between the explanations. Values normally
            lie in ``[-1, 1]``. If the correlation is undefined, for example
            because one explanation is constant, ``0.0`` is returned.

        Raises
        ------
        ValueError
            If the two explanations do not have the same shape.

        Notes
        -----
        For image attribution maps represented as ``(H, W, C)``, the channel
        dimension is averaged before computing the correlation.

        The returned value is the signed Spearman correlation. Therefore,
        negative values indicate an inversion of the attribution ranking rather
        than an absence of association.
        """
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
        """
        Compute the Model Randomization metric.

        The method first creates a deep copy of the PyTorch model contained in
        the metric context. Parameterised layers are collected and optionally
        traversed in reverse order. A configurable fraction of these layers is
        then randomized by replacing their parameter values with samples drawn
        independently from a uniform distribution over ``[0, 1]``.

        The explanation function is evaluated using the randomized model for all
        selected observations. Each randomized explanation is then compared with
        its corresponding original explanation using Spearman rank correlation.

        Explanations that strongly depend on the learned model parameters are
        expected to change after randomization and therefore produce low
        correlations. High positive correlations indicate that the attribution
        ranking remains similar despite model randomization.

        Returns
        -------
        List[float]
            Spearman correlation between the original and randomized explanation
            for each evaluated observation. Values normally lie in ``[-1, 1]``.

            Values close to ``1`` indicate highly similar attribution rankings,
            values close to ``0`` indicate weak rank association, and negative
            values indicate reversed rankings.

        Raises
        ------
        MetricSkipped
            If the model is not a PyTorch ``torch.nn.Module``, if no observations
            are selected, or if the model contains no parameterised layers.
        ValueError
            If the number of original explanations differs from the number of
            selected observations, if ``batch_size`` is not positive, if
            ``fraction`` does not belong to ``[0, 1]``, or if ``explain_func``
            returns explanations whose shape differs from the original
            attribution shape.

        Notes
        -----
        ``fraction`` controls the proportion of parameterised layers that are
        randomized. The actual number is computed using integer truncation:

            int(number_of_parameterised_layers * fraction)

        Consequently, small positive fractions may result in zero randomized
        layers when the model contains only a few parameterised layers.

        When ``reverse=True``, the order of the collected layers is reversed
        before selecting the layers to randomize. This approximates progressive
        randomization beginning from the output side of sequential models.

        The original model is never modified because randomization is performed
        on a deep copy.

        Unlike the original Xplique ``evaluate`` method, this implementation
        returns one correlation value per observation rather than automatically
        averaging them into a single dataset-level score.
        """
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