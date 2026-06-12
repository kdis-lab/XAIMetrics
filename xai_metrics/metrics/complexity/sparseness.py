# xai_metrics/metrics/complexity/sparseness.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any

@register_metric
class Sparseness(BaseMetric):
    """
    Quantus Sparseness metric.

    This metric measures how concentrated the attribution magnitude is across
    the input features. Quantus applies the absolute-value operation to the
    attribution values and computes the Gini index of the resulting attribution
    vector.

    Explanations that assign most of their attribution magnitude to a small
    subset of features obtain higher scores and are considered sparser.
    Conversely, explanations that distribute attribution magnitude more
    uniformly across the features obtain lower scores.

    A score close to ``0`` indicates that attribution magnitude is distributed
    relatively uniformly across the features. Higher values indicate greater
    inequality in the attribution distribution and, therefore, a sparser
    explanation.

    The metric is based on the Sparseness metric proposed by Chalasani et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Sparseness'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution
            values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses the default normalisation function provided by
        Quantus when ``normalise=True``.

        Quantus applies the absolute-value operation to the attribution values
        before computing the metric. This behaviour is fixed in this wrapper
        because the ``abs`` parameter is not exposed through ``params``.
        """
        super().__init__(context, params)

    def run(self):
        """
        Compute the Sparseness metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.Sparseness`.

        Quantus flattens each attribution vector, applies the absolute-value
        operation, sorts the resulting attribution magnitudes and computes
        their Gini index. The calculation depends only on the attribution
        values; the input data, labels and model are passed as part of the
        standard Quantus metric interface.

        If all attribution values are negative, this wrapper converts them to
        their absolute values before calling Quantus. Quantus also applies its
        own absolute-value preprocessing, including when the attribution array
        contains a mixture of positive and negative values.

        The model is set to training mode before the metric is evaluated,
        following the current implementation of this wrapper. However, the
        Sparseness calculation itself does not use model predictions.

        Returns
        -------
        List[float]
            Sparseness score for each evaluated observation. Higher values
            indicate that attribution magnitude is concentrated on fewer
            features, while lower values indicate a more uniform attribution
            distribution.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Sparseness(
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        return results