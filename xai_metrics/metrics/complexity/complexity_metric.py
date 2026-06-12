# xai_metrics/metrics/complexity/complexity_metric.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any

@register_metric
class Complexity(BaseMetric):
    """
    Quantus Complexity metric.

    This metric measures the complexity of an explanation as the entropy of
    the relative contribution of each feature to the total attribution
    magnitude. Before computing the entropy, Quantus applies the absolute-value
    operation to the attribution values and represents each feature attribution
    as a fraction of the total attribution magnitude.

    Explanations whose importance is distributed across many features tend to
    produce higher entropy values and are therefore considered more complex.
    Conversely, explanations that concentrate most of their importance on a
    small number of features tend to produce lower entropy values and are
    considered easier to interpret.

    For an explanation with ``n`` features, the maximum entropy is
    approximately ``log(n)`` when the attribution magnitude is distributed
    uniformly across all features. The scores returned by this wrapper are not
    divided by this maximum value.

    The metric is based on the Complexity metric proposed by Bhatt et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Complexity'

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
        Compute the Complexity metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.Complexity`.

        Quantus flattens each attribution vector, applies the absolute-value
        operation and converts each feature attribution into its fractional
        contribution to the total attribution magnitude. The complexity score
        is then computed as the entropy of this distribution.

        If all attribution values are negative, this wrapper converts them to
        their absolute values before calling Quantus. Quantus also applies its
        own absolute-value preprocessing to the attribution values.

        The model is set to training mode before the metric is evaluated,
        following the current implementation of this wrapper. However, the
        Complexity calculation itself depends only on the attribution values
        and does not use model predictions.

        Returns
        -------
        List[float]
            Complexity score for each evaluated observation. Lower values
            indicate that attribution importance is concentrated on fewer
            features, while higher values indicate that importance is more
            widely distributed.

            The scores are not normalised by the theoretical maximum entropy
            ``log(n)``, where ``n`` is the number of features.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Complexity(
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        # Normalización
        # n_features = attributions.shape[1]
        # max_entropy = np.log(n_features)
        # results = np.array(results) / max_entropy

        return results