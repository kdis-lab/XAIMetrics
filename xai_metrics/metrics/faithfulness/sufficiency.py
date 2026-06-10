# XAI_metrics/metrics/faithfulness/sufficiency.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Sufficiency(BaseMetric):
    """
    Quantus Sufficiency metric.

    This metric evaluates whether similar explanations are associated with the
    same model prediction. For the complete set of evaluated observations,
    Quantus flattens the attribution vectors and computes the pairwise distances
    between them. The resulting distance matrix is normalised, and two
    explanations are considered similar when their distance is less than or
    equal to ``threshold``.

    For each observation, the metric computes the proportion of other
    observations with similar explanations that receive the same predicted
    class. The observation itself is excluded from this comparison. If no other
    explanation lies within the specified distance threshold, the score for
    that observation is ``0.0``.

    Higher scores indicate that observations with similar explanations more
    frequently receive the same model prediction. Because explanations are
    compared within the evaluated batch, the resulting scores depend on the
    observations included in the metric context.

    The metric is based on the Sufficiency test proposed by Dasgupta et al.
    (2022) and implemented in Quantus.
    """
    NAME = 'Sufficiency'

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
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``threshold`` : float, optional
              Maximum distance between two attribution vectors for their
              explanations to be considered similar. The default value is
              ``0.6``.

            - ``distance_func`` : str, optional
              Distance function used to compare attribution vectors. The value
              is passed to Quantus and is typically a valid SciPy distance name,
              such as ``"seuclidean"``. The default value is ``"seuclidean"``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is
              ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses the default attribution normalisation function
        provided by Quantus.

        The metric compares explanations only among the observations evaluated
        in the same call. Consequently, changing the selected observations can
        change the neighbourhood of each explanation and therefore its score.
        """
        super().__init__(context, params)
        

    def run(self):
        """
        Compute the Sufficiency metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.Sufficiency`.

        Quantus computes the pairwise distances between the flattened
        attribution vectors, normalises the distance matrix and identifies
        explanation pairs whose distance is less than or equal to
        ``threshold``. For each observation, it then calculates the fraction of
        similar explanations whose observations receive the same predicted
        class. Self-comparisons are excluded.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        list[float]
            Sufficiency score for each evaluated observation. Scores lie
            between ``0.0`` and ``1.0``. Higher values indicate that
            observations with similar explanations more frequently receive
            the same predicted class. A score of ``0.0`` is returned when no
            other explanation satisfies the distance threshold.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        threshold = float(p.get("threshold", 0.6))
        distance_func = str(p.get("distance_func", "seuclidean"))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.Sufficiency(
            threshold=threshold,
            distance_func=distance_func,
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results
