# xai_metrics/metrics/faithfulness/monotonicity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Monotonicity(BaseMetric):
    """
    Quantus Monotonicity metric.

    This metric evaluates whether the target model output evolves
    monotonically as feature groups are progressively introduced according to
    their attribution values.

    For each observation, features are ordered by increasing attribution value.
    Quantus starts from an input defined by the selected baseline and processes
    the features in groups of size ``features_in_step``. After each group is
    introduced, the model output associated with the target label is evaluated.
    The explanation satisfies the monotonicity criterion when the resulting
    sequence of target outputs is monotonically non-decreasing.

    The metric returns one Boolean value per observation. ``True`` indicates
    that introducing successive feature groups never decreases the target
    model output, whereas ``False`` indicates that the monotonicity condition
    is violated at least once.

    The metric is based on the Monotonicity metric proposed by Arya et al.
    (2019) and the monotonic attribute functions described by Luss et al.
    (2019), as implemented in Quantus.
    """
    NAME = 'Monotonicity'

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

            - ``features_in_step`` : int, optional
              Number of features added at each perturbation step. The default
              value is ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used to initialise the perturbed input. Supported
              values depend on the Quantus perturbation function. Common values
              are ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses the default normalisation and perturbation functions
        provided by Quantus.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Monotonicity metric.

        The method selects the observations defined in the metric context and
        passes their input data, target labels and attribution values to
        :class:`quantus.Monotonicity`.

        For each observation, Quantus orders the features by increasing
        attribution value and processes them in groups of size
        ``features_in_step`` starting from a baseline input. The target model
        output is evaluated after each step. The metric returns ``True`` when
        the resulting sequence is monotonically non-decreasing.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[bool]
            Monotonicity result for each evaluated observation. ``True``
            indicates that the target model output is monotonically
            non-decreasing across the feature-processing steps.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "black"))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.Monotonicity(
            features_in_step=features_in_step,
            abs=abs_,
            normalise=normalise,
            perturb_baseline=perturb_baseline
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results
