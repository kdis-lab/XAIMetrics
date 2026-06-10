# XAI_metrics/metrics/faithfulness/sensitivity_n.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class SensitivityN(BaseMetric):
    """
    Quantus Sensitivity-N metric.

    This metric evaluates whether the attribution values assigned to groups of
    features agree with the variation in the target model output produced when
    those features are perturbed.

    For each observation, features are ordered by decreasing attribution value
    and progressively perturbed in groups of size ``features_in_step``. At each
    perturbation step, Quantus records the change in the target output and the
    attribution values associated with the processed feature group. Pearson
    correlation is then used to measure the agreement between both quantities
    for the evaluated perturbation steps.

    The number of evaluated steps is limited by ``n_max_percentage``, which
    determines the maximum proportion of input features considered by the
    experiment. Higher correlation values indicate stronger agreement between
    the attribution values and the effect of the corresponding perturbations
    on the model output.

    The metric is based on the Sensitivity-N test proposed by Ancona et al.
    (2018) and implemented in Quantus.
    """
    NAME = 'SensitivityN'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model, ``X_test``,
            ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``n_max_percentage`` : float, optional
              Maximum percentage of features to evaluate. The default value is
              ``0.8``.

            - ``features_in_step`` : int, optional
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is
              ``False``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses the default Pearson correlation, attribution
        normalisation and baseline-replacement functions provided by Quantus.

        Quantus enables result aggregation by default for this metric.
        Therefore, the returned result may contain an aggregate of the
        correlations obtained for the evaluated perturbation steps rather than
        one independent score per input observation.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Sensitivity-N metric.

        The method selects the observations defined in the metric context and
        passes their input data, target labels and attribution values to
        :class:`quantus.SensitivityN`.

        Quantus orders the features by decreasing attribution value and
        progressively perturbs them in groups of size ``features_in_step``. It
        compares the changes in the target model output with the corresponding
        attribution values using Pearson correlation. Only the perturbation
        steps covered by ``n_max_percentage`` are included.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[float]
            Sensitivity-N result returned by Quantus. Higher values indicate
            stronger agreement between attribution values and target-output
            changes caused by perturbing the corresponding features. With the
            default Quantus configuration, the correlations obtained across
            the evaluated perturbation steps are aggregated.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        n_max_percentage = float(p.get("n_max_percentage", 0.8))
        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", False))
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

        results = quantus.SensitivityN(
            n_max_percentage=n_max_percentage,
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
