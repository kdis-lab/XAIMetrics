# xai_metrics/metrics/faithfulness/monotonicity_correlation.py
import quantus
import numpy as np
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class MonotonicityCorrelation(BaseMetric):
    """
    Quantus Monotonicity Correlation metric.

    This metric evaluates whether feature attribution values are monotonically
    related to the uncertainty produced in the target model output when the
    corresponding features are perturbed.

    For each observation, features are ordered by increasing attribution value
    and divided into groups. Each group is perturbed repeatedly, and its output
    uncertainty is estimated from the mean squared change in the target model
    output relative to the original output magnitude. The metric then computes
    the Spearman correlation between the summed attribution values of the
    feature groups and their corresponding uncertainty estimates.

    This wrapper uses a safe Spearman correlation implementation that returns
    ``0.0`` when either of the compared vectors has zero variance, avoiding
    undefined correlation values.

    Higher scores indicate a stronger positive monotonic relationship between
    feature importance and the uncertainty caused by perturbing those features.

    The metric is based on the Monotonicity Correlation metric proposed by
    Nguyen and Rodríguez Martínez (2020) and implemented in Quantus.
    """
    NAME = 'MonotonicityCorrelation'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``eps`` : float, optional
              Threshold used when computing the inverse prediction factor. The
              default value is ``1e-5``.

            - ``nr_samples`` : int, optional
              Number of perturbation samples generated for each feature group.
              The default value is ``100``.

            - ``features_in_step`` : int, optional
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"uniform"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses :meth:`_safe_spearman` as the similarity function.
        The default normalisation and perturbation functions provided by
        Quantus are used.
        """
        super().__init__(context, params)


    def _safe_spearman(
        self,
        a: Any,
        b: Any,
        batched: bool = False,
        **kwargs: Any
    ) -> float | np.ndarray:
        """
        Compute Spearman correlation safely.

        This helper avoids undefined Spearman correlations when one of the input
        vectors has zero variance. In that case, it returns ``0.0`` instead of
        ``nan``.

        Parameters
        ----------
        a : Any
            First input array or batch of arrays.
        b : Any
            Second input array or batch of arrays.
        batched : bool, default=False
            Whether ``a`` and ``b`` contain batches of vectors. If ``True``, the
            Spearman correlation is computed independently for each pair of
            vectors.
        **kwargs : Any
            Additional keyword arguments. These are accepted for compatibility
            with Quantus similarity functions and are not used.

        Returns
        -------
        float or numpy.ndarray
            Spearman correlation score. If ``batched=False``, a single float is
            returned. If ``batched=True``, a NumPy array with one score per input
            pair is returned.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if batched:
            scores = []
            for ai, bi in zip(a, b):
                if np.std(ai) == 0 or np.std(bi) == 0:
                    scores.append(0.0)
                else:
                    scores.append(spearmanr(ai, bi).correlation)
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return spearmanr(a, b).correlation

    def run(self):
        """
        Compute the Monotonicity Correlation metric.

        The method selects the observations defined in the metric context and
        passes their input data, target labels and attribution values to
        :class:`quantus.MonotonicityCorrelation`.

        For each observation, Quantus orders features by increasing attribution
        value and perturbs each feature group ``nr_samples`` times. It estimates
        the uncertainty associated with each group as the mean squared change
        in the target model output, scaled relative to the magnitude of the
        original output. The internal safe Spearman function then compares
        these uncertainty estimates with the corresponding attribution sums.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[float]
            Monotonicity Correlation score for each evaluated observation.
            Higher values indicate a stronger positive monotonic relationship
            between the attribution values and the uncertainty caused by
            perturbing the corresponding feature groups.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        eps = float(p.get("eps", 1e-5))
        nr_samples = int(p.get("nr_samples", 100))
        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "uniform"))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.MonotonicityCorrelation(
            similarity_func=self._safe_spearman,
            eps=eps,
            nr_samples=nr_samples,
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
