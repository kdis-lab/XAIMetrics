# xai_metrics/metrics/sensitivity/avg_sensitivity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Any, Mapping
from xai_metrics.base.types import ExplainFunc

@register_metric
class AvgSensitivity(BaseMetric):
    """
    Quantus Average Sensitivity metric.

    This metric measures how much an explanation changes, on average, when
    small random perturbations are applied to the corresponding input.

    For each observation, Quantus generates several perturbed versions of the
    input and recomputes their explanations using ``explain_func``. The
    sensitivity associated with each perturbation is computed as the norm of
    the difference between the original and perturbed explanations, divided by
    the norm of the original explanation. The final score is the average of
    these sensitivity values over all sampled perturbations.

    Lower scores indicate that the explanation changes less under small input
    perturbations and is therefore considered more robust. Higher scores
    indicate greater sensitivity to perturbations.

    This wrapper uses the default similarity, norm, normalisation and
    perturbation functions provided by Quantus. By default, Quantus compares
    explanations using their element-wise difference, computes the numerator
    and denominator with the Frobenius norm, and generates perturbations using
    uniform noise.

    The metric is based on Average Sensitivity proposed by Yeh et al. (2019)
    and subsequently discussed by Bhatt et al. (2020), as implemented in
    Quantus.
    """
    NAME = 'AvgSensitivity'

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None
    ):
        """
        Initialize the Average Sensitivity metric.

        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations, attribution values and
            optional device information.
        explain_func : ExplainFunc
            Function used to generate explanations for perturbed inputs. The
            function must be compatible with Quantus explanation functions and
            return a NumPy array containing the generated attributions.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``nr_samples`` : int, optional
              Number of perturbed samples generated for each observation. The
              default value is ``200``.
            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``False``.
            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``False``.
            - ``lower_bound`` : float, optional
              Lower bound of the uniform noise used for perturbations. The default
              value is ``0.2``.
            - ``upper_bound`` : float or None, optional
              Upper bound of the uniform noise used for perturbations. If ``None``,
              Quantus uses its default behaviour. The default value is ``None``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default functions provided by Quantus:

        - element-wise difference as the explanation comparison function;
        - Frobenius norm for the numerator and denominator;
        - the default Quantus normalisation function when
          ``normalise=True``;
        - uniform-noise perturbations.

        Raises
        ------
        ValueError
            If ``explain_func`` is ``None``.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("AvgSensitivity requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func
    
    def run(self):
        """
        Compute the Average Sensitivity metric.

        The method selects the observations defined in the metric context and
        passes their inputs, target labels, original attributions and
        explanation function to :class:`quantus.AvgSensitivity`.

        Quantus repeatedly perturbs each input, recomputes its explanation and
        measures the relative change with respect to the original explanation.
        The score returned for each observation is the average sensitivity
        across the configured number of perturbations.

        If every attribution value is negative, their treatment depends on the
        ``abs`` parameter. When ``abs=True``, their absolute values are passed
        to Quantus. When ``abs=False``, the metric is skipped.

        The model is set to training mode before evaluation, following the
        current behaviour of this wrapper. The device stored in the metric
        context is forwarded to Quantus.

        Returns
        -------
        list[float]
            Average Sensitivity score for each evaluated observation. Lower
            values indicate explanations that are more robust to small random
            input perturbations.

        Raises
        ------
        MetricSkipped
            If every attribution value is negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        nr_samples = int(p.get("nr_samples", 200))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", False))
        lower_bound = float(p.get("lower_bound", 0.2))
        upper_bound = p.get("upper_bound")
        if upper_bound is not None:
            upper_bound = float(upper_bound)

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.train()

        results = quantus.AvgSensitivity(
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions,
            explain_func=self.explain_func,
            device=ctx.device
        )

        return results