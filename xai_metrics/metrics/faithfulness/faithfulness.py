# XAI_metrics/metrics/faithfulness/faithfulness.py
import numpy as np
import pandas as pd
from aix360.metrics import faithfulness_metric

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping


@register_metric
class Faithfulness(BaseMetric):
    """
    AIX360 Faithfulness metric.

    This metric evaluates whether feature attribution values reflect the
    influence of the corresponding features on the model prediction.

    For each observation, the wrapped AIX360 implementation first determines
    the class predicted by the model. It then replaces each feature
    individually with its corresponding baseline value and obtains the
    probability assigned to the original predicted class. Finally, it computes
    the negative Pearson correlation between the attribution values and the
    probabilities obtained after feature replacement.

    A high positive score indicates that features with larger attribution
    values tend to produce larger decreases in the predicted class probability
    when replaced by their baseline values. Conversely, a low or negative score
    indicates weak or inverse agreement between the assigned importance and the
    influence of the features on the prediction.

    The wrapped AIX360 implementation requires a classification model exposing
    a ``predict_proba`` method.

    The metric is based on the faithfulness criterion proposed by
    Alvarez-Melis and Jaakkola (2018) and implemented in AIX360.
    """
    NAME = "Faithfulness"

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
            values. The model must implement ``predict_proba``.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``base_values`` : array-like or None, optional
              Explicit baseline values used to replace feature values. If
              provided, this takes priority over ``base_func`` and
              ``base_strategy``.

            - ``base_strategy`` : str, optional
              Strategy used to compute baseline values from ``X_test`` when
              ``base_values`` and ``base_func`` are not provided. Supported
              values are ``"mean"``, ``"median"`` and ``"zero"``. The default
              value is ``"mean"``.

            If ``None``, an empty dictionary is used.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Faithfulness metric.

        The method selects the observations defined in the metric context,
        resolves a common baseline vector from the complete test dataset and
        evaluates each selected observation independently using
        :func:`aix360.metrics.faithfulness_metric`.

        For each observation, AIX360 determines the originally predicted class.
        Each feature is then replaced individually with its baseline value, and
        the probability assigned to that class is recorded. The score is the
        negative Pearson correlation between the attribution values and these
        perturbed-input probabilities.

        Returns
        -------
        List[float]
            Faithfulness score for each evaluated observation. Higher positive
            values indicate stronger agreement between feature importance and
            the decrease in predicted class probability caused by replacing
            individual features.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        AttributeError
            If the model does not implement the ``predict_proba`` method
            required by AIX360.

        Notes
        -----
        The underlying Pearson correlation may be undefined and return ``nan``
        when the attribution values or the perturbed-input probabilities have
        zero variance.
        """
        ctx = self.context
        p = self.params

        model = ctx.model
        X_selected = ctx.X_test.loc[ctx.observations]
        base = self._resolve_base(
            X_reference=ctx.X_test,
            base_values=p.get("base_values"),
            base_strategy=p.get("base_strategy", "mean")
        )

        scores = []
        for x_row, coefs in zip(X_selected.values, ctx.attributions):
            score = faithfulness_metric(
                model=model,
                x=np.asarray(x_row, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                base=base,
            )
            scores.append(float(score))

        return scores

    @staticmethod
    def _resolve_base(
        X_reference: pd.DataFrame | np.ndarray,
        base_values: np.ndarray | None = None,
        base_strategy: str = "mean"
    ) -> np.ndarray:
        """
        Resolve the baseline vector used by the Faithfulness metric.

        Explicit baseline values are used when provided. Otherwise, a single
        baseline vector is computed feature-wise from the reference dataset
        using the selected strategy.

        Parameters
        ----------
        X_reference : pandas.DataFrame or numpy.ndarray
            Reference dataset used to compute the baseline when
            ``base_values`` is not provided. Rows represent observations and
            columns represent input features.
        base_values : array-like or None, optional
            Explicit baseline values containing one value per input feature.
            If provided, they are converted to a floating-point NumPy array and
            returned without applying ``base_strategy``.
        base_strategy : str, default="mean"
            Strategy used to compute the baseline from ``X_reference``.
            Supported values are:

            - ``"mean"``: feature-wise mean of the reference dataset.
            - ``"median"``: feature-wise median of the reference dataset.
            - ``"zero"``: vector of zeros with one value per feature.

        Returns
        -------
        numpy.ndarray
            One-dimensional floating-point array containing one baseline value
            per input feature.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        """
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        values = (
            X_reference.values
            if hasattr(X_reference, "values")
            else np.asarray(X_reference, dtype=float)
        )

        if base_strategy == "mean":
            return np.mean(values, axis=0)
        if base_strategy == "median":
            return np.median(values, axis=0)
        if base_strategy == "zero":
            return np.zeros(values.shape[1], dtype=float)

        raise ValueError(f"Unknown base_strategy: {base_strategy}")
