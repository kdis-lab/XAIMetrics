# XAI_metrics/metrics/faithfulness/faithfulness.py
import numpy as np
import pandas as pd
from aix360.metrics import faithfulness_metric

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping, Callable, Dict


@register_metric
class Faithfulness(BaseMetric):
    """
    AIX360 Faithfulness metric.

    This metric evaluates the relationship between feature attribution values
    and the effect that replacing each feature with its baseline value has on
    the model output.

    For each observation, the wrapped AIX360 implementation determines the
    predicted class and replaces each feature individually with its
    corresponding baseline value. It then computes the correlation between the
    attribution values and the predicted probabilities obtained after those
    replacements.

    Higher scores indicate stronger agreement between the importance assigned
    to the features and their influence on the model prediction.

    The wrapped AIX360 implementation requires a model exposing a
    ``predict_proba`` method.

    The metric is based on the faithfulness metric proposed by Alvarez-Melis
    and Jaakkola (2018) and implemented in AIX360.
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
        resolves a baseline vector and evaluates each observation independently
        using :func:`aix360.metrics.faithfulness_metric`.

        For every feature, AIX360 replaces only that feature with its baseline
        value and obtains the probability assigned to the original predicted
        class. The final score is the negative Pearson correlation between the
        attribution values and those probabilities.

        Returns
        -------
        List[float]
            Faithfulness score for each evaluated observation. Higher values
            indicate that features with greater attribution values have a
            stronger effect on the predicted class probability when replaced
            by their baseline values.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        AttributeError
            If the model does not implement the ``predict_proba`` method
            required by AIX360.
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
        base_strategy: str = "mean",
        base_func: Callable[..., Any] | None = None,
        base_func_kwargs: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Resolve the baseline values used by the Faithfulness metric.

        The baseline can be provided directly through ``base_values``, computed
        with a custom ``base_func``, or derived from the reference dataset using
        one of the supported baseline strategies.

        Parameters
        ----------
        X_reference : pandas.DataFrame or numpy.ndarray
            Reference dataset used to compute baseline values when
            ``base_values`` and ``base_func`` are not provided.
        base_values : numpy.ndarray or None, optional
            Explicit baseline values. If provided, these values are returned as
            a NumPy array and no strategy is applied.
        base_strategy : str, default="mean"
            Strategy used to compute baseline values from ``X_reference``.
            Supported values are ``"mean"``, ``"median"`` and ``"zero"``.
        base_func : Callable[..., Any] or None, optional
            Custom function used to compute baseline values from the reference
            dataset. The function must accept ``X_reference`` as its first argument,
            where ``X_reference`` is the test data, usually a ``pandas.DataFrame``.
            It may also accept additional keyword arguments provided through
            ``base_func_kwargs``. The returned value must be array-like and convertible
            to ``numpy.ndarray`` with one value per feature.
        base_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``base_func``. If ``None``, an empty
            dictionary is used.

        Returns
        -------
        numpy.ndarray
            Baseline values used to replace feature values during metric
            computation.

        Raises
        ------
        ValueError
            If ``base_strategy`` is unknown.
        """
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        if base_func is not None:
            kwargs = base_func_kwargs or {}
            return np.asarray(base_func(X_reference, **kwargs), dtype=float)

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
