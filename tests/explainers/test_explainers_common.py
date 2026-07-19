# tests/explainers/test_explainers_common.py
import pytest
import pandas as pd
import numpy as np

from types import SimpleNamespace

from xai_metrics.explainers.lime import LIMEExplainer
from xai_metrics.explainers.shap import SHAPExplainer
from xai_metrics.explainers.breakdown import BreakDownExplainer
from xai_metrics.explainers.maple import MAPLEExplainer

import xai_metrics.explainers.lime as lime_module


@pytest.mark.parametrize(
    "explainer_class",
    [
        LIMEExplainer,
        SHAPExplainer,
        BreakDownExplainer,
        MAPLEExplainer
    ]
)
def test_explainers_reject_model_without_prediction_method(
    monkeypatch,
    xai_context,
    explainer_class
):
    if explainer_class is LIMEExplainer:
        monkeypatch.setattr(
            lime_module,
            "LimeTabularExplainer",
            lambda **kwargs: SimpleNamespace()
        )

    explainer = explainer_class(
        context=xai_context,
        params={
            "mode": "classification",
            "data_type": "time_series"
        }
    )

    class ModelWithoutPredict:
        pass

    with pytest.raises(AttributeError, match="predict_proba"):
        explainer._get_predict_fn(ModelWithoutPredict())


@pytest.mark.parametrize(
    "explainer_class",
    [
        LIMEExplainer,
        SHAPExplainer,
        MAPLEExplainer
    ]
)
def test_numpy_conversion_preserves_feature_order(
    monkeypatch,
    xai_context,
    explainer_class
):
    if explainer_class is LIMEExplainer:
        monkeypatch.setattr(
            lime_module,
            "LimeTabularExplainer",
            lambda **kwargs: SimpleNamespace()
        )

    explainer = explainer_class(
        context=xai_context,
        params={"data_type": "time_series"}
    )

    # Las columnas se proporcionan en orden inverso.
    inputs = pd.DataFrame({
        "x3": [3.0],
        "x2": [2.0],
        "x1": [1.0]
    })

    result = explainer._to_numpy(inputs)

    np.testing.assert_allclose(
        result,
        [[1.0, 2.0, 3.0]]
    )


def test_breakdown_dataframe_conversion_preserves_feature_order(xai_context):
    explainer = BreakDownExplainer(xai_context)

    inputs = pd.DataFrame({
        "x3": [3.0],
        "x2": [2.0],
        "x1": [1.0]
    })

    result = explainer._to_dataframe(inputs)

    assert list(result.columns) == ["x1", "x2", "x3"]
    np.testing.assert_allclose(
        result.to_numpy(),
        [[1.0, 2.0, 3.0]]
    )