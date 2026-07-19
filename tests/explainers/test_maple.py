# tests/explainers/test_maple.py
import numpy as np
import pandas as pd
import pytest

from xai_metrics.base import ExplainerContext

from xai_metrics.explainers.maple import MAPLEExplainer
import xai_metrics.explainers.maple as maple_module


def test_maple_forwards_background_model_response_and_returns_attributions(
    monkeypatch,
    xai_context,
    classification_model
):
    calls = {
        "constructor": [],
        "explain": []
    }

    class FakeMAPLEModel:
        def __init__(self, **kwargs):
            calls['constructor'].append(kwargs)

        def explain(self, row):
            calls['explain'].append(np.asarray(row))

            return np.array([-row[0], row[1], -row[2]])

    monkeypatch.setattr(maple_module, "_MAPLEModel", FakeMAPLEModel)

    explainer = MAPLEExplainer(
        context=xai_context,
        params={
            "data_type": "time_series",
            "validation_size": 0.4,
            "mode": "classification",
            "output_index": 1,
            "fe_type": "rf",
            "n_estimators": 10,
            "max_features": 1.0,
            "min_samples_leaf": 2,
            "regularization": 0.01,
            "random_state": 23,
            "n_jobs": 1,
            "abs": True,
        }
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch
    )

    expected = np.array([
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0]
    ])

    np.testing.assert_allclose(result, expected)

    assert len(calls['constructor']) == 1

    constructor = calls['constructor'][0]

    np.testing.assert_allclose(
        constructor['X_train'],
        xai_context.X_background.iloc[:3].to_numpy()
    )
    np.testing.assert_allclose(
        constructor['X_val'],
        xai_context.X_background.iloc[3:].to_numpy()
    )

    expected_y_train = classification_model.predict_proba(
        xai_context.X_background.iloc[:3].to_numpy()
    )[:, 1]

    expected_y_val = classification_model.predict_proba(
        xai_context.X_background.iloc[3:].to_numpy()
    )[:, 1]

    np.testing.assert_allclose(
        constructor['y_train'],
        expected_y_train
    )
    np.testing.assert_allclose(
        constructor['y_val'],
        expected_y_val
    )

    assert constructor['fe_type'] == 'rf'
    assert constructor['n_estimators'] == 10
    assert constructor['max_features'] == 1.0
    assert constructor['min_samples_leaf'] == 2
    assert constructor['regularization'] == 0.01
    assert constructor['random_state'] == 23
    assert constructor['n_jobs'] == 1

    assert len(calls['explain']) == 2

    np.testing.assert_allclose(
        calls['explain'][0],
        [10.0, 11.0, 12.0]
    )
    np.testing.assert_allclose(
        calls['explain'][1],
        [20.0, 21.0, 22.0]
    )


def test_maple_requires_at_least_three_background_observations():
    context = ExplainerContext(
        X_background=pd.DataFrame({
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
        })
    )

    with pytest.raises(
        ValueError,
        match="requires at least three background observations",
    ):
        MAPLEExplainer(context)