# tests/explainers/test_breakdown.py
import pandas as pd
import numpy as np

from types import SimpleNamespace

from xai_metrics.explainers.breakdown import BreakDownExplainer
import xai_metrics.explainers.breakdown as breakdown_module


def test_breakdown_converts_dalex_result_to_ordered_attributions(
    monkeypatch,
    xai_context,
    classification_model
):
    calls = {
        "constructor": [],
        "predict_parts": []
    }

    results = [
        pd.DataFrame({
            "variable_name": [
                "_baseline_",
                "x3 = 12.0",
                "x1 = 10.0",
                "prediction"
            ],
            "contribution": [0.5, 0.7, -0.2, 1.0]
        }),
        pd.DataFrame({
            "variable_name": [
                "_baseline_",
                "x2 = 21.0",
                "x3 = 22.0",
                "prediction"
            ],
            "contribution": [0.5, 0.4, -0.1, 0.8]
        })
    ]

    class FakeDalexExplainer:
        def __init__(self, **kwargs):
            calls['constructor'].append(kwargs)
            self.result_index = 0

        def predict_parts(self, **kwargs):
            calls['predict_parts'].append(kwargs)

            result = results[self.result_index]
            self.result_index += 1

            return SimpleNamespace(result=result)

    monkeypatch.setattr(breakdown_module.dx, "Explainer", FakeDalexExplainer)

    explainer = BreakDownExplainer(
        context=xai_context,
        params={
            "mode": "classification",
            "output_index": 1,
            "label": "dummy-model",
            "verbose": False,
            "precalculate": False,
            "n_samples": 25,
            "keep_distributions": True,
            "random_state": 11
        }
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch
    )

    expected = np.array([
        [-0.2, 0.0, 0.7],
        [0.0, 0.4, -0.1]
    ])

    np.testing.assert_allclose(result, expected)

    assert len(calls['constructor']) == 1

    constructor = calls['constructor'][0]

    assert constructor['model'] is classification_model
    assert constructor['data'] is xai_context.X_background
    assert constructor['y'] is xai_context.y_background
    assert constructor['label'] == 'dummy-model'
    assert constructor['verbose'] is False
    assert constructor['precalculate'] is False
    assert constructor['model_type'] == 'classification'

    prediction = constructor['predict_function'](
        classification_model,
        np.array([[0.1, 0.2, 0.3]])
    )

    assert prediction.shape == (1,)

    assert len(calls['predict_parts']) == 2

    first_call = calls['predict_parts'][0]

    assert first_call['type'] == 'break_down'
    assert first_call['N'] == 25
    assert first_call['keep_distributions'] is True
    assert first_call['random_state'] == 11

    np.testing.assert_allclose(
        first_call['new_observation'].to_numpy(),
        [[10.0, 11.0, 12.0]]
    )


def test_breakdown_applies_absolute_value(
    monkeypatch,
    xai_context,
    classification_model
):
    class FakeDalexExplainer:
        def __init__(self, **kwargs):
            pass

        def predict_parts(self, **kwargs):
            return SimpleNamespace(
                result=pd.DataFrame({
                    "variable_name": ["x1", "x2", "x3"],
                    "contribution": [-1.0, 2.0, -3.0],
                })
            )

    monkeypatch.setattr(breakdown_module.dx, "Explainer", FakeDalexExplainer)

    explainer = BreakDownExplainer(
        context=xai_context,
        params={"abs": True}
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch.iloc[[0]]
    )

    np.testing.assert_allclose(
        result,
        [[1.0, 2.0, 3.0]]
    )