# tests/explainers/test_shap.py
import numpy as np
from types import SimpleNamespace

from xai_metrics.explainers.shap import SHAPExplainer
import xai_metrics.explainers.shap as shap_module


def test_shap_selects_target_output_and_returns_attributions(
    monkeypatch,
    xai_context,
    classification_model
):
    calls = {
        "constructor": [],
        "explain": []
    }

    shap_values = np.array([
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        [
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ],
    ])

    class FakeShapExplainer:
        def __init__(
            self,
            predict_fn,
            background,
            algorithm,
            output_names,
            feature_names,
            seed
        ):
            calls['constructor'].append({
                "predict_fn": predict_fn,
                "background": background,
                "algorithm": algorithm,
                "output_names": output_names,
                "feature_names": feature_names,
                "seed": seed
            })

        def __call__(self, inputs, max_evals, batch_size):
            calls['explain'].append({
                "inputs": inputs,
                "max_evals": max_evals,
                "batch_size": batch_size
            })

            return SimpleNamespace(values=shap_values)
        
    monkeypatch.setattr(shap_module.shap, "Explainer", FakeShapExplainer)

    explainer = SHAPExplainer(
        context=xai_context,
        params={
            "mode": "classification",
            "algorithm": "permutation",
            "random_state": 17,
            "max_evals": 100,
            "batch_size": 8
        }
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch,
        targets=np.array([1, 0])
    )

    expected = np.array([
        [10.0, 20.0, 30.0],
        [4.0, 5.0, 6.0]
    ])

    np.testing.assert_allclose(result, expected)

    assert len(calls['constructor']) == 1

    constructor = calls['constructor'][0]

    np.testing.assert_allclose(
        constructor['background'],
        xai_context.X_background.to_numpy()
    )

    assert constructor['algorithm'] == 'permutation'
    assert constructor['feature_names'] == ['x1', 'x2', 'x3']
    assert constructor['seed'] == 17

    assert len(calls['explain']) == 1

    explanation_call = calls['explain'][0]

    np.testing.assert_allclose(
        explanation_call['inputs'],
        xai_context.X_batch.to_numpy()
    )

    assert explanation_call['max_evals'] == 100
    assert explanation_call['batch_size'] == 8


def test_shap_applies_absolute_value(
    monkeypatch,
    xai_context,
    classification_model
):
    class FakeShapExplainer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, inputs, **kwargs):
            return SimpleNamespace(
                values=np.array([
                    [-1.0, 2.0, -3.0],
                    [-4.0, 5.0, -6.0]
                ])
            )

    monkeypatch.setattr(shap_module.shap, "Explainer", FakeShapExplainer)

    explainer = SHAPExplainer(
        context=xai_context,
        params={"abs": True}
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch
    )

    np.testing.assert_allclose(
        result,
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    )
