# tests/explainers/test_lime.py
import numpy as np

from xai_metrics.explainers.lime import LIMEExplainer
import xai_metrics.explainers.lime as lime_module

def test_lime_forwards_inputs_parameters_and_returns_ordered_attributions(
    monkeypatch,
    xai_context,
    classification_model
):
    calls = {
        "constructor": None,
        "explain": []
    }

    class FakeLimeExplanation:
        def __init__(self, label, weights):
            self.label = label
            self.weights = weights
            self.top_labels = [label]

        def as_map(self):
            return {self.label: self.weights}
        
    class FakeLimeTabularExplainer:
        def __init__(self, **kwargs):
            calls['constructor'] = kwargs
            self.current_observation = 0

        def explain_instance(self, **kwargs):
            calls['explain'].append(kwargs)

            label = kwargs['labels'][0]

            if self.current_observation == 0:
                weights = [
                    (2, 0.7),
                    (0, -0.2),
                ]
            else:
                weights = [
                    (1, 0.5),
                    (2, -0.1),
                ]

            self.current_observation += 1

            return FakeLimeExplanation(label, weights)
        
    monkeypatch.setattr(lime_module, "LimeTabularExplainer", FakeLimeTabularExplainer)

    explainer = LIMEExplainer(
        context=xai_context,
        params={
            "mode": "classification",
            "num_features": 2,
            "num_samples": 50,
            "distance_metric": "manhattan",
            "random_state": 7,
            "discretize_continuous": False,
        },
    )

    result = explainer.explain(
        model=classification_model,
        inputs=xai_context.X_batch,
        targets=np.array([1, 0])
    )

    expected = np.array([
        [-0.2, 0.0, 0.7],
        [0.0, 0.5, -0.1],
    ])

    np.testing.assert_allclose(result, expected)

    constructor = calls['constructor']

    np.testing.assert_allclose(
        constructor['training_data'],
        xai_context.X_background.to_numpy()
    )
    np.testing.assert_array_equal(
        constructor['training_labels'],
        xai_context.y_background.to_numpy()
    )

    assert constructor['mode'] == 'classification'
    assert constructor['feature_names'] == ['x1', 'x2', 'x3']
    assert constructor['random_state'] == 7
    assert constructor['discretize_continuous'] is False

    assert len(calls['explain']) == 2

    first_call = calls['explain'][0]
    second_call = calls['explain'][1]

    np.testing.assert_allclose(
        first_call['data_row'],
        [10.0, 11.0, 12.0]
    )
    np.testing.assert_allclose(
        second_call['data_row'],
        [20.0, 21.0, 22.0]
    )

    assert first_call['labels'] == (1,)
    assert second_call['labels'] == (0,)
    assert first_call['num_features'] == 2
    assert first_call['num_samples'] == 50
    assert first_call['distance_metric'] == 'manhattan'

    predictions = first_call['predict_fn'](
        np.array([[0.1, 0.2, 0.3]])
    )
    assert predictions.shape == (1, 2)