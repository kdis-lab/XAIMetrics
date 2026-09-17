# tests/metrics_tests/test_sensitivity_metrics.py
import numpy as np
import torch

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.sensitivity import (
    AvgSensitivity,
    ModelRandomization,
    RandomLogit
)
import xai_metrics.metrics.sensitivity.avg_sensitivity as avg_sensitivity_module

def test_avg_sensitivity_forwards_explainer_device_and_output(
    monkeypatch,
    context,
    explain_func
):
    expected = [0.10, 0.20]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(avg_sensitivity_module.quantus, "AvgSensitivity", fake)

    result = AvgSensitivity(
        context,
        explain_func,
        {
            "nr_samples": 3,
            "abs": False,
            "normalise": False,
            "lower_bound": 0.01,
            "upper_bound": 0.05
        }
    ).run()

    assert result == expected
    assert calls['init'] == {

        "nr_samples": 3,
        "abs": False,
        "normalise": False,
        "lower_bound": 0.01,
        "upper_bound": 0.05
    }
    assert_common_quantus_inputs(calls, context)
    assert calls['call']['explain_func'] is explain_func
    assert calls['call']['device'] == 'cpu'
    assert context.model.training is True


def test_model_randomization_does_not_modify_original_model(context, explain_func):
    original_parameters = [
        parameter.detach().clone()
        for parameter in context.model.parameters()
    ]
    received_models = []

    def recording_explain_func(model, inputs, targets):
        received_models.append(model)
        return explain_func(model, inputs, targets)

    result = ModelRandomization(
        context,
        recording_explain_func,
        {
            "fraction": 1.0,
            "reverse": True,
            "random_state": 42
        }
    ).run()

    assert len(result) == len(context.observations)
    assert received_models[0] is not context.model

    for current, original in zip(context.model.parameters(), original_parameters):
        assert torch.equal(current.detach(), original)


def test_random_logit_uses_a_different_class_per_observation(context, explain_func):
    received_targets = []

    def recording_explain_func(model, inputs, targets):
        received_targets.append(np.asarray(targets))
        return explain_func(model, inputs, targets)

    result = RandomLogit(
        context,
        recording_explain_func,
        {
            "num_classes": 2,
            "batch_size": 2,
            "random_state": 42
        }
    ).run()

    np.testing.assert_array_equal(received_targets[0], [0, 1])

    assert len(result) == len(context.observations)
    assert np.all(np.isfinite(result))