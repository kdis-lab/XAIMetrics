# tests/metrics_tests/test_sensitivity_metrics.py
import numpy as np
import torch

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.base import MetricContext
from xai_metrics.metrics.sensitivity import (
    AvgSensitivity,
    ModelRandomization,
    RandomLogit
)
import xai_metrics.metrics.sensitivity.avg_sensitivity as avg_sensitivity_module


def test_avg_sensitivity_expected_values(exp_val_context):
    def explain_func(model, inputs, targets=None, **kwargs):
        return 0.3 * (np.asarray(inputs, dtype=np.float32) + 1.0)

    adapted_context = MetricContext(
        model=exp_val_context.model,
        X_test=exp_val_context.X_test,
        y_test=exp_val_context.y_test,
        observations=exp_val_context.observations,
        attributions=explain_func(
            exp_val_context.model,
            exp_val_context.X_test.to_numpy(dtype=np.float32)
        )
    )

    rng_state = np.random.get_state()
    
    try:
        np.random.seed(3)
        result = AvgSensitivity(
            adapted_context,
            explain_func,
            {
                "nr_samples": 8,
                "abs": False,
                "normalise": False,
                "lower_bound": 0.01,
                "upper_bound": 0.02
            } # pyright: ignore[reportCallIssue]
        ).run()
    finally:
        np.random.set_state(rng_state)

        
    np.testing.assert_allclose(
        result,
        [0.00879588, 0.00953731, 0.00833813],
        rtol=0.0,
        atol=1e-6
    )


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
        } # pyright: ignore[reportCallIssue]
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


def test_model_randomization_expected_values(exp_val_context):
    def weight_explainer(model, inputs, targets=None, **kwargs):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=int)

        with torch.no_grad():
            weights = model.linear.weight.detach().cpu().numpy()

        return inputs * weights[targets]

    adapted_context = MetricContext(
        model=exp_val_context.model,
        X_test=exp_val_context.X_test,
        y_test=exp_val_context.y_test,
        observations=exp_val_context.observations,
        attributions=weight_explainer(
            exp_val_context.model,
            exp_val_context.X_test.to_numpy(dtype=np.float32),
            exp_val_context.y_test.loc[exp_val_context.observations].to_numpy(dtype=int)
        )
    )
    
    result = ModelRandomization(
        adapted_context,
        weight_explainer,
        {"fraction": 1.0, "random_state": 3}, # pyright: ignore[reportCallIssue]
    ).run()

    np.testing.assert_allclose(
        result,
        [0.5, -0.5, -1.0],
        rtol=0.0,
        atol=1e-8
    )


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
        } # pyright: ignore[reportCallIssue]
    ).run()

    assert len(result) == len(context.observations)
    assert received_models[0] is not context.model

    for current, original in zip(context.model.parameters(), original_parameters):
        assert torch.equal(current.detach(), original)


def test_random_logit_expected_values(exp_val_context):
    def explain_func(model, inputs, targets=None, **kwargs):
        return 0.3 * (np.asarray(inputs, dtype=np.float32) + 1.0)
    
    result = RandomLogit(
        exp_val_context,
        explain_func,
        {"num_classes": 2, "random_state": 3} # pyright: ignore[reportCallIssue]
    ).run()
    
    np.testing.assert_allclose(
        result,
        [0.67647599, 0.58228961, 0.69605679],
        rtol=0.0,
        atol=1e-8
    )


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
        } # pyright: ignore[reportCallIssue]
    ).run()

    np.testing.assert_array_equal(received_targets[0], [0, 1])

    assert len(result) == len(context.observations)
    assert np.all(np.isfinite(result))