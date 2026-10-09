# tests/metrics_tests/test_robustness_metrics.py
import numpy as np

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.base import MetricContext
from xai_metrics.metrics.robustness import (
    AverageStability,
    LocalLipschitzEstimate,
    MaxSensitivity,
    RelativeInputStability,
    RelativeOutputStability
)
import xai_metrics.metrics.robustness.max_sensitivity as max_sensitivity_module
import xai_metrics.metrics.robustness.local_lipschitz_estimate as lipschitz_module
import xai_metrics.metrics.robustness.relative_input_stability as ris_module
import xai_metrics.metrics.robustness.relative_output_stability as ros_module


def test_average_stability_expected_values(exp_val_context):
    def explain_func(model, inputs, targets=None, **kwargs):
        return 0.3 * np.asarray(inputs, dtype=np.float32)
    
    adapted_context = MetricContext(
        model=exp_val_context.model,
        X_test=exp_val_context.X_test,
        y_test=exp_val_context.y_test,
        observations=exp_val_context.observations,
        attributions=0.3 * exp_val_context.X_test.to_numpy(dtype=np.float32)
    )

    result = AverageStability(
        adapted_context,
        explain_func,
        {
            "radius": 0.10,
            "nb_samples": 8,
            "random_state": 3,
            "distance": "l2",
        }, # pyright: ignore[reportCallIssue]
    ).run()
    
    np.testing.assert_allclose(
        result,
        [0.026880339, 0.026880339, 0.026880339],
        rtol=0.0,
        atol=1e-7
    )


def test_average_stability_returns_one_score_per_observation(context):
    received_targets = []

    def explain_func(model, inputs, targets):
        received_targets.append(np.asarray(targets))
        return np.zeros_like(inputs, dtype=np.float32)

    result = AverageStability(
        context,
        explain_func,
        {
            "radius": 0.2,
            "nb_samples": 3,
            "distance": "l2",
            "random_state": 42
        } # pyright: ignore[reportCallIssue]
    ).run()

    expected = [
        np.sqrt(0.1**2 + 0.2**2 + 0.7**2),
        np.sqrt(0.6**2 + 0.3**2 + 0.1**2)
    ]

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(received_targets[0], [1, 1, 1])
    np.testing.assert_array_equal(received_targets[1], [0, 0, 0])


def test_local_lipschitz_estimate_expected_values(exp_val_context):
    def explain_func(model, inputs, targets=None, **kwargs):
        return 0.3 * np.asarray(inputs, dtype=np.float32)
    
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
        result = LocalLipschitzEstimate(
            adapted_context,
            explain_func,
            {
                "nr_samples": 8,
                "abs": False,
                "normalise": False,
                "perturb_mean": 0.01,
                "perturb_std": 0.02,
            } # pyright: ignore[reportCallIssue]
        ).run()
    finally:
        np.random.set_state(rng_state)

    np.testing.assert_allclose(
        result,
        [0.3, 0.3, 0.3],
        rtol=1e-6,
        atol=1e-6
    )


def test_local_lipschitz_forwards_explainer_device_and_output(
    monkeypatch,
    context,
    explain_func
):
    expected = [1.2, 1.4]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(lipschitz_module.quantus, "LocalLipschitzEstimate", fake)

    result = LocalLipschitzEstimate(
        context,
        explain_func,
        {
            "nr_samples": 5,
            "abs": True,
            "normalise": False,
            "perturb_mean": 0.1,
            "perturb_std": 0.2
        } # pyright: ignore[reportCallIssue]
    ).run()

    assert result == expected
    assert calls['init'] == {
        "nr_samples": 5,
        "abs": True,
        "normalise": False,
        "perturb_mean": 0.1,
        "perturb_std": 0.2
    }
    assert_common_quantus_inputs(calls, context)
    assert calls['call']['explain_func'] is explain_func
    assert calls['call']['device'] == 'cpu'
    assert context.model.training is True


def test_max_sensitivity_expected_values(exp_val_context):
    def explain_func(model, inputs, targets=None, **kwargs):
        return 0.3 * np.asarray(inputs, dtype=np.float32)
    
    adapted_context = MetricContext(
        model=exp_val_context.model,
        X_test=exp_val_context.X_test,
        y_test=exp_val_context.y_test,
        observations=exp_val_context.observations,
        attributions=0.3 * exp_val_context.X_test.to_numpy(dtype=np.float32)
    )

    rng_state = np.random.get_state()

    try:
        np.random.seed(3)
        result = MaxSensitivity(
            adapted_context,
            explain_func,
            {
                "nr_samples": 8,
                "abs": False,
                "normalise": False,
                "lower_bound": 0.01,
                "upper_bound": 0.02,
            } # pyright: ignore[reportCallIssue]
        ).run()
    finally:
        np.random.set_state(rng_state)
        
    np.testing.assert_allclose(
        result,
        [0.0228532, 0.02873633, 0.02230252],
        rtol=1e-6,
        atol=1e-7
    )


def test_max_sensitivity_forwards_explainer_device_and_output(
    monkeypatch,
    context,
    explain_func
):
    expected = [0.30, 0.4]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(max_sensitivity_module.quantus, "MaxSensitivity", fake)

    result = MaxSensitivity(
        context,
        explain_func,
        {
            "nr_samples": 4,
            "abs": True,
            "normalise": True,
            "lower_bound": 0.02,
            "upper_bound": 0.08
        } # pyright: ignore[reportCallIssue]
    ).run()

    assert result == expected
    assert calls['init'] == {
        "nr_samples": 4,
        "abs": True,
        "normalise": True,
        "lower_bound": 0.02,
        "upper_bound": 0.08
    }
    assert_common_quantus_inputs(calls, context)
    assert calls['call']['explain_func'] is explain_func
    assert calls['call']['device'] == 'cpu'
    assert context.model.training is True


def test_relative_input_stability_expected_values(exp_val_context):
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
        result = RelativeInputStability(
            adapted_context,
            explain_func,
            {"nr_samples": 8, "abs": False, "normalise": False} # pyright: ignore[reportCallIssue]
        ).run()
    finally:
        np.random.set_state(rng_state)
        
    np.testing.assert_allclose(
        result,
        [0.41804043, 0.4215035, 0.41804802],
        rtol=0.0,
        atol=1e-6
    )


def test_relative_input_stability_forwards_inputs_and_output(
    monkeypatch,
    context,
    explain_func
):
    expected = [0.12, 0.25]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(ris_module.quantus, "RelativeInputStability", fake)

    result = RelativeInputStability(
        context,
        explain_func,
        {
            "nr_samples": 6,
            "abs": True,
            "normalise": True
        } # pyright: ignore[reportCallIssue]
    ).run()

    assert result == expected
    assert calls['init'] == {
        "nr_samples": 6,
        "abs": True,
        "normalise": True
    }
    assert_common_quantus_inputs(calls, context)
    assert calls['call']['explain_func'] is explain_func
    assert calls['call']['device'] == 'cpu'
    assert context.model.training is False


def test_relative_output_stability_expected_values(exp_val_context):
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
        result = RelativeOutputStability(
            adapted_context,
            explain_func,
            {"nr_samples": 8, "abs": False, "normalise": False} # pyright: ignore[reportCallIssue]
        ).run()
    finally:
        np.random.set_state(rng_state)
        
    np.testing.assert_allclose(
        result,
        [1.5852848, 1.5531414, 1.6003472],
        rtol=0.0,
        atol=1e-6
    )


def test_relative_output_stability_forwards_inputs_and_output(
    monkeypatch,
    context,
    explain_func
):
    expected = [0.15, 0.35]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(ros_module.quantus, "RelativeOutputStability", fake)

    result = RelativeOutputStability(
        context,
        explain_func,
        {
            "nr_samples": 7,
            "abs": True,
            "normalise": False
        } # pyright: ignore[reportCallIssue]
    ).run()

    assert result == expected
    assert calls['init'] == {
        "nr_samples": 7,
        "abs": True,
        "normalise": False
    }
    assert_common_quantus_inputs(calls, context)
    assert calls['call']['explain_func'] is explain_func
    assert calls['call']['device'] == 'cpu'
    assert context.model.training is False