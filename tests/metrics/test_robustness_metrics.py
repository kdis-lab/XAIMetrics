# tests/metrics_tests/test_robustness_metrics.py
import pytest
import numpy as np

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.robustness import (
    AverageStability,
    LocalLipschitzEstimate,
    MaxSensitivity,
    MeGe,
    RelativeInputStability,
    RelativeOutputStability
)
import xai_metrics.metrics.robustness.max_sensitivity as max_sensitivity_module
import xai_metrics.metrics.robustness.local_lipschitz_estimate as lipschitz_module
import xai_metrics.metrics.robustness.relative_input_stability as ris_module
import xai_metrics.metrics.robustness.relative_output_stability as ros_module


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
        }
    ).run()

    expected = [
        np.sqrt(0.1**2 + 0.2**2 + 0.7**2),
        np.sqrt(0.6**2 + 0.3**2 + 0.1**2)
    ]

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(received_targets[0], [1, 1, 1])
    np.testing.assert_array_equal(received_targets[1], [0, 0, 0])


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
        }
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
        }
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


def test_mege_returns_one_socre_per_Observation(mege_context):
    training_calls = []

    class PerfectModel:
        def predict(self, inputs):
            return np.asarray(inputs)[:, 0].astype(int)

    def training_func(X_train, y_train, X_holdout, y_holdout):
        training_calls.append(
            {
                "X_train": np.asarray(X_train),
                "y_train": np.asarray(y_train),
                "X_holdout": np.asarray(X_holdout),
                "y_holdout": np.asarray(y_holdout)
            }
        )

        return PerfectModel()

    def explain_func(model, inputs, targets):
        return np.asarray(inputs, dtype=np.float32)

    result = MeGe(
        mege_context,
        training_func,
        explain_func,
        {"k_splits": 2}
    ).run()

    assert len(training_calls) == 2
    assert all(call['X_train'].shape == (2, 3) for call in training_calls)
    assert all(call['y_train'].shape == (2,) for call in training_calls)

    np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0])


def test_mege_requires_training_and_explanation_functions(context, explain_func):
    def training_func(X_train, y_train, X_holdout, y_holdout):
        return context.model

    with pytest.raises(ValueError, match="requires 'training_func'"):
        MeGe(context, None, explain_func)

    with pytest.raises(ValueError, match="requires 'explain_func'"):
        MeGe(context, training_func, None)


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
        }
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
        }
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