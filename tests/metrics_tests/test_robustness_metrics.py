# tests/metrics_tests/test_robustness_metrics.py
from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.robustness import (
    MaxSensitivity,
    LocalLipschitzEstimate,
    RelativeInputStability,
    RelativeOutputStability
)
import xai_metrics.metrics.robustness.max_sensitivity as max_sensitivity_module
import xai_metrics.metrics.robustness.local_lipschitz_estimate as lipschitz_module
import xai_metrics.metrics.robustness.relative_input_stability as ris_module
import xai_metrics.metrics.robustness.relative_output_stability as ros_module

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