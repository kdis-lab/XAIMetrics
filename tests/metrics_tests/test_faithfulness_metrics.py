# tests/metrics_tests/test_faithfulness_metrics.py
import numpy as np

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.faithfulness import (
    Consistency,
    Faithfulness,
    FaithfulnessEstimate,
    Monotonicity,
    MonotonicityMetric,
    MonotonicityCorrelation,
    SensitivityN,
    Sufficiency
)
import xai_metrics.metrics.faithfulness.consistency as consistency_module
import xai_metrics.metrics.faithfulness.faithfulness as faithfulness_module
import xai_metrics.metrics.faithfulness.faithfulness_estimate as estimate_module
import xai_metrics.metrics.faithfulness.monotonicity as monotonicity_module
import xai_metrics.metrics.faithfulness.monotonicity_metric as monotonicity_metric_module
import xai_metrics.metrics.faithfulness.monotonicity_correlation as correlation_module
import xai_metrics.metrics.faithfulness.sensitivity_n as sensitivity_n_module
import xai_metrics.metrics.faithfulness.sufficiency as sufficiency_module


def test_consistency_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [1.0, 0.5]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(consistency_module.quantus, "Consistency", fake)

    result = Consistency(
        context,
        {"abs": False, "normalise": False}
    ).run()

    assert result == expected
    assert calls['init'] == {
        "abs": False,
        "normalise": False
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_faithfulness_uses_selected_rows_baseline_and_returns_floats(
    monkeypatch,
    context
):
    calls = []

    def fake_faithfulness_metric(model, x, coefs, base):
        calls.append(
            {
                "model": model,
                "x": x,
                "coefs": coefs,
                "base": base
            }
        )
        return 0.8
    
    monkeypatch.setattr(faithfulness_module, "faithfulness_metric", fake_faithfulness_metric)

    result = Faithfulness(context, {"base_strategy": "mean"}).run()

    assert result == [0.8, 0.8]
    assert len(calls) == 2

    np.testing.assert_allclose(calls[0]['x'], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(calls[1]['x'], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(calls[0]['coefs'], [0.1, 0.2, 0.7])
    np.testing.assert_allclose(calls[1]['coefs'], [0.6, 0.3, 0.1])

    np.testing.assert_allclose(calls[0]['base'], [4.0, 5.0, 6.0])
    assert calls[0]['model'] is context.model


def test_faithfulness_estimate_forwards_inputs_and_output(monkeypatch, context):
    expected = [0.70, 0.85]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(estimate_module.quantus, "FaithfulnessEstimate", fake)

    metric = FaithfulnessEstimate(
        context,
        {
            "features_in_step": 2,
            "abs": True,
            "normalise": False,
            "perturb_baseline": "mean"
        }
    )
    result = metric.run()

    assert result == expected
    assert calls['init']['features_in_step'] == 2
    assert calls['init']['abs'] is True
    assert calls['init']['normalise'] is False
    assert calls['init']['perturb_baseline'] == 'mean'
    assert calls['init']['similarity_func'] == metric._safe_pearson
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_monotonicity_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [True, False]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(monotonicity_module.quantus, "Monotonicity", fake)

    result = Monotonicity(
        context,
        {
            "features_in_step": 2,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "mean"
        }
    ).run()

    assert result == expected
    assert calls['init'] == {
        "features_in_step": 2,
        "abs": False,
        "normalise": False,
        "perturb_baseline": "mean"
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_aix360_monotonicity_uses_explicit_baseline_and_boolean_output(
    monkeypatch,
    context
):
    calls = []
    returned_values = iter([True, False])

    def fake_monotonicity_metric(model, x, coefs, base):
        calls.append(
            {
                "model": model,
                "x": x,
                "coefs": coefs,
                "base": base
            }
        )
        return next(returned_values)
    
    monkeypatch.setattr(monotonicity_metric_module, "monotonicity_metric", fake_monotonicity_metric)

    result = MonotonicityMetric(context, {"base_values": [0.0, 0.0, 0.0]}).run()

    assert result == [True, False]
    assert all(type(value) is bool for value in result)

    np.testing.assert_allclose(calls[0]['x'], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(calls[1]['x'], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(calls[0]['base'], [0.0, 0.0, 0.0])


def test_monotonicity_correlation_forwards_inputs_and_safe_spearman(
    monkeypatch,
    context
):
    expected = [0.50, 0.75]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(correlation_module.quantus, "MonotonicityCorrelation", fake)

    metric = MonotonicityCorrelation(
        context,
        {
            "eps": 0.001,
            "nr_samples": 8,
            "features_in_step": 2,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "mean"
        }
    )
    result = metric.run()

    assert result == expected
    assert calls['init']['eps'] == 0.001
    assert calls['init']['nr_samples'] == 8
    assert calls['init']['features_in_step'] == 2
    assert calls['init']['abs'] is False
    assert calls['init']['normalise'] is False
    assert calls['init']['perturb_baseline'] == 'mean'
    assert calls['init']['similarity_func'] == metric._safe_spearman
    assert_common_quantus_inputs(calls, context)


def test_sensitivity_n_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [0.60, 0.90]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(sensitivity_n_module.quantus, "SensitivityN", fake)

    result = SensitivityN(
        context,
        {
            "n_max_percentage": 0.5,
            "features_in_step": 2,
            "abs": True,
            "normalise": False,
            "perturb_baseline": "mean"
        }
    ).run()

    assert result == expected
    assert calls['init'] == {
        "n_max_percentage": 0.5,
        "features_in_step": 2,
        "abs": True,
        "normalise": False,
        "perturb_baseline": "mean"
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_sufficiency_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [1.0, 0.5]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(sufficiency_module.quantus, "Sufficiency", fake)

    result = Sufficiency(
        context,
        {
            "threshold": 0.25,
            "distance_func": "euclidean",
            "abs": False,
            "normalise": False,
        }
    ).run()

    assert result == expected
    assert calls['init'] == {
        "threshold": 0.25,
        "distance_func": "euclidean",
        "abs": False,
        "normalise": False,
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False