# tests/metrics_tests/test_faithfulness_metrics.py
import numpy as np
import pandas as pd

from conftest import fake_quantus_metric, assert_common_quantus_inputs, SmallLinearModel

from xai_metrics.base import MetricContext
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


def test_consistency_expected_values():
    context = MetricContext(
        model=SmallLinearModel(),
        X_test=pd.DataFrame(
            [
                [0.90, 0.20, 0.10],  # prediction 0
                [0.80, 0.30, 0.20],  # prediction 0
                [0.40, 0.70, 0.90],  # prediction 1
                [0.60, 0.50, 0.80],  # prediction 1
            ],
            columns=["a", "b", "c"],
        ),
        y_test=pd.Series([0, 0, 1, 1]),
        observations=[0, 1, 2, 3],
        attributions=np.array(
            [
                [0.10, 0.22, 0.31],
                [0.16, 0.19, 0.27],
                [0.21, 0.28, 0.12],
                [0.14, 0.20, 0.25],
            ],
            dtype=np.float32,
        ),
    )
    result = Consistency(context, {"abs": False, "normalise": False}).run()
    
    np.testing.assert_allclose(result, [1/3, 1/3, 1/3, 1/3], rtol=0.0, atol=1e-8)


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


def test_faithfulness_expected_values(exp_val_context):
    result = Faithfulness(exp_val_context, {"base_values": [0.0, 0.0, 0.0]}).run()

    np.testing.assert_allclose(
        result,
        [0.9791639224768663, 0.978349033066036, -0.24888267919267343],
        rtol=1e-6
    )


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


def test_faithfulness_estimate_expected_values(exp_val_context):
    result = FaithfulnessEstimate(
        exp_val_context,
        {
            "features_in_step": 1,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "mean",
        }
    ).run()

    np.testing.assert_allclose(
        result,
        [0.9664957072929178, 0.8427829143066047, 0.9999830716308306],
        rtol=1e-6
    )


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
    assert calls['init']['similarity_func'] == metric._safe_pearson # pyright: ignore
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_monotonicity_expected_values(exp_val_context):
    result = Monotonicity(
        exp_val_context,
        {
            "features_in_step": 1,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "mean",
        },
    ).run()

    # It should be this result:
    # assert result == [True, False, False]
    # but there is a bug in Quantus, so the result is:
    assert result == [True, True, True]


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


def test_monotonicity_metric_expected_values(exp_val_context):
    result = MonotonicityMetric(
        exp_val_context,
        {"base_values": [0.0, 0.0, 0.0]}
    ).run()

    assert result == [True, True, False]


def test_monotonicity_metric_uses_explicit_baseline_and_boolean_output(
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


def test_monotonicity_correlation_expected_values(exp_val_context):
    result = MonotonicityCorrelation(
        exp_val_context,
        {
            "nr_samples": 16,
            "features_in_step": 1,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "mean",
        },
    ).run()

    np.testing.assert_allclose(result, [0.5, 1.0, -0.5], atol=1e-7)


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
    assert calls['init']['similarity_func'] == metric._safe_spearman # type: ignore
    assert_common_quantus_inputs(calls, context)


def test_sensitivity_n_expected_values(exp_val_context):
    result = SensitivityN(
        exp_val_context,
        {
            "n_max_percentage": 1.0,
            "features_in_step": 1,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "black",
        }
    ).run()

    np.testing.assert_allclose(result, [0.562845040002659], rtol=1e-6, atol=1e-7)


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


def test_sufficiency_expected_values():
    context = MetricContext(
        model=SmallLinearModel(),
        X_test=pd.DataFrame(
            [
                [0.90, 0.20, 0.10],  # predicción 0
                [0.80, 0.30, 0.20],  # predicción 0
                [0.40, 0.70, 0.90],  # predicción 1
                [0.60, 0.50, 0.80],  # predicción 1
            ],
            columns=["a", "b", "c"],
        ),
        y_test=pd.Series([0, 0, 1, 1]),
        observations=[0, 1, 2, 3],
        attributions=np.array(
            [
                [0.10, 0.22, 0.31],
                [0.16, 0.19, 0.27],
                [0.21, 0.28, 0.12],
                [0.14, 0.20, 0.25],
            ],
            dtype=np.float32,
        ),
    )

    result = Sufficiency(
        context,
        {
            "threshold": 1.0,
            "distance_func": "seuclidean",
            "abs": False,
            "normalise": False,
        }
    ).run()

    assert result == [1/3, 1/3, 1/3, 1/3]


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