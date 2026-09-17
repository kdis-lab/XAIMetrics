# tests/metrics_tests/test_fidelity_metrics.py
import numpy as np

from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.fidelity.completeness import(
    AverageDrop,
    AverageGain,
    AverageIncrease,
    Completeness,
    Deletion,
    Insertion,
    MuFidelity
)
import xai_metrics.metrics.fidelity.completeness.completeness_metric as completeness_module

from xai_metrics.metrics.fidelity.soundness import NonSensitivity
import xai_metrics.metrics.fidelity.soundness.non_sensitivity as non_sensitivity_module

def test_average_drop_returns_one_score_per_observation(context):
    def probability_operator(mode, inputs, targets):
        return np.sum(inputs, axis=1) / 20.0

    result = AverageDrop(
        context,
        {
            "operator": probability_operator,
            "batch_size": 1
        }
    ).run()

    np.testing.assert_allclose(result, [49.0 / 90.0, 0.7], rtol=1e-6)
    assert len(result) == len(context.observations)


def test_average_gain_returns_one_score_per_observation(context):
    def probability_operator(mode, inputs, targets):
        return 1.0 - np.sum(inputs, axis=1) / 20.0

    result = AverageGain(
        context,
        {
            "operator": probability_operator,
            "batch_size": 1
        }
    ).run()

    np.testing.assert_allclose(result, [49.0 / 90.0, 0.7], rtol=1e-6)
    assert len(result) == len(context.observations)


def test_average_increase_returns_one_score_per_observation(context):
    def probability_operator(mode, inputs, targets):
        return 1.0 - np.sum(inputs, axis=1) / 20.0

    result = AverageIncrease(
        context,
        {
            "operator": probability_operator,
            "batch_size": 1
        }
    ).run()

    assert result == [1.0, 1.0]
    assert len(result) == len(context.observations)


def test_completeness_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [True, False]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(completeness_module.quantus, "Completeness", fake)

    result = Completeness(
        context,
        {
            "abs": True,
            "normalise": False,
            "perturb_baseline": "mean"
        }
    ).run()

    assert result == expected
    assert calls['init'] == {
        "abs": True,
        "normalise": False,
        "perturb_baseline": "mean"
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False


def test_deletion_returns_auc_per_observation(context):
    def score_operator(mode, inputs, targets):
        return np.sum(inputs, axis=1)

    result = Deletion(
        context,
        {
            "operator": score_operator,
            "baseline_model": 0.0,
            "steps": 3,
            "max_percentage_perturbed": 1.0,
            "batch_size": 1
        }
    ).run()

    np.testing.assert_allclose(result, [41.0 / 6.0, 22.0 / 6.0])
    assert len(result) == len(context.observations)


def test_insertion_returns_auc_per_observation(context):
    def score_operator(mode, inputs, targets):
        return np.sum(inputs, axis=1)

    result = Insertion(
        context,
        {
            "operator": score_operator,
            "baseline_model": 0.0,
            "steps": 3,
            "max_percentage_perturbed": 1.0,
            "batch_size": 1
        }
    ).run()

    np.testing.assert_allclose(result, [49.0 / 6.0, 14.0 / 6.0])
    assert len(result) == len(context.observations)


def test_mufidelity_returns_deterministc_score_per_observation(context):
    def score_operator(model, inputs, targets):
        return np.sum(inputs, axis=1)

    params = {
        "operator": score_operator,
        "nb_samples": 7,
        "subset_percent": 0.5,
        "batch_size": 4,
        "random_state": 42
    }

    first = MuFidelity(context, params).run()
    second = MuFidelity(context, params).run()

    assert len(first) == len(context.observations)
    np.testing.assert_allclose(first, second)
    assert np.all(np.isfinite(first))


def test_non_sensitivity_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [0.0, 1.0]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(non_sensitivity_module.quantus, "NonSensitivity", fake)

    result = NonSensitivity(
        context,
        {
            "eps": 0.01,
            "features_in_step": 2,
            "abs": False,
            "normalise": False,
            "perturb_baseline": "zero"
        },
    ).run()

    assert result == expected
    assert calls['init'] == {
        "eps": 0.01,
        "features_in_step": 2,
        "abs": False,
        "normalise": False,
        "perturb_baseline": "zero"
    }
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is False