# tests/metrics_tests/test_fidelity_metrics.py
from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.fidelity.completeness import Completeness
import xai_metrics.metrics.fidelity.completeness.completeness_metric as completeness_module

from xai_metrics.metrics.fidelity.soundness import NonSensitivity
import xai_metrics.metrics.fidelity.soundness.non_sensitivity as non_sensitivity_module

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