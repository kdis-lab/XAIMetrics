# tests/metrics_tests/test_sensitivity_metrics.py
from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.sensitivity import AvgSensitivity
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