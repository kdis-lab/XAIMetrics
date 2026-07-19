# tests/metrics_tests/test_complexity_metrics.py
from conftest import fake_quantus_metric, assert_common_quantus_inputs

from xai_metrics.metrics.complexity import Complexity, Sparseness
import xai_metrics.metrics.complexity.complexity_metric as complexity_module
import xai_metrics.metrics.complexity.sparseness as sparseness_module

def test_complexity_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [0.25, 0.50]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(complexity_module.quantus, "Complexity", fake)

    result = Complexity(context, {"normalise": False}).run()

    assert result == expected
    assert calls['init'] == {"normalise": False}
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is True


def test_sparseness_forwards_inputs_parameters_and_output(monkeypatch, context):
    expected = [0.75, 0.40]
    fake, calls = fake_quantus_metric(expected)
    monkeypatch.setattr(sparseness_module.quantus, "Sparseness", fake)

    result = Sparseness(context, {"normalise": False}).run()

    assert result == expected
    assert calls['init'] == {"normalise": False}
    assert_common_quantus_inputs(calls, context)
    assert context.model.training is True