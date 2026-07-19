# tests/conftest.py
import pytest
import torch.nn as nn
import pandas as pd
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, BaseExplainer, ExplainerContext
from xai_metrics.base.metric_registry import METRIC_REGISTRY
from xai_metrics.base.explainer_registry import EXPLAINER_REGISTRY

class DummyMetric(BaseMetric):
    NAME = "dummy"

    def run(self):
        return self.params.get("value", 1.0)
    

class DummyExplainer(BaseExplainer):
    NAME = "dummy_xai"

    def explain(self, model, inputs, targets=None, **kwargs):
        return np.ones_like(inputs.to_numpy(), dtype=float)
    

@pytest.fixture
def dummy_metric_class():
    return DummyMetric


@pytest.fixture
def dummy_explainer_class():
    return DummyExplainer


@pytest.fixture
def clean_metric_registry():
    old_registry = dict(METRIC_REGISTRY)
    METRIC_REGISTRY.clear()

    yield

    METRIC_REGISTRY.clear()
    METRIC_REGISTRY.update(old_registry)


@pytest.fixture
def clean_explainer_registry():
    old_registry = dict(EXPLAINER_REGISTRY)
    EXPLAINER_REGISTRY.clear()

    yield

    EXPLAINER_REGISTRY.clear()
    EXPLAINER_REGISTRY.update(old_registry)


@pytest.fixture
def metric_context():
    return MetricContext(
        model=nn.Identity(),
        X_test=pd.DataFrame({"x1": [1.0]}),
        y_test=pd.Series([0]),
        observations=[0],
        attributions=np.array([[0.5]])
    )


@pytest.fixture
def explainer_context():
    return ExplainerContext(
        model=nn.Identity(),
        X_background=pd.DataFrame({"x1": [0.0], "x2": [1.0]}),
        y_background=pd.Series([0]),
        X_batch=pd.DataFrame({"x1": [1.0], "x2": [2.0]}, index=[10]),
        y_batch=pd.Series([1], index=[10]),
        device="cpu",
    )

# -----------------------
# METRICS TEST FIXTURE
# -----------------------

@pytest.fixture
def context():
    model = nn.Linear(3, 2)

    return MetricContext(
        model=model,
        X_test=pd.DataFrame(
            {
                "x1": [1.0, 4.0, 7.0],
                "x2": [2.0, 5.0, 8.0],
                "x3": [3.0, 6.0, 9.0],
            },
            index=[10, 20, 30]
        ),
        y_test=pd.Series([0, 1, 0], index=[10, 20, 30]),
        observations=[20, 10],
        attributions=np.array(
            [
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1]
            ]
        ),
        device='cpu'
    )


@pytest.fixture
def explain_func():
    def explain(model, inputs, targets=None, **kwargs):
        return np.asarray(inputs, dtype=float)
    
    return explain


def fake_quantus_metric(result):
    calls = {}

    class FakeMetric:
        def __init__(self, **kwargs):
            calls['init'] = kwargs
        
        def __call__(self, **kwargs):
            calls['call'] = kwargs
            return result
        
    return FakeMetric, calls


def assert_common_quantus_inputs(calls, context):
    call = calls['call']

    np.testing.assert_allclose(
        np.asarray(call['x_batch']),
        context.X_test.loc[context.observations].to_numpy()
    )
    np.testing.assert_array_equal(
        np.asarray(call['y_batch']),
        context.y_test.loc[context.observations].to_numpy()
    )
    np.testing.assert_allclose(
        call['a_batch'],
        context.attributions
    )