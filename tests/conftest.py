# tests/conftest.py
import pytest
import torch
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


# ------------------------
# METRICS TEST FIXTURES
# ------------------------


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


@pytest.fixture
def mege_context():
    inputs = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [0.0, 3.0, 4.0],
            [1.0, 4.0, 5.0],
        ]
    )

    return MetricContext(
        model=None, # pyright: ignore[reportArgumentType]
        X_test=pd.DataFrame(inputs, index=[10, 20, 30, 40]),
        y_test=pd.Series([0, 1, 0, 1], index=[10, 20, 30, 40]),
        observations=[10, 20, 30, 40],
        attributions=inputs.copy()
    )


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


# ---------------------------
# EXPLAINERS TEST FIXTURES
# ---------------------------

class DummyClassificationModel:
    def predict_proba(self, inputs):
        inputs = np.asarray(inputs, dtype=float)

        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        score = inputs.mean(axis=1)

        return np.column_stack([
            1.0 - score,
            score
        ])
    

@pytest.fixture
def xai_context():
    return ExplainerContext(
        model=None,
        X_background=pd.DataFrame(
            {
                "x1": [0.0, 0.1, 0.2, 0.3, 0.4],
                "x2": [1.0, 1.1, 1.2, 1.3, 1.4],
                "x3": [2.0, 2.1, 2.2, 2.3, 2.4],
            }
        ),
        y_background=pd.Series([0, 0, 1, 1, 1]),
        X_batch=pd.DataFrame(
            {
                "x1": [10.0, 20.0],
                "x2": [11.0, 21.0],
                "x3": [12.0, 22.0],
            },
            index=[10, 20]
        ),
        y_batch=pd.Series([1, 0], index=[10, 20]),
        device='cpu'
    )


@pytest.fixture
def classification_model():
    return DummyClassificationModel()


class SmallLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.25, 0.15, 0.10],
                        [0.10, 0.20, 0.30]
                    ]
                )
            )
            self.linear.bias.copy_(torch.tensor([0.05, 0.10]))

    def forward(self, inputs):
        return self.linear(inputs)

    def predict_proba(self, inputs):
        inputs = np.array(inputs, dtype=np.float32, copy=True)

        with torch.no_grad():
            logits = self(
                torch.as_tensor(inputs, dtype=torch.float32)
            ).numpy()

        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


@pytest.fixture
def exp_val_context():
    return MetricContext(
        model=SmallLinearModel(),
        X_test=pd.DataFrame(
            [
                [0.40, 0.70, 0.90],
                [0.60, 0.50, 0.80],
                [0.70, 0.90, 0.40],
            ],
            columns=['a', 'b', 'c']
        ),
        y_test=pd.Series([1, 1, 0]),
        observations=[0, 1, 2],
        attributions=np.array(
            [
                [0.10, 0.22, 0.31],
                [0.16, 0.19, 0.27],
                [0.21, 0.28, 0.12],
            ],
            dtype=np.float32,
        )
    )