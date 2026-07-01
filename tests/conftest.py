# tests/conftest.py
from typing import Any

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