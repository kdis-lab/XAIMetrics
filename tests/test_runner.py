# tests/test_runner.py
from typing import Any

import numpy as np

import pytest
from torch.nn import Module

import xai_metrics.runner.runner as runner_module
from xai_metrics.base import BaseMetric, MetricSkipped, BaseExplainer, ExplainerSkipped
from xai_metrics.base.metric_registry import METRIC_REGISTRY
from xai_metrics.base.explainer_registry import EXPLAINER_REGISTRY
from xai_metrics.runner import run_evaluation, run_explanation


def test_resolve_explainer_selection_returns_configured_registered_explainers():
    with pytest.warns(
        UserWarning,
        match="Configured explainers not registered and will be skipped",
    ):
        result = runner_module._resolve_explainer_selection(
            selected_explainers=None,
            configured_explainers=['dummy_xai', 'missing_xai'],
            registered_explainers=['dummy_xai']
        )

    assert result == ['dummy_xai']


def test_resolve_metric_selection_returns_configured_registered_metrics():
    with pytest.warns(
        UserWarning,
        match="Configured metrics not registered and will be skipped",
    ):
        result = runner_module._resolve_metric_selection(
            selected_metrics=None,
            configured_metrics=['dummy', 'missing_metric'],
            registered_metrics=['dummy']
        )

    assert result == ['dummy']


def test_resolve_explainer_selection_filters_selected_explainers():
    with pytest.warns(UserWarning) as warnings:
        result = runner_module._resolve_explainer_selection(
            selected_explainers=['dummy_xai', 'missing_xai'],
            configured_explainers=['dummy_xai', 'other_xai'],
            registered_explainers=['dummy_xai', 'other_xai'],
        )

    warning_messages = [str(warning.message) for warning in warnings]

    assert result == ["dummy_xai"]
    assert any(
        "Selected explainers not present in config and will be skipped" in message
        for message in warning_messages
    )
    assert any(
        "Selected explainers not registered and will be skipped" in message
        for message in warning_messages
    )


def test_resolve_metric_selection_filters_selected_metrics():
    with pytest.warns(UserWarning) as warnings:
        result = runner_module._resolve_metric_selection(
            selected_metrics=['dummy', 'missing_metric'],
            configured_metrics=['dummy', 'other_metric'],
            registered_metrics=['dummy', 'other_metric']
        )

    warning_messages = [str(warning.message) for warning in warnings]

    assert result == ['dummy']
    assert any(
        "Selected metrics not present in config and will be skipped" in message
        for message in warning_messages
    )
    assert any(
        "Selected metrics not registered and will be skipped" in message
        for message in warning_messages
    )


def test_validate_context_metadata_requires_metadata():
    with pytest.raises(ValueError, match="metadata must also be provided"):
        runner_module._validate_context_metadata(None)


def test_run_explanation_requires_expected_metadata_fields(
    monkeypatch,
    explainer_context
):
    monkeypatch.setattr(
        runner_module,
        "autodiscover_explainers",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="Explanation metadata is missing required fields"):
        run_explanation(
            context=explainer_context,
            config={
                "context": {
                    "dataset_name": "dataset",
                },
                "explainers": [],
            },
            attribution_output_dir=None,
        )


def test_validate_context_metadata_requires_expected_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        runner_module._validate_context_metadata(
            {
                "dataset_name": "dataset",
                "model_name": "model"
            }
        )


def test_run_explanation_runs_registered_explainer_without_saving_attributions(
    monkeypatch,
    clean_explainer_registry,
    explainer_context,
    dummy_explainer_class
):
    EXPLAINER_REGISTRY['dummy_xai'] = dummy_explainer_class

    monkeypatch.setattr(
        runner_module,
        "autodiscover_explainers",
        lambda *args, **kwargs: None
    )

    result = run_explanation(
        context=explainer_context,
        config={
            "context": {
                "dataset_name": "dataset",
                "model_name": "model",
            },
            "explainers": [
                {"name": "dummy_xai", "params": {"seed": 1}}
            ],
        },
        attribution_output_dir=None,
    )

    assert result['contexts'][0]['metadata'] == {
        "dataset_name": "dataset",
        "model_name": "model"
    }
    assert result['contexts'][0]['explainer_params'] == {
        "dummy_xai": {"seed": 1}
    }
    assert result['contexts'][0]['skipped'] == {}
    assert result['attribution_paths'] == {}

    assert np.array_equal(
        result['contexts'][0]['attributions']['dummy_xai'],
        np.array([[1.0, 1.0]])
    )


def test_run_evaluation_runs_registered_metric_without_saving_reports(
    monkeypatch,
    clean_metric_registry,
    metric_context,
    dummy_metric_class
):
    METRIC_REGISTRY["dummy"] = dummy_metric_class

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={
            "metrics": [
                {"name": "dummy", "params": {"value": 3.0}}
            ]
        },
        report_output_dir=None
    )

    assert result['contexts'][0]['results'] == [
        {
            "metric": "dummy",
            "metric_params": {"value": 3.0},
            "value": 3.0,
        }
    ]
    assert result['contexts'][0]['skipped'] == []
    assert result['report_paths'] == {}

    report = result['reports']['dataset']['model']
    report_by_metric = report.set_index("metric")

    assert report_by_metric.loc["dummy", "dataset_name"] == "dataset"
    assert report_by_metric.loc["dummy", "model_name"] == "model"
    assert report_by_metric.loc["dummy", "metric_params"] == {"value": 3.0}
    assert report_by_metric.loc["dummy", "lime"] == 3.0


def test_run_explanation_records_skipped_explainers(
    monkeypatch,
    clean_explainer_registry,
    explainer_context
):
    class SkippedExplainer(BaseExplainer):
        NAME = "skipped_xai"

        def explain(self, model, inputs, targets=None, **kwargs):
            raise ExplainerSkipped("not applicable")
        
    EXPLAINER_REGISTRY['skipped_xai'] = SkippedExplainer

    monkeypatch.setattr(
        runner_module,
        "autodiscover_explainers",
        lambda *args, **kwargs: None,
    )

    result = run_explanation(
        context=explainer_context,
        config={
            "context": {
                "dataset_name": "dataset",
                "model_name": "model",
            },
            "explainers": [
                {"name": "skipped_xai"}
            ],
        },
        attribution_output_dir=None,
    )

    assert result['contexts'][0]['attributions'] == {}
    assert result['contexts'][0]['skipped'] == {
        "skipped_xai": "not applicable"
    }


def test_run_evaluation_records_skipped_metrics(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    class SkippedMetric(BaseMetric):
        NAME = "skipped"

        def run(self):
            raise MetricSkipped("not applicable")

    METRIC_REGISTRY['skipped'] = SkippedMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime",
        },
        config={"metrics": [{"name": "skipped"}]},
        report_output_dir=None,
    )

    assert result['contexts'][0]['results'] == []
    assert result['contexts'][0]['skipped'] == [
        {
            "metric": "skipped",
            "metric_params": {},
            "reason": "not applicable",
        }
    ]


def test_run_evaluation_accepts_explainer_as_explain_func(
    monkeypatch,
    clean_metric_registry,
    metric_context,
    explainer_context,
    dummy_explainer_class
):
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            if self.explain_func is None:
                raise ValueError("explain_func is required")
    
            return self.explain_func(
                model=self.context.model,
                inputs=self.context.X_test,
                targets=self.context.y_test,
            )

    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    explainer = dummy_explainer_class(context=explainer_context)

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "dummy_xai",
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_funcs={"dummy_xai": explainer.as_explain_func()},
    )

    assert np.array_equal(
        result['contexts'][0]['results'][0]['value'],
        np.array([[1.0]])
    )


def test_run_evaluation_passes_explain_func_by_xai_method(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    def lime_explain_func():
        return "lime explanation"
    
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            if self.explain_func is None:
                raise ValueError("explain_func is required")
    
            return self.explain_func()
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_funcs={"lime": lime_explain_func}
    )

    assert result['contexts'][0]['results'] == [
        {
            "metric": "explain_func_metric",
            "metric_params": {},
            "value": "lime explanation",
        }
    ]


def test_run_evaluation_accepts_single_explain_func(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    def explain_func():
        return "single explanation"
    
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            if self.explain_func is None:
                raise ValueError("explain_func is required")
    
            return self.explain_func()
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime"
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_func=explain_func
    )

    assert result['contexts'][0]['results'] == [
        {
            "metric": "explain_func_metric",
            "metric_params": {},
            "value": "single explanation",
        }
    ]


def test_run_evaluation_fails_when_explain_func_is_missing(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def run(self):
            return 1.0
        
    METRIC_REGISTRY['explain_func_metric'] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="No explain_func provided"):
        run_evaluation(
            context=metric_context,
            metadata={
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "shap"
            },
            config={"metrics": [{"name": "explain_func_metric"}]},
            report_output_dir=None,
            explain_funcs={"lime": lambda: None}
        )


def test_run_evaluation_accepts_explain_funcs_by_xai_method(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    def lime_explain_func(model, inputs, targets=None):
        return "lime explanation"

    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            if self.explain_func is None:
                raise ValueError("explain_func is required")
    
            return self.explain_func(
                model=self.context.model,
                inputs=self.context.X_test,
                targets=self.context.y_test,
            )

    METRIC_REGISTRY["explain_func_metric"] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime",
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_funcs={"lime": lime_explain_func},
    )

    assert result['contexts'][0]['results'] == [
        {
            "metric": "explain_func_metric",
            "metric_params": {},
            "value": "lime explanation",
        }
    ]


def test_run_evaluation_accepts_explain_funcs_by_dataset_and_xai_method(
    monkeypatch,
    clean_metric_registry,
    metric_context
):
    def dataset_lime_explain_func(model, inputs, targets=None):
        return "dataset lime explanation"

    class ExplainFuncMetric(BaseMetric):
        NAME = "explain_func_metric"

        def __init__(self, context, params=None, explain_func=None):
            super().__init__(context, params)
            self.explain_func = explain_func

        def run(self):
            if self.explain_func is None:
                raise ValueError("explain_func is required")
    
            return self.explain_func(
                model=self.context.model,
                inputs=self.context.X_test,
                targets=self.context.y_test,
            )

    METRIC_REGISTRY["explain_func_metric"] = ExplainFuncMetric

    monkeypatch.setattr(
        runner_module,
        "autodiscover_metrics",
        lambda *args, **kwargs: None,
    )

    result = run_evaluation(
        context=metric_context,
        metadata={
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "lime",
        },
        config={"metrics": [{"name": "explain_func_metric"}]},
        report_output_dir=None,
        explain_funcs={
            "dataset": {
                "lime": dataset_lime_explain_func,
            }
        },
    )

    assert result['contexts'][0]['results'] == [
        {
            "metric": "explain_func_metric",
            "metric_params": {},
            "value": "dataset lime explanation",
        }
    ]