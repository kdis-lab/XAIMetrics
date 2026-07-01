# tests/test_registry.py
import pytest

from xai_metrics.base import (
    register_metric,
    list_metrics,
    build_metrics_from_config
)

def test_register_metric_adds_metric_to_registry(clean_metric_registry, dummy_metric_class):
    register_metric(dummy_metric_class)

    assert "dummy" in list_metrics()
    assert list_metrics() == ["dummy"]


def test_register_metric_rejects_duplicates(clean_metric_registry, dummy_metric_class):
    register_metric(dummy_metric_class)

    with pytest.raises(ValueError):
        register_metric(dummy_metric_class)


def test_build_metrics_from_config_instantiates_registered_metric(
    clean_metric_registry,
    metric_context,
    dummy_metric_class
):
    register_metric(dummy_metric_class)

    metrics = build_metrics_from_config(
        [{"name": "dummy", "params": {"value": 3.0}}],
        context=metric_context
    )

    assert len(metrics) == 1
    assert isinstance(metrics[0], dummy_metric_class)
    assert metrics[0].run() == 3.0


def test_build_metrics_from_config_rejects_unknown_metric(
    clean_metric_registry,
    metric_context
):
    with pytest.raises(ValueError):
        build_metrics_from_config(
            [{"name": "missing_metric"}],
            context=metric_context
        )