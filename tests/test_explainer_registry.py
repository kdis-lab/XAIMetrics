# tests/test_explainer_registry.py
import pytest

from xai_metrics.base import (
    register_explainer,
    list_explainers,
    build_explainers_from_config
)

def test_register_explainer_adds_explainer_to_registry(
    clean_explainer_registry,
    dummy_explainer_class
):
    register_explainer(dummy_explainer_class)

    assert "dummy_xai" in list_explainers()
    assert list_explainers() == ['dummy_xai']


def test_register_explainer_rejects_duplicates(
    clean_explainer_registry,
    dummy_explainer_class
):
    register_explainer(dummy_explainer_class)

    with pytest.raises(ValueError):
        register_explainer(dummy_explainer_class)


def test_build_explainers_from_config_instantiates_registered_explainer(
    clean_explainer_registry,
    explainer_context,
    dummy_explainer_class
):
    register_explainer(dummy_explainer_class)

    explainers = build_explainers_from_config(
        [{"name": "dummy_xai", "params": {"seed": 1}}],
        context=explainer_context
    )

    assert len(explainers) == 1
    assert isinstance(explainers[0], dummy_explainer_class)
    assert explainers[0].params == {"seed": 1}


def test_build_explainers_from_config_rejects_unknown_explainer(
    clean_explainer_registry,
    explainer_context
):
    with pytest.raises(ValueError):
        build_explainers_from_config(
            [{"name": "missing_xai"}],
            context=explainer_context
        )