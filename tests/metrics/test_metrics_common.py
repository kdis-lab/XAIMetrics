# tests/metrics_tests/test_metrics_common.py
from dataclasses import replace
import pytest
import numpy as np

from xai_metrics.base import MetricSkipped

from xai_metrics.metrics.faithfulness import (
    Consistency,
    Faithfulness,
    FaithfulnessEstimate,
    Monotonicity,
    MonotonicityMetric,
    MonotonicityCorrelation,
    SensitivityN,
    Sufficiency
)
import xai_metrics.metrics.faithfulness.consistency as consistency_module
import xai_metrics.metrics.faithfulness.faithfulness_estimate as estimate_module
import xai_metrics.metrics.faithfulness.monotonicity as monotonicity_module
import xai_metrics.metrics.faithfulness.monotonicity_correlation as correlation_module
import xai_metrics.metrics.faithfulness.sensitivity_n as sensitivity_n_module
import xai_metrics.metrics.faithfulness.sufficiency as sufficiency_module

from xai_metrics.metrics.robustness import (
    AverageStability,
    LocalLipschitzEstimate,
    MaxSensitivity,
    MeGe,
    RelativeInputStability,
    RelativeOutputStability
)
import xai_metrics.metrics.robustness.max_sensitivity as max_sensitivity_module
import xai_metrics.metrics.robustness.local_lipschitz_estimate as lipschitz_module
import xai_metrics.metrics.robustness.relative_input_stability as ris_module
import xai_metrics.metrics.robustness.relative_output_stability as ros_module

from xai_metrics.metrics.sensitivity import (
    AvgSensitivity,
    RandomLogit,
    ModelRandomization
)
import xai_metrics.metrics.sensitivity.avg_sensitivity as avg_sensitivity_module

from xai_metrics.metrics.fidelity.completeness import (
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

@pytest.mark.parametrize(
    ("metric_class", "module", "quantus_name", "requires_explainer"),
    [
        (Completeness, completeness_module, "Completeness", False),
        (NonSensitivity, non_sensitivity_module, "NonSensitivity", False),
        (AvgSensitivity, avg_sensitivity_module, "AvgSensitivity", True),
        (MaxSensitivity, max_sensitivity_module, "MaxSensitivity", True),
        (
            LocalLipschitzEstimate,
            lipschitz_module,
            "LocalLipschitzEstimate",
            True,
        ),
        (
            RelativeInputStability,
            ris_module,
            "RelativeInputStability",
            True,
        ),
        (
            RelativeOutputStability,
            ros_module,
            "RelativeOutputStability",
            True,
        ),
        (Consistency, consistency_module, "Consistency", False),
        (
            FaithfulnessEstimate,
            estimate_module,
            "FaithfulnessEstimate",
            False,
        ),
        (Monotonicity, monotonicity_module, "Monotonicity", False),
        (
            MonotonicityCorrelation,
            correlation_module,
            "MonotonicityCorrelation",
            False,
        ),
        (SensitivityN, sensitivity_n_module, "SensitivityN", False),
        (Sufficiency, sufficiency_module, "Sufficiency", False),
    ],
)
def test_negative_attributions_with_abs_false_skip_metric(
    monkeypatch,
    context,
    explain_func,
    metric_class,
    module,
    quantus_name,
    requires_explainer
):
    context.attributions[:] = -1.0

    class QuantusMustNotRun:
        def __init__(self, **kwargs):
            pytest.fail("Quantus no debería construirse")

    monkeypatch.setattr(module.quantus, quantus_name, QuantusMustNotRun)

    if requires_explainer:
        metric = metric_class(
            context,
            explain_func,
            {"abs": False}
        )
    else:
        metric = metric_class(
            context,
            {"abs": False}
        )

    with pytest.raises(MetricSkipped, match="all attributions are negative"):
        metric.run()


@pytest.mark.parametrize(
    "metric_class",
    [
        AvgSensitivity,
        MaxSensitivity,
        LocalLipschitzEstimate,
        RelativeInputStability,
        RelativeOutputStability
    ]
)
def test_metrics_requiring_explainer_reject_none(context, metric_class):
    with pytest.raises(ValueError, match="requires 'explain_func'"):
        metric_class(context, None)


@pytest.mark.parametrize(
    ("metric_class", "strategy", "expected"),
    [
        (Faithfulness, "mean", [4.0, 5.0, 6.0]),
        (Faithfulness, "median", [4.0, 5.0, 6.0]),
        (Faithfulness, "zero", [0.0, 0.0, 0.0]),
        (MonotonicityMetric, "mean", [4.0, 5.0, 6.0]),
        (MonotonicityMetric, "median", [4.0, 5.0, 6.0]),
        (MonotonicityMetric, "zero", [0.0, 0.0, 0.0]),
    ],
)
def test_aix360_baseline_strategies(context, metric_class, strategy, expected):
    result = metric_class._resolve_base(
        context.X_test,
        base_strategy=strategy
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "metric_class",
    [Faithfulness, MonotonicityMetric],
)
def test_aix360_unknown_baseline_strategy_raises(context, metric_class):
    with pytest.raises(ValueError, match="Unknown base_strategy"):
        metric_class._resolve_base(
            context.X_test,
            base_strategy="unsupported"
        )


def test_safe_pearson_returns_zero_for_constant_inputs(context):
    metric = FaithfulnessEstimate(context)

    result = metric._safe_pearson(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0]
    )

    assert result == 0.0


def test_safe_spearman_returns_zero_for_constant_inputs(context):
    metric = MonotonicityCorrelation(context)

    result = metric._safe_spearman(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0]
    )

    assert result == 0.0


@pytest.mark.parametrize(
    "metric_factory",
    [
        pytest.param(
            lambda ctx, explain: MuFidelity(ctx, {"nb_samples": 2}),
            id="mufidelity",
        ),
        pytest.param(
            lambda ctx, explain: Deletion(ctx, {"steps": 2}),
            id="deletion",
        ),
        pytest.param(
            lambda ctx, explain: Insertion(ctx, {"steps": 2}),
            id="insertion",
        ),
        pytest.param(
            lambda ctx, explain: AverageDrop(ctx),
            id="average_drop",
        ),
        pytest.param(
            lambda ctx, explain: AverageIncrease(ctx),
            id="average_increase",
        ),
        pytest.param(
            lambda ctx, explain: AverageGain(ctx),
            id="average_gain",
        ),
        pytest.param(
            lambda ctx, explain: RandomLogit(
                ctx,
                explain,
                {"num_classes": 2},
            ),
            id="random_logit",
        ),
        pytest.param(
            lambda ctx, explain: ModelRandomization(ctx, explain),
            id="model_randomization",
        ),
        pytest.param(
            lambda ctx, explain: AverageStability(ctx, explain),
            id="average_stability",
        ),
        pytest.param(
            lambda ctx, explain: MeGe(
                ctx,
                lambda X_train, y_train, X_holdout, y_holdout: ctx.model,
                explain,
            ),
            id="mege",
        ),
    ],
)
def test_new_metrics_skip_when_no_observations(
    context,
    explain_func,
    metric_factory
):
    empty_context = replace(
        context,
        observations=[],
        attributions=np.empty((0, 3), dtype=np.float32)
    )

    metric = metric_factory(empty_context, explain_func)

    with pytest.raises(MetricSkipped, match="no observations were selected"):
        metric.run()


@pytest.mark.parametrize(
    "metric_factory",
    [
        pytest.param(
            lambda ctx, explain: MuFidelity(ctx, {"nb_samples": 2}),
            id="mufidelity",
        ),
        pytest.param(
            lambda ctx, explain: Deletion(ctx, {"steps": 2}),
            id="deletion",
        ),
        pytest.param(
            lambda ctx, explain: Insertion(ctx, {"steps": 2}),
            id="insertion",
        ),
        pytest.param(
            lambda ctx, explain: AverageDrop(ctx),
            id="average_drop",
        ),
        pytest.param(
            lambda ctx, explain: AverageIncrease(ctx),
            id="average_increase",
        ),
        pytest.param(
            lambda ctx, explain: AverageGain(ctx),
            id="average_gain",
        ),
        pytest.param(
            lambda ctx, explain: RandomLogit(
                ctx,
                explain,
                {"num_classes": 2},
            ),
            id="random_logit",
        ),
        pytest.param(
            lambda ctx, explain: ModelRandomization(ctx, explain),
            id="model_randomization",
        ),
        pytest.param(
            lambda ctx, explain: AverageStability(ctx, explain),
            id="average_stability",
        ),
    ],
)
def test_new_metrics_reject_misaligned_attributions(
    context,
    explain_func,
    metric_factory
):
    invalid_context = replace(
        context,
        attributions=context.attributions[:1]
    )

    metric = metric_factory(invalid_context, explain_func)

    with pytest.raises(ValueError, match="explanations must match"):
        metric.run()