# tests\config\test_build_metrics_context.py
import pandas as pd
import pytest
import torch.nn as nn
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.base import MetricContext

def test_build_metric_context_rejects_non_torch_model(tmp_path):
    X_batch_path = tmp_path / "X_batch.csv"
    y_batch_path = tmp_path / "y_batch.csv"
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame({"x1": [1.0]}, index=[0]).to_csv(X_batch_path)
    pd.DataFrame({"target": [0]}, index=[0]).to_csv(y_batch_path)
    pd.DataFrame({"x1": [0.1]}, index=[0]).to_csv(attributions_path)

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "xai",
                "model_path": str(tmp_path / "model.pkl"),
                "X_batch_path": str(X_batch_path),
                "y_batch_path": str(y_batch_path),
                "attributions_path": str(attributions_path),
            }
        },
        model_loader=lambda model_path: object(),
    )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        controller.build_metric_context()


def test_build_metric_context_loads_model_data_labels_and_attributions(tmp_path):
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame(
        {"x1": [1.0, 2.0], "x2": [3.0, 4.0]},
        index=[10, 11],
    ).to_csv(X_test_path)

    pd.DataFrame(
        {"target": [0, 1]},
        index=[10, 11],
    ).to_csv(y_test_path)

    pd.DataFrame(
        {"x1": [0.1, 0.2], "x2": [0.3, 0.4]},
        index=[10, 11],
    ).to_csv(attributions_path)

    config = {
        "context": {
            "dataset_name": "dataset",
            "model_name": "model",
            "xai_method_name": "xai",
            "model_path": str(tmp_path / "model.pt"),
            "X_test_path": str(X_test_path),
            "y_test_path": str(y_test_path),
            "attributions_path": str(attributions_path),
            "device": "cpu",
        }
    }

    controller = ConfigController(
        config=config,
        model_loader=lambda model_path: nn.Identity()
    )

    with pytest.warns(
        UserWarning,
        match=(
            "Using X_test_path and y_test_path.*"
            "Attribution indexes are expected to match"
        ),
    ):
        context, metadata = controller.build_metric_context()

    assert isinstance(context, MetricContext)
    assert isinstance(context.model, nn.Identity)
    assert context.X_test.shape == (2, 2)
    assert context.y_test.tolist() == [0, 1]
    assert context.observations == [10, 11]
    assert np.array_equal(
        context.attributions,
        np.array([[0.1, 0.3], [0.2, 0.4]])
    )
    assert context.device == "cpu"

    assert metadata == {
        "dataset_name": "dataset",
        "model_name": "model",
        "xai_method_name": "xai"
    }


def test_metric_discovery_uses_test_files_when_batch_files_do_not_exist(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    attributions_dir = tmp_path / "attributions"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
    attribution_dir = attributions_dir / "demo_dataset" / "demo_model" / "demo_xai"

    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)

    pd.DataFrame(
        {"feature_1": [1.0, 2.0]},
        index=[10, 11],
    ).to_csv(dataset_dir / "X_test.csv")

    pd.DataFrame(
        {"target": [0, 1]},
        index=[10, 11],
    ).to_csv(dataset_dir / "y_test.csv")

    pd.DataFrame(
        {"feature_1": [0.1, 0.2]},
        index=[10, 11],
    ).to_csv(attribution_dir / "attributions.csv")

    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir),
                "attributions_dir": str(attributions_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.warns(
        UserWarning,
        match=(
            "Using X_test_path and y_test_path.*"
            "Attribution indexes are expected to match"
        ),
    ):
        contexts = controller.build_metric_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert context.X_test.index.tolist() == [10, 11]
    assert context.y_test.tolist() == [0, 1]
    assert context.observations == [10, 11]

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model",
        "xai_method_name": "demo_xai"
    }


def test_metric_discovery_prioritises_batch_over_test(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    attributions_dir = tmp_path / "attributions"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
    attribution_dir = attributions_dir / "demo_dataset" / "demo_model" / "demo_xai"

    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)

    pd.DataFrame(
        {"feature_1": [1.0]},
        index=[20],
    ).to_csv(dataset_dir / "X_batch.csv")

    pd.DataFrame(
        {"target": [1]},
        index=[20],
    ).to_csv(dataset_dir / "y_batch.csv")

    pd.DataFrame(
        {"feature_1": [99.0]},
        index=[99],
    ).to_csv(dataset_dir / "X_test.csv")

    pd.DataFrame(
        {"target": [9]},
        index=[99],
    ).to_csv(dataset_dir / "y_test.csv")

    pd.DataFrame(
        {"feature_1": [0.1]},
        index=[20],
    ).to_csv(attribution_dir / "attributions.csv")

    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir),
                "attributions_dir": str(attributions_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    context, _ = controller.build_metric_contexts()[0]

    assert context.X_test.index.tolist() == [20]
    assert context.X_test.iloc[:, 0].tolist() == [1.0]
    assert context.y_test.tolist() == [1]
    assert context.observations == [20]


@pytest.mark.parametrize(
    ("context_fields", "expected_message"),
    [
        (
            {
                "X_batch_path": "X_batch.csv",
            },
            "X_batch_path and y_batch_path",
        ),
        (
            {
                "y_batch_path": "y_batch.csv",
            },
            "X_batch_path and y_batch_path",
        ),
        (
            {
                "X_test_path": "X_test.csv",
            },
            "X_test_path and y_test_path",
        ),
        (
            {
                "y_test_path": "y_test.csv",
            },
            "X_test_path and y_test_path",
        ),
    ],
)
def test_explicit_metric_context_requires_complete_input_target_pair(
    tmp_path,
    context_fields,
    expected_message
):
    controller = ConfigController(
        config={
            "context": {
                 "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "xai_method_name": "demo_xai",
                "model_path": str(tmp_path / "model.pt"),
                "attributions_path": str(tmp_path / "attributions.csv"),
                **{
                    name: str(tmp_path / path)
                    for name, path in context_fields.items()
                }
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.raises(ValueError, match=expected_message):
        controller.build_metric_context()


@pytest.mark.parametrize(
    ("feature_filename", "target_filename", "expected_message"),
    [
        (
            "X_batch.csv",
            None,
            "X_batch.csv and y_batch.csv",
        ),
        (
            None,
            "y_batch.csv",
            "X_batch.csv and y_batch.csv",
        ),
        (
            "X_test.csv",
            None,
            "X_test.csv and y_test.csv",
        ),
        (
            None,
            "y_test.csv",
            "X_test.csv and y_test.csv",
        ),
    ],
)
def test_metric_discovery_requires_complete_input_target_pair(
    tmp_path,
    feature_filename,
    target_filename,
    expected_message
):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    attributions_dir = tmp_path / "attributions"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
    attribution_dir = attributions_dir / "demo_dataset" / "demo_model" / "demo_xai"

    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)

    if feature_filename is not None:
        pd.DataFrame(
            {"feature_1": [1.0]},
            index=[10],
        ).to_csv(dataset_dir / feature_filename)

    if target_filename is not None:
        pd.DataFrame(
            {"target": [0]},
            index=[10],
        ).to_csv(dataset_dir / target_filename)

    pd.DataFrame(
        {"feature_1": [0.1]},
        index=[10],
    ).to_csv(attribution_dir / "attributions.csv")

    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir),
                "attributions_dir": str(attributions_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.raises(ValueError, match=expected_message):
        controller.build_metric_contexts()


def test_metric_discovery_raises_error_when_no_batch_or_test_exists(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    attributions_dir = tmp_path / "attributions"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
    attribution_dir = attributions_dir / "demo_dataset" / "demo_model" / "demo_xai"

    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)

    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir),
                "attributions_dir": str(attributions_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.raises(ValueError, match="No MetricContext configs found"):
        controller.build_metric_contexts()


def test_explicit_metric_context_raises_error_when_no_batch_or_test_exists(tmp_path):
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame(
        {"feature_1": [0.1]},
        index=[10],
    ).to_csv(attributions_path)

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "xai_method_name": "xai",
                "model_path": str(tmp_path / "model.pt"),
                "attributions_path": str(attributions_path)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.raises(ValueError, match="X_batch_path or X_test_path"):
        controller.build_metric_context()