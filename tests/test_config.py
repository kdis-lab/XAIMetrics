# tests/test_config.py
import pandas as pd
import pytest
import torch.nn as nn
import numpy as np

from conftest import create_explainer_data

from xai_metrics.config import ConfigController
from xai_metrics.config.validators import validate_X_y_indexes, validate_observations
from xai_metrics.base import MetricContext

def test_config_controller_loads_config_from_mapping():
    config = {
        "metrics": [
            {"name": "dummy", "params": {"value": 1.0}}
        ]
    }

    controller = ConfigController(config=config)

    assert controller.config == config
    assert controller.get_metrics_config() == config['metrics']


def test_config_controller_loads_config_from_yaml_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "metrics:\n"
            "  - name: dummy\n"
            "    params:\n"
            "      value: 3.0\n"
        ),
        encoding="utf-8"
    )

    controller = ConfigController(config=config_path)

    assert controller.get_metrics_config() == [
        {"name": "dummy", "params": {"value": 3.0}}
    ]


def test_validate_x_y_indexes_reorders_y_to_match_x():
    X_test = pd.DataFrame({"x1": [1.0, 2.0]}, index=[20, 10])
    y_test = pd.Series([0, 1], index=[10, 20])

    X_valid, y_valid = validate_X_y_indexes(X_test, y_test)

    assert X_valid.index.tolist() == [20, 10]
    assert y_valid.index.tolist() == [20, 10]
    assert y_valid.tolist() == [1, 0]


def test_validate_x_y_indexes_rejects_mismatched_indexes():
    X_test = pd.DataFrame({"x1": [1.0]}, index=[1])
    y_test = pd.Series([0], index=[2])

    with pytest.raises(ValueError, match="X_test and y_test must contain the same indexes"):
        validate_X_y_indexes(X_test, y_test)


def test_validate_observations_casts_to_x_test_index_dtype():
    observations = validate_observations(
        observations=["1", "2"],
        X_test_index=pd.Index([1, 2, 3])
    )

    assert observations == [1, 2]


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
            "xai_method_name": "lime",
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
        "xai_method_name": "lime"
    }


def test_build_metric_context_rejects_non_torch_model(tmp_path):
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    attributions_path = tmp_path / "attributions.csv"

    pd.DataFrame({"x1": [1.0]}, index=[0]).to_csv(X_test_path)
    pd.DataFrame({"target": [0]}, index=[0]).to_csv(y_test_path)
    pd.DataFrame({"x1": [0.1]}, index=[0]).to_csv(attributions_path)

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "dataset",
                "model_name": "model",
                "xai_method_name": "lime",
                "model_path": str(tmp_path / "model.pkl"),
                "X_test_path": str(X_test_path),
                "y_test_path": str(y_test_path),
                "attributions_path": str(attributions_path),
            }
        },
        model_loader=lambda model_path: object(),
    )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        controller.build_metric_context()


def test_build_explainers_contexts_uses_discovery_without_attribution_dir(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"

    create_explainer_data(dataset_dir)
    model_dir.mkdir(parents=True)

    model_path = model_dir / "model.pt"
    model_path.touch()

    loaded_paths = []

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir)
            }
        },
        model_loader=lambda path: loaded_paths.append(path) or nn.Identity()
    )

    contexts = controller.build_explainers_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert loaded_paths == [str(model_path)]
    assert isinstance(context.model, nn.Identity)

    assert context.X_background.shape == (2, 2)
    assert context.y_background is not None
    assert context.y_background.tolist() == [0, 1]

    assert context.X_batch is not None
    assert context.X_batch.shape == (1, 2)
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model"
    }


def test_build_explainers_contexts_supports_explicit_config_without_xai_method_name(tmp_path):
    paths = create_explainer_data(tmp_path / "explicit_data")

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(tmp_path / "model.pt"),
                "X_background_path": str(paths['X_background_path']),
                "y_background_path": str(paths['y_background_path']),
                "X_batch_path": str(paths['X_batch_path']),
                "y_batch_path": str(paths['y_batch_path'])
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    contexts = controller.build_explainers_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert isinstance(context.model, nn.Identity)
    assert context.X_background.shape == (2, 2)
    assert context.y_background is not None
    assert context.y_background.tolist() == [0, 1]
    assert context.X_batch is not None
    assert context.X_batch.shape == (1, 2)
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model"
    }


def test_explainer_discovery_uses_train_when_background_files_do_not_exist(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"

    create_explainer_data(dataset_dir, background_source="train")
    model_dir.mkdir(parents=True)

    model_path = model_dir / "model.pt"
    model_path.touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    contexts = controller.build_explainers_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert context.X_background.index.tolist() == [10, 11]
    assert context.y_background is not None
    assert context.y_background.tolist() == [0, 1]
    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model",
    }


def test_explainer_discovery_raises_error_when_no_background_or_train_exists(
    tmp_path,
):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        {"feature_1": [3.0]},
        index=[20],
    ).to_csv(dataset_dir / "X_batch.csv")

    pd.DataFrame(
        {"target": [1]},
        index=[20],
    ).to_csv(dataset_dir / "y_batch.csv")

    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir),
            }
        },
        model_loader=lambda path: nn.Identity(),
    )

    with pytest.raises(
        ValueError,
        match="No ExplainerContext configs found",
    ):
        controller.build_explainers_contexts()


# @pytest.mark.xfail(
#     strict=True,
#     reason="El modo explícito aún solo admite X_background_path.",
# )
def test_explicit_explainer_context_uses_train_paths_when_background_paths_are_absent(
    tmp_path,
):
    X_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"

    pd.DataFrame(
        {"feature_1": [1.0, 2.0]},
        index=[10, 11],
    ).to_csv(X_train_path)

    pd.DataFrame(
        {"target": [0, 1]},
        index=[10, 11],
    ).to_csv(y_train_path)

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(tmp_path / "model.pt"),
                "X_train_path": str(X_train_path),
                "y_train_path": str(y_train_path),
            }
        },
        model_loader=lambda path: nn.Identity(),
    )

    context, metadata = controller.build_explainers_context()

    assert context.X_background.index.tolist() == [10, 11]
    assert context.y_background.tolist() == [0, 1]
    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model",
    }


def test_explicit_explainer_context_raises_error_when_no_background_or_train_exists(
    tmp_path,
):
    X_batch_path = tmp_path / "X_batch.csv"
    y_batch_path = tmp_path / "y_batch.csv"
    model_path = tmp_path / "model.pt"

    pd.DataFrame(
        {"feature_1": [3.0]},
        index=[20],
    ).to_csv(X_batch_path)

    pd.DataFrame(
        {"target": [1]},
        index=[20],
    ).to_csv(y_batch_path)

    model_path.touch()

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(model_path),
                "X_batch_path": str(X_batch_path),
                "y_batch_path": str(y_batch_path),
            }
        },
        model_loader=lambda path: nn.Identity(),
    )

    with pytest.raises(
        ValueError,
        match="X_background_path or X_train_path",
    ):
        controller.build_explainers_contexts()