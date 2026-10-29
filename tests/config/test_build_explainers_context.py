# tests\config\test_build_explainers_context.py
import pandas as pd
import torch.nn as nn
import pytest

from conftest import create_explainer_data

from xai_metrics.config import ConfigController


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


def test_explainer_discovery_raises_error_when_no_background_or_train_exists(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"
   
    paths = create_explainer_data(dataset_dir)
    paths['X_background_path'].unlink()
    paths['y_background_path'].unlink()

    model_dir.mkdir(parents=True)
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


def test_explicit_explainer_context_uses_train_paths_when_background_paths_are_absent(tmp_path):
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
    assert context.y_background is not None
    assert context.y_background.tolist() == [0, 1]
    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model",
    }


def test_explicit_explainer_context_raises_error_when_no_background_or_train_exists(tmp_path):
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


def test_explicit_explainer_context_prioritises_background_over_train(tmp_path):
    background_paths = create_explainer_data(
        tmp_path / "explicit_data",
        background_source="background"
    )
    train_paths = create_explainer_data(
        tmp_path / "explicit_data",
        background_source="train"
    )

    pd.DataFrame(
        {"feature_1": [99.0], "feature_2": [100.0]},
        index=[99]
    ).to_csv(train_paths['X_background_path'])

    pd.DataFrame(
        {"target": [9]},
        index=[99]
    ).to_csv(train_paths['y_background_path'])

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(tmp_path / "model.pt"),
                "X_background_path": str(
                    background_paths["X_background_path"]
                ),
                "y_background_path": str(
                    background_paths["y_background_path"]
                ),
                "X_train_path": str(train_paths["X_background_path"]),
                "y_train_path": str(train_paths["y_background_path"]),
                "X_batch_path": str(background_paths["X_batch_path"]),
                "y_batch_path": str(background_paths["y_batch_path"])
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    with pytest.warns(UserWarning) as captured_warnings:
        context, metadata = controller.build_explainers_context()

    warning_messages = [str(warning.message) for warning in captured_warnings]

    assert (
        "Both X_background_path and X_train_path are configured. "
        "Using X_background_path as explainer background data."
        in warning_messages
    )
    assert (
        "Both y_background_path and y_train_path are configured. "
        "Using y_background_path as explainer background labels."
        in warning_messages
    )

    assert context.X_background.index.tolist() == [10, 11]
    assert context.X_background.iloc[:, 0].tolist() == [1.0, 2.0]
    assert context.y_background is not None
    assert context.y_background.tolist() == [0, 1]

    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model"
    }


def test_explainer_discovery_uses_test_when_batch_files_do_not_exist(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"

    paths = create_explainer_data(dataset_dir, batch_source="test")
    model_dir.mkdir(parents=True)
    (model_dir / "model.pt").touch()

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir)
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    context, _ = controller.build_explainers_contexts()[0]

    assert paths['X_batch_path'].name == "X_test.csv"
    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]


def test_explicit_explainer_context_uses_test_paths_when_batch_paths_are_absent(tmp_path):
    paths = create_explainer_data(
        tmp_path / "explicit_data",
        batch_source="test"
    )

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(tmp_path / "model.pt"),
                "X_background_path": str(paths["X_background_path"]),
                "y_background_path": str(paths["y_background_path"]),
                "X_test_path": str(paths["X_batch_path"]),
                "y_test_path": str(paths["y_batch_path"])
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    context, _ = controller.build_explainers_context()

    assert paths['X_batch_path'].name == "X_test.csv"
    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]
    assert context.y_batch is not None
    assert context.y_batch.tolist() == [1]


def test_explicit_explainer_context_allows_no_batch_or_test_paths(tmp_path):
    paths = create_explainer_data(tmp_path / "explicit_data")

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "model_name": "demo_model",
                "model_path": str(tmp_path / "model.pt"),
                "X_background_path": str(paths["X_background_path"]),
                "y_background_path": str(paths["y_background_path"])
            }
        },
        model_loader=lambda path: nn.Identity()
    )

    context, metadata = controller.build_explainers_context()

    assert context.X_batch is None
    assert context.y_batch is None

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model"
    }


def test_explainer_discovery_keeps_dataset_without_batch_or_test(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"

    paths = create_explainer_data(dataset_dir)
    paths['X_batch_path'].unlink()
    paths['y_batch_path'].unlink()

    model_dir.mkdir(parents=True)
    (model_dir / "model.pt").touch()

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

    assert context.X_batch is None
    assert context.y_batch is None

    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model",
    }


def test_explicit_explainer_context_allows_missing_model(tmp_path):
    paths = create_explainer_data(tmp_path / "explicit_data")

    controller = ConfigController(
        config={
            "context": {
                "dataset_name": "demo_dataset",
                "X_background_path": str(paths["X_background_path"]),
                "y_background_path": str(paths["y_background_path"]),
                "X_batch_path": str(paths["X_batch_path"]),
                "y_batch_path": str(paths["y_batch_path"])
            }
        }
    )

    context, metadata = controller.build_explainers_context()

    assert context.model is None
    assert context.X_background.index.tolist() == [10, 11]
    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]

    assert metadata == {
        "dataset_name": "demo_dataset"
    }


def test_explainer_discovery_allows_model_directory_without_model_file(tmp_path):
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    dataset_dir = datasets_dir / "demo_dataset"
    model_dir = models_dir / "demo_dataset" / "demo_model"

    create_explainer_data(dataset_dir)
    model_dir.mkdir(parents=True)

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir),
                "models_dir": str(models_dir)
            }
        }
    )

    contexts = controller.build_explainers_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert context.model is None
    assert metadata == {
        "dataset_name": "demo_dataset",
        "model_name": "demo_model"
    }


def test_explainer_discovery_allows_no_model_configuration(tmp_path):
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "demo_dataset"

    create_explainer_data(dataset_dir)

    controller = ConfigController(
        config={
            "context": {
                "datasets_dir": str(datasets_dir)
            }
        }
    )

    contexts = controller.build_explainers_contexts()

    assert len(contexts) == 1

    context, metadata = contexts[0]

    assert context.model is None
    assert context.X_background.index.tolist() == [10, 11]
    assert context.X_batch is not None
    assert context.X_batch.index.tolist() == [20]

    assert metadata == {
        "dataset_name": "demo_dataset",
    }