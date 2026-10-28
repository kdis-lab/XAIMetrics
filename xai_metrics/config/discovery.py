# xai_metrics\config\discovery.py
from pathlib import Path
from itertools import product

from typing import Mapping, Any, List, Dict


def discover_metric_context_configs(ctx_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    datasets_root = Path(ctx_cfg["datasets_dir"]).expanduser().resolve()
    models_root = Path(ctx_cfg["models_dir"]).expanduser().resolve()
    attributions_root = Path(ctx_cfg["attributions_dir"]).expanduser().resolve()

    context_configs = []

    for dataset_path in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        dataset_name = dataset_path.name

        models_dataset_root = models_root / dataset_name
        attribution_dataset_root = attributions_root / dataset_name

        if not models_dataset_root.exists() or not attribution_dataset_root.exists():
            continue

        X_batch_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("x_batch")
        )
        y_batch_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("y_batch")
        )
        X_test_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("x_test")
        )
        y_test_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("y_test")
        )

        if bool(X_batch_files) != bool(y_batch_files):
            raise ValueError(
                "X_batch.csv and y_batch.csv must be present together."
            )

        if bool(X_test_files) != bool(y_test_files):
            raise ValueError(
                "X_test.csv and y_test.csv must be present together."
            )

        using_batch_files = bool(X_batch_files)

        if using_batch_files:
            X_files = X_batch_files
            y_files = y_batch_files
        else:
            X_files = X_test_files
            y_files = y_test_files

        if not X_files:
            continue

        for model_dir in sorted(path for path in models_dataset_root.iterdir() if path.is_dir()):
            model_name = model_dir.name
            attribution_dataset_model_root = attribution_dataset_root / model_name

            if not attribution_dataset_model_root.exists():
                continue

            model_files = sorted(
                path for path in model_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".joblib"}
            )
            
            for xai_method_dir in sorted(
                path for path in attribution_dataset_model_root.iterdir()
                if path.is_dir()
            ):
                xai_method_name = xai_method_dir.name
                attribution_dataset_model_xai_dir = attribution_dataset_model_root / xai_method_name

                attribution_files = sorted(
                    path for path in attribution_dataset_model_xai_dir.rglob("*.csv")
                    if path.is_file()
                )

                for model_path, X_path, y_path, attributions_path in product(
                    model_files,
                    X_files,
                    y_files,
                    attribution_files
                ):
                    context_config = {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "xai_method_name": xai_method_name,
                        "model_path": str(model_path),
                        "attributions_path": str(attributions_path),
                        "device": ctx_cfg.get("device"),
                    }

                    if using_batch_files:
                        context_config["X_batch_path"] = str(X_path)
                        context_config["y_batch_path"] = str(y_path)
                    else:
                        context_config["X_test_path"] = str(X_path)
                        context_config["y_test_path"] = str(y_path)

                    context_configs.append(context_config)

    if not context_configs:
        raise ValueError(
            "No MetricContext configs found from datasets_dir, models_dir and attributions_dir."
        )

    return context_configs


def discover_explainer_context_configs(ctx_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    datasets_root = Path(ctx_cfg['datasets_dir']).expanduser().resolve()
    models_root = Path(ctx_cfg['models_dir']).expanduser().resolve()

    context_configs = []

    for dataset_path in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        dataset_name = dataset_path.name
        models_dataset_root = models_root / dataset_name

        if not models_dataset_root.exists():
            continue

        X_background_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("x_background")
        )
        train = False

        if not X_background_files:
            train = True
            X_background_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("x_train")
            )

        if not X_background_files:
            continue

        if not train:
            y_background_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("y_background")
            )
        else:
            y_background_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("y_train")
            )

        X_batch_files = sorted(
            path for path in dataset_path.glob("*.csv")
            if path.name.lower().startswith("x_batch")
        )
        test = False

        if not X_batch_files:
            test = True
            X_batch_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("x_test")
            )

        if X_batch_files:
            if not test:
                y_batch_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_batch")
                )
            else:
                y_batch_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_test")
                )
        else:
            X_batch_files = [None]
            y_batch_files = []

        for model_dir in sorted(path for path in models_dataset_root.iterdir() if path.is_dir()):
            model_name = model_dir.name

            model_files = sorted(
                path for path in model_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".joblib"}
            )

            for model_path, X_background_path, X_batch_path in product(
                model_files,
                X_background_files,
                X_batch_files
            ):
                base_config = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "X_background_path": str(X_background_path),
                    "device": ctx_cfg.get("device"),
                }

                if y_background_files:
                    base_config["y_background_path"] = str(y_background_files[0])

                if X_batch_path:
                    base_config["X_batch_path"] = str(X_batch_path)

                if y_batch_files:
                    base_config["y_batch_path"] = str(y_batch_files[0])

                context_configs.append(base_config)

    if not context_configs:
        raise ValueError(
            "No ExplainerContext configs found from datasets_dir and models_dir."
        )
    
    return context_configs