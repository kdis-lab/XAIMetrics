import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import xai_metrics.explainers as explainers_pkg
from xai_metrics.base import ExplainerContext, build_explainers_from_config
from xai_metrics.config import ConfigController
from xai_metrics.explainers import autodiscover_explainers
from xai_metrics.runner import run_evaluation


CONFIG_PATH = Path("xai_metrics/config.yaml")
DATASETS_DIR = Path("experiments/datasets")
MODELS_DIR = Path("experiments/models")
ATTRIBUTIONS_DIR = Path("experiments/attributions")
REPORTS_DIR = Path("experiments/results/reports")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(config_path: Path, device: str | None) -> dict[str, Any]:
    controller = ConfigController(config=config_path)
    config = dict(controller.config)
    context = dict(config.get("context") or {})

    context["datasets_dir"] = str(DATASETS_DIR)
    context["models_dir"] = str(MODELS_DIR)
    context["attributions_dir"] = str(ATTRIBUTIONS_DIR)

    if device is not None:
        context["device"] = device

    config["context"] = context
    return config


def build_explain_funcs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    autodiscover_explainers(explainers_pkg)

    controller = ConfigController(config=config)
    explain_funcs = {}

    for ctx_cfg in controller._iter_explainer_context_configs():
        dataset_name = ctx_cfg["dataset_name"]

        if dataset_name in explain_funcs:
            continue

        X_background = pd.read_csv(ctx_cfg["X_background_path"], index_col=0)
        y_background = None

        if "y_background_path" in ctx_cfg:
            y_background = pd.read_csv(
                ctx_cfg["y_background_path"],
                index_col=0,
            ).iloc[:, 0]

        explainer_context = ExplainerContext(
            X_background=X_background,
            y_background=y_background,
            device=ctx_cfg.get("device"),
        )

        explainers = build_explainers_from_config(
            controller.get_explainers_config(),
            explainer_context,
        )

        explain_funcs[dataset_name] = {
            explainer.NAME: explainer.as_explain_func()
            for explainer in explainers
        }

    return explain_funcs


def run_eval(config_path: Path, output_dir: Path, seed: int, device: str | None) -> None:
    set_seed(seed)

    config = load_config(
        config_path=config_path,
        device=device,
    )
    explain_funcs = build_explain_funcs(config)

    print("Running evaluation")
    print(f"  Config      : {config_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Seed        : {seed}")
    print(f"  Device      : {config['context'].get('device')}")
    print(f"  Datasets    : {sorted(explain_funcs)}")

    results = run_evaluation(
        config=config,
        explain_funcs=explain_funcs,
        report_output_dir=output_dir,
    )

    print("Evaluation finished")
    print(f"  Contexts    : {len(results['contexts'])}")
    print(f"  Reports     : {results['report_paths']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the XAI metric evaluation experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory where reports are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
