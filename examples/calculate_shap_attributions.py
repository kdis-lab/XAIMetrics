# examples/calculate_shap_attributions.py

from pathlib import Path

from xai_metrics.base import EXPLAINER_REGISTRY, register_explainer
from xai_metrics.config.config_controller import default_model_loader
from xai_metrics.explainers.shap import SHAPExplainer
from xai_metrics.runner import run_explanation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

config_path = PROJECT_ROOT / "xai_metrics/config.yaml"
output_dir = PROJECT_ROOT / "examples/attributions"


def main():
    if SHAPExplainer.NAME not in EXPLAINER_REGISTRY:
        register_explainer(SHAPExplainer)

    results = run_explanation(
        config=config_path,
        selected_explainers=["SHAP"],
        model_loader=default_model_loader,
        attribution_output_dir=output_dir,
    )

    print("Attribution paths:")
    for name, path in results["attribution_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
