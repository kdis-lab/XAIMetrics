# examples/calculate_breakdown_attributions.py

from pathlib import Path

from xai_metrics.config.config_controller import default_model_loader
from xai_metrics.runner import run_explanation

# Import necesario para registrar BreakDown si no tienes autodiscovery funcionando
import xai_metrics.explainers.breakdown  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_path = PROJECT_ROOT / "xai_metrics/config.yaml"
output_dir = PROJECT_ROOT / "examples/attributions"


def main():
    results = run_explanation(
        config=config_path,
        selected_explainers=["BreakDown"],
        model_loader=default_model_loader,
        attribution_output_dir=output_dir,
    )

    print("Attribution paths:")
    for name, path in results["attribution_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
