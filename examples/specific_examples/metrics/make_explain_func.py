# examples/specific_examples/make_explain_func.py
from pathlib import Path

from xai_metrics.config import ConfigController
from xai_metrics.explainers.lime import LIMEExplainer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

def make_lime_explain_func():
    context, _ = ConfigController(config_path).build_explainers_context()

    params = {
        "mode": "classification",
        "random_state": 42,
        "num_samples": 5000,
        "labels": [1],
        "distance_metric": "euclidean"
    }

    return LIMEExplainer(context=context, params=params).as_explain_func()