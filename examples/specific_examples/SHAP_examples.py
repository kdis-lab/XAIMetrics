# examples/specific_examples/SHAP_examples.py
from pathlib import Path

import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.explainers.shap import SHAPExplainer
from xai_metrics.runner import run_explanation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

# direct class use
context, metadata = ConfigController(config=config_path).build_explainers_context()
explainer = SHAPExplainer(
    context=context,
    params={
        "mode": "classification",
        "algorithm": "auto",
        "max_background_samples": 100,
        "output_index": 1,
        "random_state": 42,
    },
)

attributions = explainer.explain(
    model=context.model,
    inputs=context.X_batch,
    targets=context.y_batch,
)

print("\nDirect class usage")
print("------------------")
print("Metadata:", metadata)
print("Attributions shape:", attributions.shape)
print("First observation attributions:", attributions[0])
print("Mean absolute attribution:", float(np.mean(np.abs(attributions))))

# run_explanation use
results = run_explanation(
    selected_explainers=["SHAP"],
    config=config_path,
    attribution_output_dir=None,
)

context_result = results['contexts'][0]
runner_attributions = context_result['attributions']['SHAP']

print("\nrun_explanation usage")
print("---------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("Attributions shape:", runner_attributions.shape)
print("First observation attributions:", runner_attributions[0])
print("Mean absolute attribution:", float(np.mean(np.abs(runner_attributions))))
print("Attribution paths:", results['attribution_paths'])
