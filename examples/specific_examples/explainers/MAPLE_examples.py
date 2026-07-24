# examples/specific_examples/MAPLE_examples.py
from pathlib import Path

import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.explainers.maple import MAPLEExplainer
from xai_metrics.runner import run_explanation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

# direct class use
context, metadata = ConfigController(config=config_path).build_explainers_context()
explainer = MAPLEExplainer(
    context=context,
    params={
        "mode": "classification",
        "output_index": 1,
        "fe_type": "rf",
        "n_estimators": 200,
        "max_features": 0.5,
        "min_samples_leaf": 10,
        "regularization": 0.001,
        "validation_size": 0.2,
        "random_state": 42,
        "n_jobs": -1,
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
    selected_explainers=["MAPLE"],
    config=config_path,
    attribution_output_dir=None,
)

context_result = results["contexts"][0]
runner_attributions = context_result["attributions"]["MAPLE"]

print("\nrun_explanation usage")
print("---------------------")
print("Config file:", config_path)
print("Metadata:", context_result["metadata"])
print("Attributions shape:", runner_attributions.shape)
print("First observation attributions:", runner_attributions[0])
print("Mean absolute attribution:", float(np.mean(np.abs(runner_attributions))))
print("Attribution paths:", results["attribution_paths"])
