# examples/specific_examples/metrics/NonSensitivity_examples.py
from pathlib import Path
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.metrics.fidelity.soundness import NonSensitivity
from xai_metrics.runner import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = NonSensitivity(
    context=context,
    params={
        "eps": 1e-5,
        "features_in_step": 1,
        "abs": True,
        "normalise": True,
        "perturb_baseline": "black"
    }
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("NonSensitivity scores:", scores)
print("Mean NonSensitivity:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["NonSensitivity"],
    config=config_path,
    report_output_dir=None
)

context_result = results['contexts'][0]
scores = context_result['results'][0]['value']

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("NonSensitivity scores:", scores)
print("Mean NonSensitivity:", float(np.mean(scores)))
print("Report paths:", results['report_paths'])