# examples/specific_examples/metrics/Deletion_examples.py
from pathlib import Path

import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.metrics.fidelity.completeness import Deletion
from xai_metrics.runner import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = Deletion(
    context=context,
    params={
        "baseline_mode": 0.0,
        "steps": 10,
        "max_percentage_perturbed": 1.0,
        "batch_size": 64,
        "activation": "softmax"
    }
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("Deletion scores:", scores)
print("Mean Deletion:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["Deletion"],
    config=config_path,
    report_output_dir=None,
)

context_result = results['contexts'][0]
scores = context_result['results'][0]['value']

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("Deletion scores:", scores)
print("Mean Deletion:", float(np.mean(scores)))
print("Report paths:", results['report_paths'])
