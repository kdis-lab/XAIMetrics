# examples/specific_examples/metrics/Complexity_examples.py
from pathlib import Path
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.metrics.complexity import Complexity
from xai_metrics.runner import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = Complexity(
    context=context,
    params={
        "normalise": True
    }
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("Complexity scores:", scores)
print("Mean Complexity:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["Complexity"],
    config=config_path,
    report_output_dir=None
)

context_result = results['contexts'][0]
scores = context_result['results'][0]['value']

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("Complexity scores:", scores)
print("Mean Complexity:", float(np.mean(scores)))
print("Report paths:", results['report_paths'])