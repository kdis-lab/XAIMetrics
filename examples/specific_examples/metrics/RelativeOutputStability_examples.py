# examples/specific_examples/metrics/RelativeOutputStability_examples.py
from pathlib import Path
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.metrics.robustness import RelativeOutputStability
from xai_metrics.runner import run_evaluation

from make_explain_func import make_lime_explain_func

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = RelativeOutputStability(
    context=context,
    params={
        "nr_samples": 20,
        "abs": False,
        "normalise": False
    },
    explain_func = make_lime_explain_func()
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("RelativeOutputStability scores:", scores)
print("Mean RelativeOutputStability:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["RelativeOutputStability"],
    config=config_path,
    report_output_dir=None,
    explain_func = make_lime_explain_func()
)

context_result = results['contexts'][0]
scores = context_result['results'][0]['value']

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("RelativeOutputStability scores:", scores)
print("Mean RelativeOutputStability:", float(np.mean(scores)))
print("Report paths:", results['report_paths'])