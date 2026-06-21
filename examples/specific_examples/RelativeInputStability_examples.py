# examples/specific_examples/RelativeInputStability_examples.py
from pathlib import Path
import numpy as np
import pandas as pd

from xai_metrics.config import ConfigController
from xai_metrics.metrics.robustness import RelativeInputStability
from xai_metrics.runner import run_evaluation

from make_explain_func import make_lime_explain_func

PROJECT_ROOT = Path(__file__).resolve().parents[2]
X_train_path = PROJECT_ROOT / "examples/data/hydraulic/X_train.csv"
X_train = pd.read_csv(X_train_path, index_col=0)

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = RelativeInputStability(
    context=context,
    params={
        "nr_samples": 200,
        "abs": False,
        "normalise": False
    },
    explain_func = make_lime_explain_func(X_train)
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("RelativeInputStability scores:", scores)
print("Mean RelativeInputStability:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["RelativeInputStability"],
    config=config_path,
    report_output_dir=None,
    explain_func = make_lime_explain_func(X_train)
)

context_result = results["contexts"][0]
scores = context_result["results"]["RelativeInputStability"]

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result["metadata"])
print("RelativeInputStability scores:", scores)
print("Mean RelativeInputStability:", float(np.mean(scores)))
print("Report paths:", results["report_paths"])