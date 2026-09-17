# examples/specific_examples/metrics/MeGe_examples.py
from pathlib import Path
import numpy as np
from copy import deepcopy

from xai_metrics.config import ConfigController
from xai_metrics.metrics.robustness import MeGe
from xai_metrics.runner import run_evaluation

from make_explain_func import make_lime_explain_func

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def make_training_func(base_model):
    def training_func(X_train, y_train, X_holdout, y_holdout):
        trained_wrapped = deepcopy(base_model)
        estimator = trained_wrapped.model
        estimator.fit(X_train)
        return trained_wrapped
    return training_func

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = MeGe(
    context=context,
    params={
        "k_splits": 3
    },
    explain_func = make_lime_explain_func(),
    training_func = make_training_func(context.model)
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("MeGe scores:", scores)
print("Mean MeGe:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["MeGe"],
    config=config_path,
    report_output_dir=None,
    explain_func = make_lime_explain_func(),
    training_func = make_training_func(context.model)
)

context_result = results['contexts'][0]
scores = context_result['results'][0]['value']

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result['metadata'])
print("MeGe scores:", scores)
print("Mean MeGe:", float(np.mean(scores)))
print("Report paths:", results['report_paths'])