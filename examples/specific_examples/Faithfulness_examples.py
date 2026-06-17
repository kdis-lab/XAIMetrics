# examples/specific_examples/Faithfulness_examples.py
from pathlib import Path
import numpy as np

from xai_metrics.config import ConfigController
from xai_metrics.metrics.faithfulness import Faithfulness
from xai_metrics.runner import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# direct class use
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_metric_context()
metric = Faithfulness(
    context=context,
    params={
        "base_strategy": "mean"
    }
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("Faithfulness scores:", scores)
print("Mean Faithfulness:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["Faithfulness"],
    config=config_path,
    report_output_dir=None
)

context_result = results["contexts"][0]
scores = context_result["results"]["Faithfulness"]

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result["metadata"])
print("Faithfulness scores:", scores)
print("Mean Faithfulness:", float(np.mean(scores)))
print("Report paths:", results["report_paths"])