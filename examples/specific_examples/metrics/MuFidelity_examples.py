# # examples/specific_examples/metrics/MuFidelity_examples.py
# import os
# import warnings
# from pathlib import Path

# # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# from sklearn.exceptions import InconsistentVersionWarning

# warnings.filterwarnings(
#     "ignore",
#     message="pkg_resources is deprecated as an API",
#     category=UserWarning,
#     module="dalex._global_checks",
# )
# warnings.filterwarnings(
#     "ignore",
#     category=InconsistentVersionWarning,
#     module="sklearn.base",
# )

# import numpy as np

# from xai_metrics.config import ConfigController
# from xai_metrics.metrics.fidelity.completeness import MuFidelity
# from xai_metrics.runner import run_evaluation

# PROJECT_ROOT = Path(__file__).resolve().parents[3]

# # direct class use
# config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

# context, metadata = ConfigController(config=config_path).build_metric_context()
# metric = MuFidelity(
#     context=context,
#     params={
#         "one_hot_targets": True,
#         "num_classes": 2,
#         "nb_samples": 200,
#         "subset_percent": 0.5,
#         "abs": False,
#         "normalise": False,
#         "random_state": 42
#     }
# )

# scores = metric.run()

# print("\nDirect class usage")
# print("------------------")
# print("MuFidelity scores:", scores)
# print("Mean MuFidelity:", float(np.mean(scores)))

# # run_evaluation use
# results = run_evaluation(
#     selected_metrics=["MuFidelity"],
#     config=config_path,
#     metric_params={
#         "MuFidelity": {
#             "one_hot_targets": True,
#             "num_classes": 2,
#             "nb_samples": 200,
#             "subset_percent": 0.5,
#         }
#     },
#     report_output_dir=None,
# )

# context_result = results['contexts'][0]
# scores = context_result['results'][0]['value']

# print("\nrun_evaluation usage")
# print("--------------------")
# print("Config file:", config_path)
# print("Metadata:", context_result['metadata'])
# print("MuFidelity scores:", scores)
# print("Mean MuFidelity:", float(np.mean(scores)))
# print("Report paths:", results['report_paths'])
