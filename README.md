# XAI-metrics

A Python library for evaluationa ttribution-based explainability methods in
machine learning models. It supports the evaluation and comparison of local XAI
methods such as LIME and SHAP, producing both aggregated and observation-level
metric reports.

The package is designed for explanation methods where each attributio value
represents the importance of an input feature for a specific observation.

## Features

- Centralized metric execution through `run_evaluation`.
- Attribution generation through `run_explanation`.
- Declarative YAML-based configuration.
- Automatic discovery of registered metrics and explainers.
- Aggregated and observation-level reprots in CSV and JSON formats.
- Support for PyTorch models and models exposing interfaces such as `predict`
  or `predict_proba`, depending on the metric.
- Complexity, fidelity, faithfulness, robustness, and sensitivity metrics.

## Requirements

- Python `>=3.13`
- Dependencies declared in `pyproject.toml`

## Installation

Clone the repository and install the project in editable mode:

```bash
git clone https://github.com/mayumar/XAI_metrics.git
cd XAI_metrics
uv sync
```

Alternatively, using `pip`:
```bash
pip install -e
```

## Quick start

The main function for evaluating existing explanations is `run_evaluation`.

```python
from xai_metrics.runner import run_evaluation

results = run_evaluation(
    config="xai_metrics/config.yaml",
    report_output_dir="results/reports",
)
```

The returned dictionary contains:

- `contexts`: detailed output for every evaluated contxt.
- `reports`: aggregated tables grouped by dataset and model.
- `observation_reports`: metric results for individual observations.
- `report_paths`: paths of the generated CSV and JSON files.

Specific metrics can be selected:

```python
results = run_evaluation(
    config="xai_metrics/config.yaml",
    selected_metrics=[
        "Complexity",
        "Sparseness",
        "MuFidelity",
        "Deletion",
        "Insertion",
    ],
    report_output_dir="results/reports",
)
```

## Using a manually created context

A `MetricContext` can be created directly when data should not be leaded from
the paths declared in the configuraiton file:

```python
import numpy as np

from xai_metrics.base import MetricContext
from xai_metrics.runner import run_evaluation

context = MetricContext(
    model=model,
    X_test=X_test,
    y_test=y_test,
    observations=[10, 20, 30],
    attributions=np.asarray(attributions),
    device="cpu",
)

results = run_evaluation(
    context=context,
    metadata={
        "dataset_name": "MetroPT3",
        "model_name": "IForest",
        "xai_method_name": "LIME",
    },
    selected_metrics=["Complexity", "Sparseness"],
    config="xai_metrics/config.yaml",
)
```

`observations` must contain indices available in `X_test`and `y_test`.
Additionally, `attributions` must contain one row for every selected
observation, in the same order.

## Configuration

The YAML configuration file mainly contains `context`, `metrics` and
optionally `explainers` sections.

```yaml
context:
  device: "cpu"
  datasets_dir: "prueba/data"
  models_dir: "prueba/results/models"
  attributions_dir: "prueba/results/attributions"

metrics:
  - name: Complexity
    params:
      normalise: true

  - name: Sparseness
    params:
      normalise: true

  - name: Faithfulness
    params:
      base_strategy: mean

  - name: LocalLipschitzEstimate
    params:
      nr_samples: 200
      abs: false
      normalise: true
      perturb_mean: 0.0
      perturb_std: 0.1
```

A single context can also be configured using explicit file paths:

```yaml
context:
  dataset_name: "MetroPT3"
  model_name: "IForest"
  xai_method_name: "LIME"
  device: "cpu"
  model_path: "prueba/results/models/MetroPT3/IForest/MetroPT3_IForest_seed_0.pkl"
  X_test_path: "prueba/data/MetroPT3/X_test_train_norm.csv"
  y_test_path: "prueba/data/MetroPT3/y_test_train.csv"
  attributions_path: "prueba/results/attributions/MetroPT3/IForest/LIME/MetroPT3_IForest_lime_attributions.csv"
```

## Expected data structure

For automatic context discovery, the following structure is recommended:

```text
prueba/
├── data_dir/
│   └── dataset1_name/
│       ├── X_test.csv
│       └── y_test.csv
├── models_dir/
│   └── dataset1_name/
│       └── ML_model1_name/
│           └── model.pkl
└── attributions_dir/
    └── dataset1_name/
        └── ML_model1_name/
            ├── XAI_method1_name/
            │   └── attributions.csv
            └── XAI_method2_name/
                └── attributions.csv
```

Input files must meet the following requirements:

- `X_test`: CSV file with observations as rows and features as columns.
- `y_test`: target labelsaligned by index with `X_test`.
- `attributions`: one row per explained observation and one column per input
  feature.
- Attribution indices must exist in the `X_test` index.
- The number of attribution rows must match the number of evaluated observations.
- The model must be compatible with the selected metrics. By default, the library
  con load `.pkl`, `.pickle`, `.joblib`, `.jl`, `.pt` and `.pth` models.

## Avariable metrics

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Complexity</td>
      <td><a href="xai_metrics/metrics/complexity/complexity_metric.py">Complexity</a></td>
    </tr>
    <tr>
      <td><a href="xai_metrics/metrics/complexity/sparseness.py">Sparseness</a></td>
    </tr>
    <tr>
      <td rowspan="10">Faithfulness</td>
      <td><a href="xai_metrics/metrics/faithfulness/consistency.py">Consistency</a></td>
    </tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/faithfulness.py">Faithfulness</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/faithfulness_estimate.py">FaithfulnessEstimate</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/monotonicity.py">Monotonicity</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/monotonicity_correlation.py">MonotonicityCorrelation</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/monotonicity_metric.py">MonotonicityMetric</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/sensitivity_n.py">SensitivityN</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/faithfulness/sufficiency.py">Sufficiency</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/completeness_metric.py">Completeness</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/soundness/non_sensitivity.py">NonSensitivity</a></td></tr>

    <tr>
      <td rowspan="6">Fidelity</td>
      <td><a href="xai_metrics/metrics/fidelity/completeness/mufidelity.py">MuFidelity</a></td>
    </tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/deletion.py">Deletion</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/insertion.py">Insertion</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/average_drop.py">AverageDrop</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/average_increase.py">AverageIncrease</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/fidelity/completeness/average_gain.py">AverageGain</a></td></tr>

    <tr>
      <td rowspan="7">Robustness</td>
      <td><a href="xai_metrics/metrics/robustness/local_lipschitz_estimate.py">LocalLipschitzEstimate</a></td>
    </tr>
    <tr><td><a href="xai_metrics/metrics/robustness/max_sensitivity.py">MaxSensitivity</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/robustness/relative_input_stability.py">RelativeInputStability</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/robustness/relative_output_stability.py">RelativeOutputStability</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/sensitivity/avg_sensitivity.py">AvgSensitivity</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/robustness/average_stability.py">AverageStability</a></td></tr>
    <tr><td><a href="xai_metrics/metrics/robustness/mege.py">MeGe</a></td></tr>

    <tr>
      <td rowspan="2">Sensitivity</td>
      <td><a href="xai_metrics/metrics/sensitivity/random_logit.py">RandomLogit</a></td>
    </tr>
    <tr><td><a href="xai_metrics/metrics/sensitivity/model_randomization.py">ModelRandomization</a></td></tr>
  </tbody>
</table>

## Metrics with additional requirements

Some metrics require additional information besides precomputed attributions.

| Metric | Additional requirement |
|---|---|
| `MuFidelity`, `Deletion`, `Insertion`, `AverageDrop`, `AverageIncrease`, `AverageGain` | A model that can produce one score per observation. |
| `RandomLogit` | Classification targets and an explanation function. |
| `ModelRandomization` | A PyTorch `torch.nn.Module` model and an explanation function. |
| `AverageStability` | An explanation function to generate attributions for perturbed inputs. |
| `MeGe` | A training function and an explanation function. The number of selected observations must be divisible by `k_splits`. |

Explanation functions can be provided at runtime through `explain_funcs`:

```python
results = run_evaluation(
    config="xai_metrics/config.yaml",
    explain_funcs={
        "lime": lime_explain_func,
        "shap": shap_explain_func,
    },
)
```

An explanation function should follow this interface:

```python
def explain_func(model, inputs, targets=None):
    """Return one attribution array per input observation."""
    return attributions
```

`MeGe` additionally requires a training function:

```python
def training_func(X_train, y_train, X_holdout, y_holdout):
    model = build_model()
    model.fit(X_train, y_train)
    return model
```

It can be supplied when running the evaluation:

```python
results = run_evaluation(
    config="xai_metrics/config.yaml",
    selected_metrics=["MeGe"],
    explain_funcs={"lime": lime_explain_func},
    training_func=training_func,
)
```

## Generating explanations

The library can also generate attributions with the registered explainers:

```python
from xai_metrics.runner import run_explanation

results = run_explanation(
    config="xai_metrics/config.yaml",
    selected_explainers=["LIME", "SHAP"],
    attribution_output_dir="results/attributions",
)
```

## Results and reports

When `report_output_dir` is provided, XAIMetrics creates aggregated reports and
observation-level reports.

```text
results/
└── reports/
    ├── MetroPT3_IForest_xai_metrics_report.csv
    ├── MetroPT3_IForest_xai_metrics_report.json
    └── observations/
        ├── Complexity/
        ├── Sparseness/
        ├── Faithfulness/
        ├── LocalLipschitzEstimate/
        └── ...
```

Aggregated reports facilitate the comparison of multiple XAI methods for the
same dataset and model. Observation-level reports retain the value returned by
each metric for each evaluated explanation.

## Adding a metric

A metric must inherit from `BaseMetric`, define a `NAME`, be registered with
`@register_metric`, and implement `run`.

```python
from xai_metrics.base import BaseMetric, register_metric

@register_metric
class MyMetric(BaseMetric):
    NAME = "MyMetric"

    def run(self):
        X_test = self.context.X_test
        attributions = self.context.attributions
        return float(attributions.mean())
```

The metric must be added under `xai_metrics.metrics` so automatic discovery
can locate it. It can then be configured as follows:

```yaml
metrics:
  - name: MyMetric
    params: {}
```

## Tests

Run the complete test suite from the repository root:

```bash
uv run pytest
```

Run only metric tests:

```bash
uv run pytest tests/metrics
```

## Documentation

Documentation is generated with Sphinx.

From the repository root:

```bash
uv run sphinx-apidoc -f -e -o docs/sphinx/source/api xai_metrics
cd docs/spinx
uv run make html
```

The generated HTML documentation is avaible at:

```text
docs/sphinx/build/html/index.html
```

## License

This project is distributed under the MIT License.
