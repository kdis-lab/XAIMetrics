API Reference
=============

Base
----

.. automodule:: xai_metrics.base.base_metric
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.base.base_explainer
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.base.metric_registry
   :members:

.. automodule:: xai_metrics.base.explainer_registry
   :members:

Configuration and execution
---------------------------

.. automodule:: xai_metrics.config.config_controller
   :members:

.. automodule:: xai_metrics.runner.runner
   :members:

.. automodule:: xai_metrics.reporting.reporting
   :members:

Explainers
----------

.. automodule:: xai_metrics.explainers.breakdown
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.explainers.lime
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.explainers.maple
   :members:
   :show-inheritance:
   :exclude-members: _MAPLEModel

.. automodule:: xai_metrics.explainers.shap
   :members:
   :show-inheritance:

Metrics
-------

Complexity
^^^^^^^^^^

.. automodule:: xai_metrics.metrics.complexity.complexity_metric
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.complexity.sparseness
   :members:
   :show-inheritance:

Faithfulness
^^^^^^^^^^^^

.. automodule:: xai_metrics.metrics.faithfulness.consistency
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.faithfulness
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.faithfulness_estimate
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.monotonicity
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.monotonicity_correlation
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.monotonicity_metric
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.sensitivity_n
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.faithfulness.sufficiency
   :members:
   :show-inheritance:

Fidelity
^^^^^^^^

Completeness
~~~~~~~~~~~~

.. automodule:: xai_metrics.metrics.fidelity.completeness.average_drop
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.average_gain
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.average_increase
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.completeness_metric
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.deletion
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.insertion
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.fidelity.completeness.mufidelity
   :members:
   :show-inheritance:

Soundness
~~~~~~~~~

.. automodule:: xai_metrics.metrics.fidelity.soundness.non_sensitivity
   :members:
   :show-inheritance:

Robustness
^^^^^^^^^^

.. automodule:: xai_metrics.metrics.robustness.average_stability
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.robustness.local_lipschitz_estimate
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.robustness.max_sensitivity
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.robustness.mege
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.robustness.relative_input_stability
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.robustness.relative_output_stability
   :members:
   :show-inheritance:

Sensitivity
^^^^^^^^^^^

.. automodule:: xai_metrics.metrics.sensitivity.avg_sensitivity
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.sensitivity.model_randomization
   :members:
   :show-inheritance:

.. automodule:: xai_metrics.metrics.sensitivity.random_logit
   :members:
   :show-inheritance:
