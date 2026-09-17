# xai_metrics\metrics\robustness\mege.py
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc, TrainingFunc

from typing import Mapping, Any, Tuple, List

@register_metric
class MeGe(BaseMetric):
    r"""
    MeGe representativity metric.

    This metric evaluates the representativity and stability of explanations
    across models trained on different subsets of the available data. The main
    idea is that explanations produced by models with similar predictive
    behaviour should also be similar.

    The selected observations are divided into ``k_splits`` equally sized
    subsets. For each split, a new model is trained using all remaining subsets,
    while the excluded subset is provided to the training function as validation
    or test data.

    Each trained model is then used to predict and explain every selected
    observation. Explanations produced by pairs of models are compared only when
    the observation belongs to a split excluded from the training data of at
    least one of the two models.

    A comparison is considered valid when:

    - at least one of the two models predicts the true class; and
    - both models predict the same class.

    For each valid comparison, explanation similarity is measured using the
    Spearman-based distance:

    .. math::
        d(e_i, e_j) = sqrt(1 - abs(rho(e_i, e_j)))

    where :math::``rho`` is the Spearman rank correlation between the two explanations.

    For each observation, the MeGe score is computed as:

    .. math::
        MeGe_i = 1 / (1 + mean(D_i))

    where ``D_i`` is the set of valid explanation distances associated with that
    observation.

    Higher values are better. A score close to ``1`` indicates that models with
    equivalent predictive behaviour produce highly similar explanations for the
    observation. Lower values indicate greater disagreement between explanations.

    If no valid model pair is available for an observation, its score is returned
    as ``NaN``. If no valid comparison is available for any selected observation,
    the metric is skipped.

    The implementation is based on the MeGe representativity metric proposed by
    Fel et al. (2020) and follows the pairwise comparison strategy implemented in
    Xplique. Unlike the original Xplique implementation, this wrapper returns one
    MeGe score per observation and does not compute the ReCo consistency metric.

    Fel, T., Vigouroux, D., Cadène, R., & Serre, T. (2020).
    How Good Is Your Explanation? Algorithmic Stability Measures to Assess the
    Quality of Explanations for Deep Neural Networks.
    """
    NAME = "MeGe"

    def __init__(
        self,
        context: MetricContext,
        training_func: TrainingFunc,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain ``X_test``,
            ``y_test`` and the selected observations. Classification targets are
            required by this metric.
        training_func : TrainingFunc
            Function used to train one model for each data split. It must accept
            ``(X_train, y_train, X_validation, y_validation)`` and return a
            trained model compatible with the prediction procedure used by the
            metric.
        explain_func : ExplainFunc
            Explanation function used to compute attribution values for each
            trained model. It must accept ``(model, inputs, targets)`` and return
            one explanation per input observation.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``k_splits`` : int, optional
              Number of equally sized data splits used to train the model
              ensemble. For every split, one model is trained without that
              split. Must be at least ``2`` and exactly divide the number of
              selected observations. The default value is ``4``.

            If ``None``, an empty dictionary is used.

        Raises
        ------
        ValueError
            If ``training_func`` or ``explain_func`` is ``None``.

        Notes
        -----
        The metric trains ``k_splits`` independent models. Consequently, its
        computational cost can be substantially higher than metrics that only
        evaluate the model contained in the metric context.

        The model stored in ``context.model`` is not used to compute the MeGe
        score itself. Models returned by ``training_func`` are used instead.
        """
        super().__init__(context, params)

        if training_func is None:
            raise ValueError("MeGe requires 'training_func' to be provided via dependencies.")

        if explain_func is None:
            raise ValueError("MeGe requires 'explain_func' to be provided via dependencies.")

        self.training_func = training_func
        self.explain_func = explain_func


    @staticmethod
    def _predict_classes(model: Any, inputs: np.ndarray, device: str | None) -> np.ndarray:
        """
        Predict one class label per observation.

        PyTorch models are evaluated directly. Models exposing ``predict`` or
        ``predict_proba`` are also supported. For multi-class outputs, the class
        associated with the largest model output is selected.

        When a scikit-learn-like model exposes ``classes_``, predicted indices
        are mapped back to the corresponding class labels.

        Parameters
        ----------
        model : Any
            Trained classification model.
        inputs : np.ndarray
            Input observations for which class predictions are required.
        device : str or None
            Device used for PyTorch inference. If ``None``, the device of the
            model parameters is used when available.

        Returns
        -------
        np.ndarray
            Predicted class label for each observation.

        Raises
        ------
        MetricSkipped
            If the model is neither a PyTorch module nor exposes a supported
            prediction interface.
        """
        if isinstance(model, torch.nn.Module):
            model_device = device

            if model_device is None:
                try:
                    model_device = str(next(model.parameters()).device)
                except StopIteration:
                    model_device = 'cpu'

            model.eval()

            with torch.no_grad():
                output = model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))

            if isinstance(output, (Tuple, List)):
                output = output[0]

            predictions = output.detach().cpu().numpy()

            if predictions.ndim == 1:
                return predictions.astype(int)

            return np.argmax(predictions, axis=-1)

        if hasattr(model, "predict"):
            predictions = np.asarray(model.predict(inputs))

            if predictions.ndim == 1:
                return predictions

            indices = np.argmax(predictions, axis=-1)

            if hasattr(model, "classes_") and len(model.classes_) == predictions.shape[-1]:
                return np.asarray(model.classes_)[indices]

            return indices

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(inputs))
            indices = np.argmax(probabilities, axis=-1)

            if hasattr(model, "classes_") and len(model.classes_) == probabilities.shape[-1]:
                return np.asarray(model.classes_)[indices]

            return indices

        raise MetricSkipped("MeGe requires models that implement predict(), predict_proba(), or are PyTorch modules.")

    @staticmethod
    def _spearman_distance(first: np.ndarray, second: np.ndarray) -> float:
        """
        Compute the Spearman-based distance between two explanations.

        Explanations with more than two dimensions are averaged over their last
        dimension before comparison. Missing values are replaced by zero and the
        resulting arrays are flattened.

        The distance is defined as:

            d = sqrt(1 - |rho|)

        where ``rho`` is the Spearman rank correlation between the two
        explanations.

        Parameters
        ----------
        first : np.ndarray
            First explanation.
        second : np.ndarray
            Second explanation.

        Returns
        -------
        float
            Spearman-based distance. Values normally belong to ``[0, 1]``.
            A value of ``0`` corresponds to a perfect positive or negative rank
            correlation, while ``1`` corresponds to zero rank correlation.
            ``NaN`` is returned when the Spearman correlation is undefined.

        Notes
        -----
        Because the absolute value of the correlation is used, perfectly
        positively and perfectly negatively correlated explanations both produce
        a distance of zero. This follows the distance definition used by the
        original MeGe implementation.
        """
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)

        if first.ndim > 2:
            first = np.mean(first, axis=-1)
        if second.ndim > 2:
            second = np.mean(second, axis=-1)

        first = np.nan_to_num(first, nan=0.0).reshape(-1)
        second = np.nan_to_num(second, nan=0.0).reshape(-1)

        correlation = spearmanr(first, second).statistic # pyright: ignore[reportAttributeAccessIssue]

        if np.isnan(correlation):
            return np.nan

        return float(np.sqrt(max(0.0, 1.0 - abs(correlation))))


    def run(self):
        """
        Compute the MeGe representativity metric.

        The selected observations are first divided into ``k_splits`` equally
        sized subsets. For each split, a model is trained on all remaining
        subsets using ``training_func``. The excluded split is passed to the
        training function as validation or test data.

        All trained models are then evaluated on every selected observation.
        Their predicted classes and explanations are computed and compared
        pairwise.

        For a given pair of models and observation, the explanation distance is
        considered only when:

        - the observation belongs to a split excluded from the training data of
          one of the two models;
        - at least one model predicts the true class; and
        - both models predict the same class.

        For every valid comparison, a Spearman-based explanation distance is
        computed as ``sqrt(1 - abs(rho))``.

        The final score for each observation is:

            1 / (1 + mean_distance)

        where ``mean_distance`` is the mean of all valid pairwise explanation
        distances obtained for that observation.

        Returns
        -------
        List[float]
            MeGe score for each evaluated observation. Higher values indicate
            stronger agreement between explanations produced by models with the
            same predictive behaviour.

            Observations for which no valid model pair is available receive
            ``NaN``.

        Raises
        ------
        MetricSkipped
            If classification targets are unavailable, if a trained model does
            not provide a supported prediction interface, if no observations
            are selected, or if no valid equal-prediction explanation pair is
            produced for any observation.
        ValueError
            If ``k_splits`` is smaller than ``2``, if the number of selected
            observations is not divisible by ``k_splits``, or if
            ``explain_func`` does not return one explanation for every selected
            observation.

        Notes
        -----
        Classification targets are required. Targets may be provided either as
        class labels or as two-dimensional encoded class vectors. In the latter
        case, the target class is obtained using ``argmax``.

        ``k_splits`` controls both the number of trained models and the number
        of possible pairwise explanation comparisons. Increasing it can provide
        more comparisons but also substantially increases the computational
        cost.

        The returned scores are computed independently for each observation.
        This differs from the original Xplique implementation, where all valid
        equal-prediction distances are pooled into a single global MeGe score.

        This implementation computes MeGe only. The ReCo consistency metric
        returned alongside MeGe by the original Xplique implementation is not
        calculated.
        """
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )

        if ctx.y_test is None:
            raise MetricSkipped("MeGe requires classification targets.")

        targets = np.asarray(ctx.y_test.loc[ctx.observations])

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        k_splits = int(p.get("k_splits", 4))

        if k_splits < 2:
            raise ValueError("k_splits must be at least 2.")

        if len(inputs) % k_splits != 0:
            raise ValueError("The number of selected observations must be divisible by k_splits")

        split_len = len(inputs) // k_splits

        x_splits = inputs.reshape(k_splits, split_len, *inputs.shape[1:])
        y_splits = targets.reshape(k_splits, split_len, *targets.shape[1:])

        models = []

        for split_index in range(k_splits):
            train_splits = [
                index
                for index in range(k_splits)
                if index != split_index
            ]

            X_train = x_splits[train_splits].reshape(-1, *inputs.shape[1:])
            y_train = y_splits[train_splits].reshape(-1, *targets.shape[1:])

            model = self.training_func(X_train, y_train, x_splits[split_index], y_splits[split_index])
            models.append(model)

        predictions = []
        explanations = []

        for model in models:
            predictions.append(self._predict_classes(model, inputs, ctx.device))
            explanations.append(np.asarray(self.explain_func(model, inputs, targets), dtype=np.float32))

        predictions = np.asarray(predictions)
        explanations = np.asarray(explanations)

        if explanations.shape[1] != len(inputs):
            raise ValueError("explain_func must return one explanation for every selected observation.")

        target_classes = np.argmax(targets, axis=-1) if targets.ndim == 2 else targets

        equal_prediction_distances = [[] for _ in range(len(inputs))]

        for first_model_index in range(k_splits):
            for second_model_index in range(first_model_index + 1, k_splits):
                for observation_index in range(len(inputs)):
                    split_index = observation_index // split_len

                    # Compare a sample only when one of both models did not train on it
                    if split_index not in (first_model_index, second_model_index):
                        continue

                    first_prediction = predictions[first_model_index, observation_index]
                    second_prediction = predictions[second_model_index, observation_index]
                    target = target_classes[observation_index]

                    # At least one model must predict the true class
                    if first_prediction != target and second_prediction != target:
                        continue

                    if first_prediction != second_prediction:
                        continue

                    distance = self._spearman_distance(
                        explanations[first_model_index, observation_index],
                        explanations[second_model_index, observation_index]
                    )

                    if not np.isnan(distance):
                        equal_prediction_distances[observation_index].append(distance)

        scores = []

        for distances in equal_prediction_distances:
            if distances:
                scores.append(float(1.0 / (1.0 + np.mean(distances))))
            else:
                # No model pair produced a valid comparison for this observation.
                scores.append(float("nan"))

        if np.all(np.isnan(scores)):
            raise MetricSkipped("MeGe skipped: no valid equal-prediction explanation pairs were produced.")

        return scores