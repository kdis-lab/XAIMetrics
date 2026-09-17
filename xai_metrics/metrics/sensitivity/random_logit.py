# xai_metrics\metrics\sensitivity\random_logit.py
import numpy as np
import torch

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped
from xai_metrics.base.types import ExplainFunc

from typing import Mapping, Any, Tuple, List

_EPS = 1e-8

@register_metric
class RandomLogit(BaseMetric):
    """
    Random Logit sensitivity metric.

    This metric evaluates whether an explanation depends on the target class for
    which it is computed. For each observation, the original explanation for the
    true target class is compared with an explanation generated for a randomly
    selected alternative class.

    For each observation ``i`` with target class ``y_i``, an alternative class
    ``y_i'`` is sampled uniformly from all available classes except ``y_i``:

        y_i' ~ Uniform({0, ..., C - 1} \\ {y_i})

    The explanation is then recomputed for the same input using the alternative
    target:

        phi_i = phi(f, x_i, y_i)

        phi_i' = phi(f, x_i, y_i')

    where ``f`` is the model and ``phi`` is the explanation function.

    Similarity between the original and alternative-target explanations is
    measured using the Structural Similarity Index (SSIM):

        RL_i = SSIM(phi_i, phi_i')

    The metric returns one similarity value per observation. Lower values
    indicate greater sensitivity to the target class, meaning that the
    explanation changes when a different class is explained. High similarity
    indicates that the explanation remains similar across different target
    classes, which may suggest insufficient class specificity.

    The number of classes can either be provided explicitly or inferred from a
    PyTorch classification model or a model exposing ``predict_proba``. Both
    integer class labels and two-dimensional encoded class targets are supported.

    The implementation is based on the Random Logit sanity check inspired by
    Adebayo et al. (2018) and follows the implementation provided by Xplique,
    adapted to per-observation outputs and NumPy-based SSIM computation.

    Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B.
    (2018). Sanity Checks for Saliency Maps.
    Advances in Neural Information Processing Systems, 31.
    """
    NAME = "RandomLogit"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution values.
            Classification targets are required by this metric.
        explain_func : ExplainFunc
            Explanation function used to recompute attribution values for
            alternative target classes. It must accept
            ``(model, inputs, targets)`` and return one explanation per input
            observation with the same shape as the original attributions.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``batch_size`` : int or None, optional
              Maximum number of observations processed at once. If ``None``,
              all selected observations are processed together. The default
              value is ``64``.

            - ``num_classes`` : int or None, optional
              Number of output classes. If provided, it must be at least ``2``.
              If ``None``, the number of classes is inferred from the model
              output. The default value is ``None``.

            - ``random_state`` : int, np.random.Generator or None, optional
              Random seed or NumPy random generator used to sample alternative
              target classes. The default value is ``42``.

            If ``None``, an empty dictionary is used.

        Raises
        ------
        ValueError
            If ``explain_func`` is ``None``.

        Notes
        -----
        Original attribution values stored in the metric context are used as
        reference explanations. Only explanations for randomly selected
        alternative classes are recomputed.

        For two-dimensional targets, the target class is obtained using
        ``argmax`` and alternative targets are passed to ``explain_func`` using
        one-hot encoding. For one-dimensional targets, alternative targets are
        passed as integer class indices.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("RandomLogit requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func


    def _check_number_classes(self, inputs: np.ndarray) -> int:
        """
        Determine the number of classes used by the classifier.

        If ``num_classes`` is provided in the metric parameters, that value is
        used directly. Otherwise, the number of classes is inferred from the
        second dimension of the model output.

        For PyTorch models, the model is evaluated directly on the selected
        inputs. For models exposing ``predict_proba``, the probability output is
        used.

        Parameters
        ----------
        inputs : np.ndarray
            Input observations used to infer the classifier output shape when
            ``num_classes`` is not explicitly configured.

        Returns
        -------
        int
            Number of output classes.

        Raises
        ------
        ValueError
            If the explicitly configured ``num_classes`` is smaller than ``2``.
        MetricSkipped
            If the number of classes cannot be inferred from the model or if the
            model output does not represent a classifier with at least two
            classes.
        """
        configured_classes = self.params.get("num_classes")

        if configured_classes is not None:
            number_classes = int(configured_classes)

            if number_classes < 2:
                raise ValueError("num_classes must be at least 2.")

            return number_classes

        model = self.context.model

        if isinstance(model, torch.nn.Module):
            model_device = self.context.device

            if model_device is None:
                try:
                    model_device = str(next(model.parameters()).device)
                except StopIteration:
                    model_device = "cpu"

            model.eval()

            with torch.no_grad():
                output = model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))

            if isinstance(output, (Tuple, List)):
                output = output[0]

            predictions = output.detach().cpu().numpy()

        elif hasattr(model, "predict_proba"):
            predictions = np.asarray(model.predict_proba(inputs))

        else:
            raise MetricSkipped("RandomLogit requires 'num_classes' or a model with predict_proba / a PyTorch classification output.")

        if predictions.ndim != 2 or predictions.shape[1] < 2:
            raise MetricSkipped("RandomLogit requires a classifier with at least two output classes.")

        return int(predictions.shape[1])


    @staticmethod
    def _ssim(first: np.ndarray, second: np.ndarray) -> float:
        """
        Compute a global SSIM similarity between two explanations.

        Both explanations are converted to one-dimensional arrays before the
        similarity is computed. Their joint value range is used to determine the
        SSIM stabilisation constants.

        The similarity is computed as:

            ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2))
            ------------------------------------------------
            ((mu_x^2 + mu_y^2 + C1) *
             (var_x + var_y + C2))

        where ``mu`` denotes the mean, ``var`` the population variance and
        ``cov`` the population covariance.

        Parameters
        ----------
        first : np.ndarray
            First explanation.
        second : np.ndarray
            Second explanation.

        Returns
        -------
        float
            Global SSIM similarity between the two explanations. Higher values
            indicate greater similarity.

            If both explanations have an effectively zero joint data range, the
            method returns ``1.0``.

        Raises
        ------
        ValueError
            If the explanations do not contain the same number of elements.

        Notes
        -----
        Explanations are flattened before computing the statistic. Consequently,
        this implementation uses a global SSIM formulation rather than the
        local-window image SSIM commonly used in computer vision.

        The result is not explicitly clipped, so numerical values may extend
        slightly outside the conventional SSIM range.
        """
        first = np.asarray(first, dtype=np.float64).reshape(-1)
        second = np.asarray(second, dtype=np.float64).reshape(-1)

        if first.shape != second.shape:
            raise ValueError("Original and randomized explanations must have the same shape.")

        minimum = min(float(np.min(first)), float(np.min(second)))
        maximum = max(float(np.max(first)), float(np.max(second)))
        data_range = maximum - minimum

        if data_range <= _EPS:
            return 1.0

        mean_first = float(np.mean(first))
        mean_second = float(np.mean(second))

        variance_first = float(np.mean((first - mean_first) ** 2))
        variance_second = float(np.mean((second - mean_second) ** 2))
        covariance = float(np.mean((first - mean_first) * (second - mean_second)))

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        numerator = (2.0 * mean_first * mean_second + c1) * (2.0 * covariance + c2)
        denominator = (mean_first ** 2 + mean_second ** 2 + c1) * (variance_first + variance_second + c2)

        return float(numerator / (denominator + _EPS))


    def run(self):
        """
        Compute the Random Logit metric.

        The method selects the observations defined in the metric context and
        determines the true target class associated with each observation. For
        every observation, a different class is sampled uniformly from the set
        of all available classes.

        The explanation function is then evaluated for the same inputs using the
        sampled alternative targets. Each resulting explanation is compared with
        its corresponding original explanation using global SSIM.

        Alternative classes are sampled so that they are always different from
        the original target class. Given ``C`` classes, each alternative class
        has probability ``1 / (C - 1)`` of being selected.

        Lower similarity values indicate that the explanation changes when the
        target class changes and therefore suggest greater target sensitivity or
        class specificity. High similarity values indicate that the explanation
        remains similar across different target classes.

        Returns
        -------
        List[float]
            SSIM similarity between the original target explanation and the
            randomly selected alternative-target explanation for each evaluated
            observation.

            Lower values indicate greater sensitivity to the target class.

        Raises
        ------
        MetricSkipped
            If no observations are selected, if classification targets are not
            available, if the number of classes cannot be determined, or if the
            model does not represent a classification problem with at least two
            output classes.
        ValueError
            If the number of explanations differs from the number of selected
            observations, if ``batch_size`` is not positive, if ``num_classes``
            is smaller than ``2``, if a target class lies outside the configured
            model output range, or if ``explain_func`` returns explanations with
            a shape different from the original attributions.

        Notes
        -----
        If targets are two-dimensional, they are interpreted as encoded class
        vectors and converted to class indices using ``argmax``. Alternative
        targets are then passed to ``explain_func`` as one-hot encoded vectors.

        If targets are one-dimensional, they are interpreted as integer class
        indices and alternative targets are passed using the same representation.

        A single alternative class is sampled independently for each observation.
        Consequently, the score can vary across executions unless
        ``random_state`` is fixed.

        Unlike the original Xplique ``evaluate`` method, this implementation
        returns one similarity value per observation rather than automatically
        averaging the results into a single dataset-level score.
        """
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        if ctx.y_test is None:
            raise MetricSkipped("RandomLogit requires classification targets.")

        targets = np.asarray(ctx.y_test.loc[ctx.observations])

        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        batch_size = p.get("batch_size", 64) or len(inputs)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")

        number_classes = self._check_number_classes(inputs)

        if targets.ndim == 2:
            target_classes = np.argmax(targets, axis=-1)
            one_hot_targets = True
        else:
            target_classes = targets.astype(int)
            one_hot_targets = False

        if np.any(target_classes < 0) or np.any(target_classes >= number_classes):
            raise ValueError("A target class is outside the model output range.")

        random_state = p.get("random_state", 42)
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

        scores = []

        for start in range(0, len(inputs), batch_size):
            stop = start + batch_size

            inputs_batch = inputs[start:stop]
            targets_batch = target_classes[start:stop]
            explanation_batch = explanations[start:stop]

            sampled_classes = rng.integers(low=0, high=number_classes - 1, size=len(inputs_batch))
            alternative_classes = np.where(sampled_classes >= targets_batch, sampled_classes + 1, sampled_classes)

            if one_hot_targets:
                alternative_targets = np.eye(number_classes, dtype=np.float32)[alternative_classes]
            else:
                alternative_targets = alternative_classes

            alternative_explanations = np.asarray(
                self.explain_func(
                    ctx.model,
                    inputs_batch,
                    alternative_targets
                ),
                dtype=np.float32
            )

            if alternative_explanations.shape != explanation_batch.shape:
                raise ValueError("explain_func must return explanations with the same shape as ctx.attributions.")

            scores.extend(
                self._ssim(original, alternative)
                for original, alternative in zip(explanation_batch, alternative_explanations)
            )

        return scores