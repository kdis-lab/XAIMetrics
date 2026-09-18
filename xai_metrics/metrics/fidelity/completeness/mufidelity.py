# xai_metrics/metrics/fidelity/completeness/mufidelity.py
import numpy as np
import torch
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, register_metric, MetricContext, MetricSkipped

from typing import Any, Mapping, Tuple, List

@register_metric
class MuFidelity(BaseMetric):
    r"""
    MuFidelity correlation metric.

    This metric evaluates whether the importance assigned by an explanation to
    randomly selected subsets of features is consistent with the effect that
    perturbing those features has on the model output.

    For each observation :math:`i`, the metric generates :math:`R` random masks.
    A mask value of one preserves a feature, while a value of zero replaces it
    with a baseline value. For every perturbation :math:`r`, the perturbed
    input is:

    .. math::

        \tilde{\mathbf{x}}_i^{(r)} =
        \mathbf{x}_i \odot \mathbf{m}_i^{(r)} +
        \mathbf{b}_i \odot \left(1 - \mathbf{m}_i^{(r)} \right)

    The prediction drop and the attribution sum associated with the perturbed
    features are:

    .. math::

        \begin{aligned}
        \Delta_i^{(R)} &=
        g(f, \mathbf{x}_i, y_i) - g(f, \tilde{\mathbf{x}}_i^{(r)}, y_i), \\
        A_i^{(r)} &=
        \sum_j
        \phi_{i,j}
        \left(1 - m_{i,j}^{(r)} \right)
        \end{aligned}

    The final MuFidelity score for observation :math:`i` is the Spearman rank
    correlation between the prediction drops and attribution sums:

    .. math::

        \operatorname{MuFidelity}_i =
        \rho_{\mathrm{Spearman}}
        \left(\{\Delta_i^{(r)}\}_{r=1}^{R}, \{A_i^{(r)}\}_{r=1}^{R} \right)

    where :math:`f` is the model, :math:`g` is the scoring operator,
    :math:`\mathbf{x}_i` is the original input, :math:`\mathbf{b}_i` if the
    baseline input, :math:`\phi_{i,j}` is the attribution of the feature :math:`j`
    and :math:`\mathbf{m}_i^{(r)}` is the sampled mask.

    A high positive correlation indicates that subsets receiving larger
    attribution values tend to cause larger decreases in the model score when
    perturbed. Higher values are therefore better and indicate stronger
    agreement between the explanation and the model behaviour.

    Random subsets are generated using Bernoulli masks. For tabular data,
    features are sampled independently. For time-series and image data, masks
    can be generated on a lower-resolution grid and expanded using
    nearest-neighbour interpolation, allowing groups of nearby features to be
    perturbed together.

    The implementation is based on the fidelity correlation metric proposed by
    Bhatt et al. (2020) and follows the implementation provided by Xplique, with
    support for tabular data, time series, images, PyTorch models, models
    exposing ``predict`` or ``predict_proba``, and custom scoring operators.

    Bhatt, U., Weller, A., & Moura, J. M. F. (2020).
    Evaluating and Aggregating Feature-based Model Explanations.
    Proceedings of the Twenty-Ninth International Joint Conference on
    Artificial Intelligence (IJCAI).
    """
    NAME = "MuFidelity"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, selected observations and attribution values. ``y_test``
            may be ``None`` when the model score can be determined without
            explicit targets.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``nb_samples`` : int, optional
              Number of random perturbation masks generated for each observation.
              The default value is ``200``.

            - ``n_masks`` : int, optional
              Alias for ``nb_samples``. It is used only when ``nb_samples`` is
              not provided.

            - ``batch_size`` : int or None, optional
              Maximum number of perturbed observations processed at once. If
              ``None``, all perturbations for the selected observations may be
              processed together. The default value is ``64``.

            - ``grid_size`` : int or None, optional
              Resolution used to generate coarse perturbation masks. If ``None``
              or zero-like, the size of the first feature dimension is used.
              For time series, masks are generated over
              ``(grid_size, n_features)`` and expanded over the temporal
              dimension. For images, masks are generated over
              ``(grid_size, grid_size)`` and expanded over the spatial
              dimensions.

            - ``subset_percent`` : float, optional
              Probability that each generated mask element is replaced by the
              baseline. The default value is ``0.2``.

            - ``subset_probability`` : float, optional
              Alias for ``subset_percent``. It is used only when
              ``subset_percent`` is not provided.

            - ``baseline_mode`` : float, array-like or callable, optional
              Baseline used to replace perturbed features. If callable, it
              receives the repeated inputs associated with the generated
              perturbations. The default value is ``0.0``.

            - ``operator`` : callable or None, optional
              Custom scoring function ``g(model, inputs, targets)`` used to
              obtain one scalar score per observation. If ``None``, the metric
              performs inference directly using the model and selects the
              corresponding target score. The default value is ``None``.

            - ``activation`` : {None, "sigmoid", "softmax"}, optional
              Activation function applied to model predictions before selecting
              the score. ``None`` leaves predictions unchanged. The default
              value is ``None``.

            - ``random_state`` : int, np.random.Generator or None, optional
              Random seed or NumPy random generator used to create perturbation
              masks. Providing a fixed value makes mask generation reproducible.
              The default value is ``None``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        When no custom ``operator`` is provided, model inference is performed
        using, in order, a PyTorch ``torch.nn.Module``, ``predict_proba``,
        ``predict`` or direct model invocation.

        For two-dimensional model outputs, targets are interpreted as class
        indices. If no targets are provided, the maximum model output for each
        observation is used.

        The effective processing batch size is divided between observations and
        perturbations so that several perturbations of several observations can
        be evaluated together without exceeding the configured ``batch_size``.

        The metric returns one correlation value per observation rather than
        automatically averaging the correlations across the dataset.
        """
        super().__init__(context, params)


    def _score(self, inputs: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
        """
        Compute one scalar model score per observation.

        If a custom ``operator`` is configured, it is used directly. Otherwise,
        the method performs model inference and optionally applies the configured
        activation function.

        For two-dimensional predictions ``(N, C)``, the score corresponding to
        each target class is selected. If ``targets`` is ``None``, the maximum
        prediction over the class dimension is returned.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input observations.
        targets : np.ndarray or None
            Target class indices associated with the observations. If ``None``,
            the maximum model output is used for two-dimensional predictions.

        Returns
        -------
        np.ndarray
            One scalar score per observation, with shape ``(N,)``.

        Raises
        ------
        ValueError
            If the default scoring procedure receives model predictions with a
            shape other than ``(N,)`` or ``(N, C)``.
        """
        if self.operator is not None:
            result = self.operator(self.context.model, inputs, targets)
            return np.asarray(result, dtype=np.float32).reshape(-1)

        # get prediction
        if isinstance(self.context.model, torch.nn.Module):
            model_device = self.context.device
            if model_device is None:
                try:
                    model_device = str(next(self.context.model.parameters()).device)
                except StopIteration:
                    model_device = "cpu"
            self.context.model.eval()
            with torch.no_grad():
                output = self.context.model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))
            if isinstance(output, (Tuple, List)):
                output = output[0]
            prediction = output.detach().cpu().numpy()
        elif hasattr(self.context.model, "predict_proba"):
            prediction = np.asarray(self.context.model.predict_proba(inputs)) # pyright: ignore[reportCallIssue]
        elif hasattr(self.context.model, "predict"):
            prediction = np.asarray(self.context.model.predict(inputs)) # pyright: ignore[reportCallIssue]
        else:
            prediction = np.asarray(self.context.model(inputs))

        if self.activation == 'sigmoid':
            prediction = 1.0 / (1.0 + np.exp(-prediction))
        elif self.activation == 'softmax':
            shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
            exp_prediction = np.exp(shifted)
            prediction = exp_prediction / np.sum(exp_prediction, axis=-1, keepdims=True)

        if prediction.ndim == 1:
            return prediction.astype(np.float32)
        if prediction.ndim != 2:
            raise ValueError("The default MuFidelity operator requires predictions shaped (N, C).")
        if targets is None:
            return np.asarray(np.max(prediction, axis=-1), dtype=np.float32)
        if targets.ndim == 1:
            return np.asarray(prediction[np.arange(len(prediction)), targets.astype(int)], dtype=np.float32)
        return np.asarray(np.sum(prediction * targets, axis=-1, dtype=np.float32), dtype=np.float32)


    def _perturb_samples(self, inputs: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate random perturbation masks and apply the configured baseline.

        The method generates ``count`` Bernoulli masks and applies each mask to
        every input observation. Mask values equal to ``1`` preserve the original
        feature value, while values equal to ``0`` replace the corresponding
        feature by the configured baseline.

        The perturbed observations are computed as:

        .. math::

            \tilde{\mathbf{x}} =
            \mathbf{x} \odot \mathbf{m} + 
            \mathbf{b} \odot 
            \left(1 - \mathbf{m} \right)

        where :math:`\mathbf{x}` is the original input, :math:`\mathbf{m}` is
        the sampled binary mask and :math:`\mathbf{b}` is the baseline.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input observations.
        count : int
            Number of random perturbation masks generated for each observation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the flattened perturbed observations and their
            corresponding masks.

        Notes
        -----
        Each mask element is preserved when a uniform random value is greater
        than ``subset_percent``. Consequently, each element is replaced by the
        baseline with probability approximately equal to ``subset_percent``.

        The perturbed observations are computed as
        ``inputs * mask + baseline * (1 - mask)``.
        """
        if inputs.ndim == 2:
            masks = self.rng.uniform(size=(count, inputs.shape[1])) > self.subset_percent
        elif inputs.ndim == 3:
            # Nearest-neighbour resize of (grid_size, channels) to (time, channels).
            coarse = self.rng.uniform(size=(count, self.grid_size, inputs.shape[2])) > self.subset_percent
            time_indices = np.floor(np.arange(inputs.shape[1]) * self.grid_size / inputs.shape[1]).astype(int)
            masks = coarse[:, time_indices, :]
        else:
            # Nearest-neighbour resize of a grid, shared by every image channel.
            coarse = self.rng.uniform(size=(count, self.grid_size, self.grid_size)) > self.subset_percent
            height_indices = np.floor(np.arange(inputs.shape[1]) * self.grid_size / inputs.shape[1]).astype(int)
            width_indices = np.floor(np.arange(inputs.shape[2]) * self.grid_size / inputs.shape[2]).astype(int)
            masks = coarse[:, height_indices][:, :, width_indices, None]

        masks = np.broadcast_to(masks.astype(np.float32), (len(inputs), *masks.shape)).copy()
        repeated_inputs = np.repeat(inputs[:, None, ...], count, axis=1)
        baseline_mode = self.params.get("baseline_mode", 0.0)
        baseline = baseline_mode(repeated_inputs) if callable(baseline_mode) else baseline_mode
        degraded = repeated_inputs * masks + (1.0 - masks) * np.asarray(baseline, dtype=np.float32)
        return degraded.reshape((-1, *inputs.shape[1:])), masks


    def run(self):
        r"""
        Compute the MuFidelity correlation metric.

        The method selects the observations defined in the metric context and
        first computes their original model scores. For each observation,
        multiple random subsets of features are then perturbed by replacing them
        with a configured baseline value.

        For every perturbation, MuFidelity compares the model prediction drop
        with the sum of attributions assigned to the perturbed features:

        .. math::

            \begin{aligned}
            \Delta_i^{(r)} &=
            g(f, \mathbf{x}_i, y_i) + g(f, \tilde{\mathbf{x}}_i* {(r)}, y_i), \\
            A_i^{(r)} &=
            \sum_j
            \phi_{i,j}
            \left(1 - m_{i,j}^{(r)}\right)
            \end{aligned}

        The reported score is:

        .. math::

            \operatorname{MuFidelity}_i =
            \rho_{\mathrm{Spearman}}
            \left(\{\Delta_i^{(r)}\}_{r=1}^{R}, \{A_i^{(r)}\}_{r=1}^{R} \right)

        A high positive correlation indicates that perturbing features assigned
        greater importance by the explanation tends to produce larger decreases
        in the model score. Higher values are therefore better.

        Returns
        -------
        List[float]
            MuFidelity correlation for each evaluated observation. Values are
            Spearman correlation coefficients and therefore normally lie in
            ``[-1, 1]``. Higher positive values indicate stronger fidelity.

            If either sequence involved in the correlation is constant and the
            Spearman coefficient is undefined, the score for that observation is
            set to ``0.0``.

        Raises
        ------
        MetricSkipped
            If no observations are selected.
        ValueError
            If ``batch_size`` is not positive, if ``activation`` is not one of
            ``None``, ``"sigmoid"`` or ``"softmax"``, or if the number of
            explanations differs from the number of selected observations.

        Notes
        -----
        ``nb_samples`` controls the number of random subsets used to estimate the
        correlation. Larger values generally provide a more stable estimate at
        the cost of additional model evaluations.

        ``subset_percent`` controls the probability that a feature or coarse mask
        element is replaced by the baseline in each perturbation.

        When ``grid_size`` is smaller than the corresponding input dimension,
        groups of neighbouring features are perturbed together. This can be
        useful for medium- or high-dimensional inputs, particularly images and
        time series.

        The returned values are computed independently for each observation.
        Unlike the original Xplique ``evaluate`` method, they are not
        automatically averaged into a single dataset-level fidelity score.
        """
        ctx = self.context
        p = self.params

        inputs = np.asarray(
            ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True),
            dtype=np.float32
        )
        targets = None if ctx.y_test is None else np.asarray(ctx.y_test.loc[ctx.observations]).reshape(-1)

        if len(inputs) == 0:
            raise MetricSkipped(f"{self.NAME} skipped: no observations were selected.")

        nb_samples = int(p.get("nb_samples", p.get("n_masks", 200)))
        if nb_samples <= 0:
            raise ValueError("nb_samples must be positive.")

        batch_size = p.get("batch_size", 64) or (len(inputs) * nb_samples)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        
        self.grid_size = p.get("grid_size", None) or inputs.shape[1]
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive or None.")
        
        self.subset_percent = float(p.get("subset_percent", p.get("subset_probability", 0.2)))
        if not 0.0 <= self.subset_percent <= 1.0:
            raise ValueError("subset_percent must be in [0, 1].")
        
        self.operator = p.get("operator")
        self.activation = p.get("activation")

        if self.activation not in (None, 'sigmoid', 'softmax'):
            raise ValueError("activation must be None, 'sigmoid', or 'softmax'.")

        perturbation_batch_size = min(batch_size, nb_samples)
        inputs_batch_size = max(1, batch_size // perturbation_batch_size)

        random_state=p.get("random_state")
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        explanations = np.asarray(ctx.attributions, dtype=np.float32)
        if len(explanations) != len(inputs):
            raise ValueError(
                "The number of explanations must match the number of inputs: "
                f"{len(explanations)} vs {len(inputs)}."
            )

        base_predictions = self._score(inputs, targets)
        correlations = []
        for start in range(0, len(inputs), inputs_batch_size):
            stop = start + inputs_batch_size
            inputs_slice = inputs[start:stop]
            targets_slice= None if targets is None else targets[start:stop]
            phi = explanations[start:stop]
            base = base_predictions[start:stop, None]

            if inputs_slice.ndim > phi.ndim:
                phi = phi[..., None]
            phi = phi[:, None, ...]

            prediction_drops = []
            attribution_sums = []
            generated = 0
            while generated < nb_samples:
                count = min(perturbation_batch_size, nb_samples - generated)
                generated += count
                degraded, masks = self._perturb_samples(inputs_slice, count)
                repeated_targets = None if targets_slice is None else np.repeat(targets_slice, count, axis=0)
                perturbed = self._score(degraded, repeated_targets).reshape(len(inputs_slice), count)
                prediction_drops.append(base - perturbed)
                attribution_sums.append(np.sum(phi * (1.0 - masks), axis=tuple(range(2, masks.ndim))))

            predictions = np.concatenate(prediction_drops, axis=1)
            attributes = np.concatenate(attribution_sums, axis=1)
            for prediction, attribute in zip(predictions, attributes):
                correlation = spearmanr(prediction, attribute).statistic # pyright: ignore[reportAttributeAccessIssue]
                correlations.append(0.0 if np.isnan(correlation) else float(correlation))

        return correlations