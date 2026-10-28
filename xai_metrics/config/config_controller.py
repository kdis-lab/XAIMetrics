# xai_metrics/config/config_controller.py
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from itertools import product

from xai_metrics.config.loaders import default_model_loader, load_config
from xai_metrics.config.validators import (
    validate_X_y_indexes,
    normalise_class_labels,
    validate_observations,
    validate_metric_context_metadata,
    validate_explainer_context_metadata
)
from xai_metrics.config.discovery import (
    discover_metric_context_configs,
    discover_explainer_context_configs
)
from xai_metrics.base import MetricContext, ExplainerContext

from typing import Mapping, Any, Callable, Dict, Tuple, List
import warnings

class ConfigController:
    """
    Controller for loading configuration files and building contexts.

    The controller can load configuration data from a mapping or from a YAML
    file. It supports direct context definitions and automatic discovery from
    dataset, model and attribution directories.

    Metric contexts include models, test data, labels and precomputed
    attributions. Explainer contexts include models, background data and,
    optionally, batches to explain.
    """
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path = "config.yaml",
        model_loader: Callable[[str | Path], Any] | None = None
    ):
        """
        Parameters
        ----------
        config : Mapping[str, Any], str or pathlib.Path, default="config.yaml"
            Configuration source. If a mapping is provided, it is copied
            directly. If a path is provided, the YAML file is loaded. The
            default value loads the ``config.yaml`` file located next to this
            module.
        model_loader : Callable[[str or pathlib.Path], Any] or None, optional
            Function used to load models from disk. The function must receive a
            model path and return the loaded model object. If ``None``,
            :func:`default_model_loader` is used.
        """
        self.config = load_config(config)
        self.model_loader = model_loader or default_model_loader
    

    def _iter_metric_context_configs(self) -> List[Dict[str, Any]]:
        """
        Build metric context configurations from the loaded configuration.

        If the ``context`` section contains direct paths, a single context
        configuration is returned. If it contains ``datasets_dir``,
        ``models_dir`` and ``attributions_dir``, the method searches those
        directories and creates one configuration for each valid combination of
        dataset, model, input data, labels and attribution file.

        The generated configurations always use the keys expected by
        :meth:`build_metric_context`, even when the discovered files are named
        as batches rather than test sets.

        Returns
        -------
        List[Dict[str, Any]]
            List of metric context configuration dictionaries.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or no valid metric context
            configurations can be found.
        """
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        if not all(key in ctx_cfg for key in ("datasets_dir", "models_dir", "attributions_dir")):
            validate_metric_context_metadata(ctx_cfg)
            return [dict(ctx_cfg)]
        
        return discover_metric_context_configs(ctx_cfg)
    

    def _iter_explainer_context_configs(self) -> List[Dict[str, Any]]:
        """
        Build explainer context configurations from the loaded configuration.

        If the ``context`` section contains direct paths, a single context
        configuration is returned. If it contains ``datasets_dir`` and
        ``models_dir``, the method searches those directories and creates one
        configuration for each valid combination of dataset, model, background
        data and batch data.

        Background data are discovered from files starting with
        ``x_background`` or, if none are found, ``x_train``. Batch data are
        discovered from files starting with ``x_batch`` or, if none are found,
        ``x_test``. Target files are added when matching ``y_background``,
        ``y_train``, ``y_batch`` or ``y_test`` files are available.

        Returns
        -------
        List[Dict[str, Any]]
            List of explainer context configuration dictionaries.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or no valid explainer context
            configurations can be found.
        """
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        if not all(key in ctx_cfg for key in ("datasets_dir", "models_dir")):
            validate_explainer_context_metadata(ctx_cfg)
            return [dict(ctx_cfg)]

        return discover_explainer_context_configs(ctx_cfg)
        
        
    def get_metrics_config(self) -> List:
        """
        Return the metrics configuration section.

        Returns
        -------
        List
            List of metric configuration dictionaries. If the section is not
            defined, an empty list is returned.
        """
        return self.config.get("metrics", [])
    

    def get_explainers_config(self) -> List:
        """
        Return the explainers configuration section.

        Returns
        -------
        List
            List of explainer configuration dictionaries. If the section is not
            defined, an empty list is returned.
        """
        return self.config.get("explainers", [])

    
    def build_metric_context(self, ctx_cfg: Mapping[str, Any] | None = None) -> Tuple[MetricContext, Dict[str, Any]]:
        """
        Build a metric evaluation context.

        The context configuration must contain paths to the model, input data,
        labels and attribution file. The method loads these objects, validates
        indexes, converts attribution values to a NumPy array, optionally moves
        the model to the configured device and returns both the
        :class:`MetricContext` and its metadata.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any] or None, optional
            Context configuration to use. If ``None``, the ``context`` section
            of the loaded configuration is used. The configuration must contain
            ``model_path``, ``X_test_path``, ``y_test_path`` and
            ``attributions_path``. It must also contain ``dataset_name``,
            ``model_name`` and ``xai_method_name``. It may optionally contain
            ``device``.

        Returns
        -------
        Tuple[MetricContext, Dict[str, Any]]
            Metric context and metadata dictionary.

        Raises
        ------
        ValueError
            If the ``context`` section is missing, required paths are missing,
            metadata are incomplete or indexes are inconsistent.
        TypeError
            If the loaded model is not an instance of ``torch.nn.Module``.
        RuntimeError
            If a CUDA device is requested but CUDA is not available.

        Warns
        -----
        UserWarning
            If ``y_test`` contains more than one column. In that case, only the
            first column is used.
        """
        ctx_cfg = dict(ctx_cfg or self.config.get("context") or {})

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        required = [
            "model_path",
            "X_test_path",
            "y_test_path",
            "attributions_path",
        ]

        missing = [key for key in required if key not in ctx_cfg]
        if missing:
            raise ValueError(f"Missing context config fields: {missing}")

        device = ctx_cfg.get("device")

        if device is not None:
            device = str(device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Device {device!r} requested but CUDA is not available.")
        
        model = self.model_loader(ctx_cfg["model_path"])

        if not isinstance(model, nn.Module):
            raise TypeError("The loaded model must be a torch.nn.Module.")

        if device is not None:
            model = model.to(device)
        
        X_test = pd.read_csv(ctx_cfg["X_test_path"], index_col=0)

        y_test_df = pd.read_csv(ctx_cfg['y_test_path'], index_col=0)
        if y_test_df.shape[1] > 1:
            warnings.warn(
                "y_test contains more than one column. "
                f"Only the first column will be used: {y_test_df.columns[0]!r}. "
                f"Ignored columns: {list(y_test_df.columns[1:])}.",
                UserWarning,
                stacklevel=2,
            )
        y_test = y_test_df.iloc[:, 0]

        X_test, y_test = validate_X_y_indexes(X_test, y_test)
        y_test = normalise_class_labels(y_test)

        attributions_df = pd.read_csv(ctx_cfg['attributions_path'], index_col=0)

        observations = attributions_df.index.tolist()
        observations = validate_observations(observations, X_test.index)

        attributions = attributions_df.to_numpy(dtype=float)

        metric_context = MetricContext(
            model=model,
            X_test=X_test,
            y_test=y_test,
            observations=observations,
            attributions=attributions,
            device=device
        )

        metadata = validate_metric_context_metadata(ctx_cfg)

        return metric_context, metadata


    def build_explainers_context(
        self,
        ctx_cfg: Mapping[str, Any] | None = None
    ) -> Tuple[ExplainerContext, Dict[str, Any]]:
        """
        Build an explainer context.

        The context configuration must contain a model path and background
        input data. It may also contain background labels, a batch of inputs to
        explain, batch labels and a device. The method loads these objects,
        validates batch indexes when labels are available and returns both the
        :class:`ExplainerContext` and its metadata.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any] or None, optional
            Context configuration to use. If ``None``, the ``context`` section
            of the loaded configuration is used. The configuration must contain
            ``model_path`` and ``X_background_path``. It must also contain
            ``dataset_name`` and ``model_name``. It may optionally contain
            ``y_background_path``, ``X_batch_path``, ``X_test_path``,
            ``y_batch_path``, ``y_test_path`` and ``device``.

        Returns
        -------
        Tuple[ExplainerContext, Dict[str, Any]]
            Explainer context and metadata dictionary.

        Raises
        ------
        ValueError
            If the ``context`` section is missing, required fields are missing,
            metadata are incomplete or batch indexes are inconsistent.
        RuntimeError
            If a CUDA device is requested but CUDA is not available.

        Warns
        -----
        UserWarning
            If a target file contains more than one column. In that case, only
            the first column is used.
        """
        ctx_cfg = dict(ctx_cfg or self.config.get("context") or {})

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")

        X_background_path = ctx_cfg.get("X_background_path")
        X_train_path = ctx_cfg.get("X_train_path")

        if X_background_path and X_train_path:
            warnings.warn(
                "Both X_background_path and X_train_path are configured. "
                "Using X_background_path as explainer background data.",
                UserWarning,
                stacklevel=2,
            )

        X_background_path = X_background_path or X_train_path

        if X_background_path is None:
            raise ValueError(
                "Explainer context requires X_background_path or X_train_path"
            )

        X_background = pd.read_csv(X_background_path, index_col=0)

        y_background_path = ctx_cfg.get("y_background_path")
        y_train_path = ctx_cfg.get("y_train_path")

        if y_background_path and y_train_path:
            warnings.warn(
                "Both y_background_path and y_train_path are configured. "
                "Using y_background_path as explainer background labels.",
                UserWarning,
                stacklevel=2,
            )

        y_background_path = y_background_path or y_train_path

        y_background = None
        if y_background_path is not None:
            y_background_df = pd.read_csv(y_background_path, index_col=0)

            if y_background_df.shape[1] > 1:
                warnings.warn(
                    "y_background contains more than one column. "
                    f"Only the first column will be used: {y_background_df.columns[0]}. "
                    f"Ignored columns: {list(y_background_df.columns[1:])}.",
                    UserWarning,
                    stacklevel=2,
                )

            y_background = y_background_df.iloc[:, 0]
            y_background = normalise_class_labels(y_background)
        
        model = None
        if "model_path" in ctx_cfg:
            model = self.model_loader(ctx_cfg["model_path"])
        
        X_batch_path = ctx_cfg.get("X_batch_path", ctx_cfg.get("X_test_path"))
        
        X_batch = None
        if X_batch_path is not None:
            X_batch = pd.read_csv(X_batch_path, index_col=0)
        
        y_batch_path = ctx_cfg.get("y_batch_path", ctx_cfg.get("y_test_path"))

        y_batch = None
        if y_batch_path is not None:
            y_batch_df = pd.read_csv(y_batch_path, index_col=0)

            if y_batch_df.shape[1] > 1:
                warnings.warn(
                    "y_batch contains more than one column. "
                    f"Only the first column will be used: {y_batch_df.columns[0]!r}. "
                    f"Ignored columns: {list(y_batch_df.columns[1:])}.",
                    UserWarning,
                    stacklevel=2,
                )

            y_batch = y_batch_df.iloc[:, 0]
            y_batch = normalise_class_labels(y_batch)

            if X_batch is not None:
                X_batch, y_batch = validate_X_y_indexes(X_batch, y_batch)
        
        device = ctx_cfg.get("device")
        if device is not None:
            device = str(device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Device {device!r} requested but CUDA is not available.")

        explainer_context = ExplainerContext(
            X_background=X_background,
            y_background=y_background,
            model=model,
            X_batch=X_batch,
            y_batch=y_batch,
            device=device
        )

        metadata = validate_explainer_context_metadata(ctx_cfg)

        return explainer_context, metadata
    
    
    def build_metric_contexts(self) -> List[Tuple[MetricContext, Dict[str, Any]]]:
        """
        Build all metric evaluation contexts defined by the configuration.

        The method first obtains all metric context configurations through
        :meth:`_iter_metric_context_configs` and then builds each corresponding
        :class:`MetricContext`.

        Returns
        -------
        List[Tuple[MetricContext, Dict[str, Any]]]
            List of metric contexts together with their metadata.
        """
        return [
            self.build_metric_context(ctx_cfg)
            for ctx_cfg in self._iter_metric_context_configs()
        ]
    

    def build_explainers_contexts(self) -> List[Tuple[ExplainerContext, Dict[str, Any]]]:
        """
        Build all explainer contexts defined by the configuration.

        The method first obtains all explainer context configurations through
        :meth:`_iter_explainer_context_configs` and then builds each
        corresponding :class:`ExplainerContext`.

        Returns
        -------
        List[Tuple[ExplainerContext, Dict[str, Any]]]
            List of explainer contexts together with their metadata.
        """
        return [
            self.build_explainers_context(ctx_cfg)
            for ctx_cfg in self._iter_explainer_context_configs()
        ]