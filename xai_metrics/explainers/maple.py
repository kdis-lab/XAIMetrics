# xai_metrics/explainers/maple.py
"""
MAPLE explainer.

Algorithm based on:

Plumb, G., Molitor, D., & Talwalkar, A. (2018).
Model Agnostic Supervised Local Explanations.
arXiv:1807.02910.

Reference implementation:
https://github.com/GDPlumb/MAPLE
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch import Tensor

from xai_metrics.base import register_explainer, BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping, Tuple


class _MAPLEModel:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fe_type: str = "rf",
        n_estimators: int = 200,
        max_features: float | int | str | None = 0.5,
        min_samples_leaf: int = 10,
        regularization: float = 0.001,
        random_state: int | None = 42
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.num_train = X_train.shape[0]
        self.num_features = X_train.shape[1]
        self.num_val = X_val.shape[0]
        self.regularization = regularization

        if fe_type == "rf":
            self.ensemble = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, # type: ignore
                random_state=random_state
            )
        elif fe_type == "gbrt":
            self.ensemble = GradientBoostingRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, # type: ignore
                max_depth=None,
                random_state=random_state
            )
        else:
            raise ValueError(
                f"Unknown MAPLE forest ensemble type: {fe_type!r}. "
                "Expected 'rf' or 'gbrt'."
            )
        
        self.ensemble.fit(X_train, y_train)

        self.train_leaf_ids = np.asarray(self.ensemble.apply(X_train))

        if self.train_leaf_ids.ndim == 3:
            self.train_leaf_ids = self.train_leaf_ids[:, :, 0]

        if self.train_leaf_ids.ndim == 1:
            self.train_leaf_ids = self.train_leaf_ids.reshape(-1, 1)

        val_leaf_ids = np.asarray(self.ensemble.apply(X_val))

        if val_leaf_ids.ndim == 3:
            val_leaf_ids = val_leaf_ids[:, :, 0]

        if val_leaf_ids.ndim == 1:
            val_leaf_ids = val_leaf_ids.reshape(-1, 1)

        self.feature_scores = np.zeros(self.num_features, dtype=float)
        if fe_type == "rf":
            estimators = self.ensemble.estimators_
        else:
            estimators = self.ensemble.estimators_[:, 0]
        
        for estimator in estimators:
            root_feature = estimator.tree_.feature[0]

            if root_feature >= 0:
                self.feature_scores[root_feature] += estimator.tree_.impurity[0]
        feature_order = np.argsort(-self.feature_scores)

        best_rmse = np.inf
        best_features = np.arange(self.num_features)

        for retain in range(1, self.num_features + 1):
            selected = np.sort(feature_order[:retain])
            predictions = np.empty(X_val.shape[0], dtype=float)

            for index, row in enumerate(X_val):
                weights = self.training_point_weights(val_leaf_ids[index])

                local_model = Ridge(alpha=regularization)
                local_model.fit(X_train[:, selected], y_train, sample_weight=weights)

                predictions[index] = local_model.predict(row[selected].reshape(1, -1))[0]

            rmse = np.sqrt(np.mean((predictions - y_val) ** 2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_features = selected
        
        self.selected_features = best_features
        self.selected_X_train = X_train[:, best_features]

        
    def training_point_weights(self, instance_leaf_ids: np.ndarray) -> np.ndarray:
        matches = self.train_leaf_ids == instance_leaf_ids
        leaf_sizes = matches.sum(axis=0)

        valid_trees = leaf_sizes > 0

        return (matches[:, valid_trees] / leaf_sizes[valid_trees]).sum(axis=1)
    

    def explain(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(1, -1)
        leaf_ids = np.asarray(self.ensemble.apply(x))

        if leaf_ids.ndim == 3:
            leaf_ids = leaf_ids[:, :, 0]

        if leaf_ids.ndim == 1:
            leaf_ids = leaf_ids.reshape(-1, 1)

        leaf_ids = leaf_ids[0]

        weights = self.training_point_weights(leaf_ids)

        local_model = Ridge(alpha=self.regularization)
        local_model.fit(self.selected_X_train, self.y_train, sample_weight=weights)

        coefficients = np.zeros(self.num_features, dtype=float)
        coefficients[self.selected_features] = np.asarray(local_model.coef_).reshape(-1)

        return coefficients



@register_explainer
class MAPLEExplainer(BaseExplainer):
    NAME = "MAPLE"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        if context.model is None:
            raise ValueError("MAPLEExplainer requires context.model.")
        
        self.cols = list(context.X_background.columns)
        background = self._to_numpy(context.X_background)

        if background.shape[0] < 3:
            raise ValueError(
                "MAPLEExplainer requires at least three background observations."
            )
        
        X_train, X_val = self._split_background(background)

        y_train = self._model_response(context.model, X_train)
        y_val = self._model_response(context.model, X_val)

        self.explainer = _MAPLEModel(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fe_type=self.params.get("fe_type", "rf"),
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_features=self.params.get("max_features", 0.5),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
            regularization=float(self.params.get("regularization", 0.001)),
            random_state=self.params.get("random_state", 42)
        )


    def _to_numpy(self, inputs: Any) -> np.ndarray:
        if isinstance(inputs, pd.DataFrame):
            values = inputs.loc[:, self.cols].to_numpy()
        elif isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return values
    
    
    def _get_predict_fn(self, model: Module, predict_fn=None):
        if predict_fn is not None:
            return predict_fn

        mode = self.params.get("mode", "classification")

        if mode == "classification":
            method_name = "predict_proba"
        else:
            method_name = "predict"

        if hasattr(model, method_name):
            return getattr(model, method_name)

        if hasattr(model, "model") and hasattr(model.model, method_name):
            return getattr(model.model, method_name)

        raise AttributeError(
            f"No se encontro {method_name} en {type(model)} ni en model.model"
        )
    

    def _split_background(
        self,
        background: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        validation_size = float(self.params.get("validation_size", 0.2))
        data_type = self.params.get("data_type", "tabular")

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if data_type not in {"tabular", "time_series"}:
            raise ValueError(
                "data_type must be 'tabular' or 'time_series'."
            )

        if len(background) < 2:
            raise ValueError(
                "MAPLE requires at least two background observations."
            )
        
        if data_type == "time_series":
            validation_samples = max(1, int(np.ceil(len(background) * validation_size)))
            split_index = len(background) - validation_samples

            if split_index < 1:
                raise ValueError(
                    "validation_size leaves no observations for training."
                )
            
            X_train = background[:split_index]
            X_val = background[split_index:]

            return X_train, X_val
        
        X_train, X_val = train_test_split(
            background,
            test_size=validation_size,
            shuffle=True,
            random_state=self.params.get("random_state", 42)
        )
        
        return X_train, X_val
    

    def _model_response(
        self,
        model: Module,
        inputs: np.ndarray
    ) -> np.ndarray:
        prediction = np.asarray(self._get_predict_fn(model)(inputs))

        if prediction.ndim == 2:
            output_index = int(self.params.get("output_index", 1))

            if output_index >= prediction.shape[1]:
                raise ValueError(
                    f"MAPLE output_index={output_index} is invalid for "
                    f"model output shape {prediction.shape}."
                )
            
            prediction = prediction[:, output_index]

        return prediction.reshape(-1).astype(float)
    

    def explain(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray[Tuple[Any, ...], np.dtype[Any]]:
        X = self._to_numpy(inputs)

        attributions = np.asarray([
            self.explainer.explain(row)
            for row in X
        ])

        if self.params.get("abs", False):
            attributions = np.abs(attributions)

        return attributions