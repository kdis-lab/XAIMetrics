# examples/specific_examples/make_explan_func.py
import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer

def make_lime_explain_func(background):
    columns = list(background.columns)
    explainer = LimeTabularExplainer(
        background.to_numpy(),
        feature_names=columns,
        mode="classification",
        random_state=42,
    )

    def explain_func(model, inputs, targets=None, **kwargs):
        detector = model
        required_method = "predict_proba"
        if hasattr(model, required_method):
            detector = model
        elif hasattr(model, "model") and hasattr(model.model, required_method):
            detector = model.model
        else:
            raise AttributeError(f"No se encontro {required_method} en {type(model)} ni en model.model")

        if isinstance(inputs, pd.DataFrame):
            return inputs.loc[:, columns]
        if isinstance(inputs, torch.Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        X_batch = pd.DataFrame(values, columns=columns)

        attributions = []
        for row in X_batch.to_numpy():
            explanation = explainer.explain_instance(
                data_row=row,
                predict_fn=detector.predict_proba,
                num_features=len(columns),
            )
            weights = np.zeros(len(columns), dtype=float)
            for feature_idx, weight in explanation.as_map()[1]:
                weights[feature_idx] = float(weight)
            attributions.append(weights)
        return np.asarray(attributions)

    return explain_func