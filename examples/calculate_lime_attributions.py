# examples/calculate_lime_attributions.py

from pathlib import Path
import pandas as pd

from xai_metrics.config.config_controller import default_model_loader
from xai_metrics.base import ExplainerContext
from xai_metrics.explainers.lime import LIMEExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_train_path = PROJECT_ROOT / "examples/data/hydraulic/X_train.csv"
X_test_path = PROJECT_ROOT / "examples/data/hydraulic/X_test.csv"
y_test_path = PROJECT_ROOT / "examples/data/hydraulic/y_test.csv"
model_path = PROJECT_ROOT / "examples/models/hydraulic/IForest/hydraulic_IForest_seed_0.pkl"

output_path = (
    PROJECT_ROOT
    / "examples/attributions/hydraulic/IForest/LIME/hydraulic_IForest_lime_attributions.csv"
)


def main():
    X_background = pd.read_csv(X_train_path, index_col=0)
    X_test = pd.read_csv(X_test_path, index_col=0)
    y_test = pd.read_csv(y_test_path, index_col=0).iloc[:, 0]

    model = default_model_loader(model_path)

    context = ExplainerContext(
        X_background=X_background,
        y_background=None,
        device="cpu",
    )

    explainer = LIMEExplainer(
        context=context,
        params={
            "mode": "classification",
            "random_state": 42,
            "num_samples": 5000,
            "num_features": len(X_background.columns),
            "labels": [1],
            "distance_metric": "euclidean",
        },
    )

    attributions = explainer.explain(
        model=model,
        inputs=X_test,
        targets=y_test,
    )

    attributions_df = pd.DataFrame(
        attributions,
        index=X_test.index,
        columns=X_test.columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    attributions_df.to_csv(output_path)

    print(f"LIME attributions saved to: {output_path}")
    print(f"Shape: {attributions_df.shape}")


if __name__ == "__main__":
    main()