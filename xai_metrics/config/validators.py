# xai_metrics\config\validators.py
import pandas as pd

from typing import Tuple, List, Mapping, Any, Dict

def validate_X_y_indexes(
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and align feature and target indexes.

    Both objects must contain exactly the same indexes. The returned target
    series is reordered to match the index order of the feature dataframe.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Input data.
    y_test : pandas.Series
        Labels or targets associated with ``X_test``.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.Series]
        Input data and target values aligned using the index order of
        ``X_test``.

    Raises
    ------
    ValueError
        If ``X_test`` and ``y_test`` do not contain the same indexes.
    """
    only_in_X = X_test.index.difference(y_test.index).tolist()
    only_in_y = y_test.index.difference(X_test.index).tolist()

    if only_in_X or only_in_y:
        raise ValueError(
            "X_test and y_test must contain the same indexes. "
            f"Indexes only in X_test: {only_in_X}. "
            f"Indexes only in y_test: {only_in_y}."
        )

    y_test = y_test.loc[X_test.index]

    return X_test, y_test


def normalise_class_labels(labels: pd.Series) -> pd.Series:
    """Return integer-valued class labels with an integer dtype.

    CSV files containing labels such as ``0.0`` and ``1.0`` are commonly
    inferred as floating point by pandas. Quantus uses these values as
    class-column indexes, so integer-valued numeric labels must be exposed
    as integers. Non-numeric or genuinely continuous targets are left
    unchanged.
    """
    numeric_labels = pd.to_numeric(labels, errors="coerce")

    if numeric_labels.notna().all() and (numeric_labels % 1 == 0).all():
        return numeric_labels.astype("int64")

    return labels
    

def validate_observations(
    observations: List,
    X_test_index: pd.Index
) -> List:
    """
    Validate that attribution observations exist in the input data.

    If the observation identifiers do not initially match the input index,
    the method tries to convert them to the same dtype as ``X_test_index``
    before validating them.

    Parameters
    ----------
    observations : List
        Observation identifiers obtained from the attribution file index.
    X_test_index : pandas.Index
        Index of the input data.

    Returns
    -------
    List
        Validated observation identifiers, possibly converted to the dtype
        of ``X_test_index``.

    Raises
    ------
    ValueError
        If the observations cannot be converted to the dtype of
        ``X_test_index`` or if any observation is missing.
    """
    observations_index = pd.Index(observations)

    if observations_index.isin(X_test_index).all():
        return observations
    
    try:
        observations_typed_idx = observations_index.astype(X_test_index.dtype).tolist()
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Observations from attributions_path are not compatible with "
            "the index type of X_test. "
            f"Observations dtype: {observations_index.dtype}. "
            f"X_test index dtype: {X_test_index.dtype}. "
        ) from e
    
    missing = [obs for obs in observations_typed_idx if obs not in X_test_index]

    if missing:
        raise ValueError(
            "All observations must exist in both X_test and y_test. "
            f"Missing in X_test: {missing}. "
        )

    return observations_typed_idx
    

def validate_metric_context_metadata(ctx_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate and return metadata required for a metric context.

    Metric contexts must be identified by dataset, model and explanation
    method names. These fields are used to organise metric results.

    Parameters
    ----------
    ctx_cfg : Mapping[str, Any]
        Context configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``dataset_name``, ``model_name`` and
        ``xai_method_name``.

    Raises
    ------
    ValueError
        If any required metadata field is missing or empty.
    """
    required = ["dataset_name", "model_name", "xai_method_name"]
    missing = [key for key in required if not ctx_cfg.get(key)]

    if missing:
        raise ValueError(
            "Context metadata is required when passing direct paths. "
            f"Missing fields: {missing}. "
            f"Required fields: {required}."
        )

    return {
        "dataset_name": ctx_cfg["dataset_name"],
        "model_name": ctx_cfg["model_name"],
        "xai_method_name": ctx_cfg["xai_method_name"],
    }
    

def validate_explainer_context_metadata(ctx_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate and return metadata required for an explainer context.

    Explainer contexts must be identified by dataset and model names. These
    fields are used to associate generated explanations with the
    corresponding data and model.

    Parameters
    ----------
    ctx_cfg : Mapping[str, Any]
        Context configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``dataset_name`` and ``model_name``.

    Raises
    ------
    ValueError
        If any required metadata field is missing or empty.
    """
    required = ["dataset_name"]
    missing = [key for key in required if not ctx_cfg.get(key)]

    if missing:
        raise ValueError(
            "Explainer context metadata is required when passing direct paths. "
            f"Missing fields: {missing}. "
            f"Required fields: {required}."
        )
    
    metadata = {
        "dataset_name": ctx_cfg['dataset_name'],
    }

    if ctx_cfg.get("model_name"):
        metadata['model_name'] = ctx_cfg['model_name']

    return metadata