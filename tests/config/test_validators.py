# tests\config\test_validators.py
import pandas as pd
import pytest

from xai_metrics.config.validators import validate_X_y_indexes, validate_observations

def test_validate_x_y_indexes_reorders_y_to_match_x():
    X_test = pd.DataFrame({"x1": [1.0, 2.0]}, index=[20, 10])
    y_test = pd.Series([0, 1], index=[10, 20])

    X_valid, y_valid = validate_X_y_indexes(X_test, y_test)

    assert X_valid.index.tolist() == [20, 10]
    assert y_valid.index.tolist() == [20, 10]
    assert y_valid.tolist() == [1, 0]


def test_validate_x_y_indexes_rejects_mismatched_indexes():
    X_test = pd.DataFrame({"x1": [1.0]}, index=[1])
    y_test = pd.Series([0], index=[2])

    with pytest.raises(ValueError, match="X_test and y_test must contain the same indexes"):
        validate_X_y_indexes(X_test, y_test)


def test_validate_observations_casts_to_x_test_index_dtype():
    observations = validate_observations(
        observations=["1", "2"],
        X_test_index=pd.Index([1, 2, 3])
    )

    assert observations == [1, 2]


def test_validate_observations_rejects_observations_missing_from_x_test():
    with pytest.raises(
        ValueError,
        match="All observations must exist in both X_test and y_test",
    ):
        validate_observations(
            observations=[10, 99],
            X_test_index=pd.Index([10, 11]),
        )