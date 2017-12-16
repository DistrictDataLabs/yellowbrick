"""Test the RadViz feature analysis visualizers."""

import numpy as np
import pytest

from yellowbrick.exceptions import DataWarning
from yellowbrick.utils.nan_warnings import count_nan_elements, \
    count_rows_with_nans, warn_if_nans_exist, clean_data


def test_raise_warning_if_nans_exist():
    """Test that a warning is raised if any nans are in the data."""
    data = np.array([
        [1, 2, 3],
        [1, 2, np.nan],
    ])

    with pytest.warns(DataWarning):
        warn_if_nans_exist(data)


def test_count_rows_in_2d_arrays_with_nans():
    """Test that a warning is raised if any nans are in the data."""
    data0 = np.array([
        [1, 2, 3],
    ])

    data2 = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [np.nan, 2, 3],
        [1, np.nan, 3],
    ])

    data3 = np.array([
        [1, 2, 3],
        [np.nan, 2, 3],
        [1, np.nan, 3],
        [np.nan, np.nan, np.nan],
    ])

    assert count_rows_with_nans(data0) == 0
    assert count_rows_with_nans(data2) == 2
    assert count_rows_with_nans(data3) == 3


def test_count_nan_elements():
    """Test that a warning is raised if any nans are in the data."""
    data0 = np.array([1, 2, 3])
    data1 = np.array([1, np.nan, 3])
    data3 = np.array([np.nan, np.nan, np.nan])

    assert count_nan_elements(data0) == 0
    assert count_nan_elements(data1) == 1
    assert count_nan_elements(data3) == 3


def test_clean_data_X_only_no_nans():
    """Test that an array with no nulls is returned intact."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    observed = clean_data(X)
    np.testing.assert_array_equal(X, observed)


def test_clean_data_X_only():
    """Test that arrays with missing data are returned without nans."""
    X = np.array([
        [1, 2, np.nan],
        [4, 5, 6],
        [np.nan, np.nan, np.nan],
    ])

    expected = np.array([
        [4, 5, 6]
    ])
    observed = clean_data(X)

    np.testing.assert_array_equal(expected, observed)


def test_clean_data_dirty_X_dirty_y():
    """Test that arrays with missing data are returned without nans."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    y = np.array([33, np.nan, 44, np.nan])

    expected_X = np.array([
        [1, 2, 3],
    ])
    expected_y = np.array([33])
    observed_X, observed_y = clean_data(X, y)

    np.testing.assert_array_equal(expected_X, observed_X)
    np.testing.assert_array_equal(expected_y, observed_y)


def test_clean_data_dirty_X_clean_y():
    """Test that arrays with missing data are returned without nans."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    y = np.array([33, 44, 55, 66])

    expected_X = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    expected_y = np.array([33, 44])
    observed_X, observed_y = clean_data(X, y)

    np.testing.assert_array_equal(expected_X, observed_X)
    np.testing.assert_array_equal(expected_y, observed_y)


def test_clean_data_clean_X_dirty_y():
    """Test that arrays with missing data are returned without nans."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    y = np.array([np.nan, 44, np.nan, 66])

    expected_X = np.array([
        [4, 5, 6],
        [10, 11, 12]
    ])
    expected_y = np.array([44, 66])
    observed_X, observed_y = clean_data(X, y)

    np.testing.assert_array_equal(expected_X, observed_X)
    np.testing.assert_array_equal(expected_y, observed_y)
