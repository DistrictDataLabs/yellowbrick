"""Test the RadViz feature analysis visualizers."""

import pytest
import numpy as np

from yellowbrick.utils.nan_warnings import warn_if_nans_exist, \
    count_rows_with_nans, \
    drop_rows_containing_nans, count_nan_elements
from yellowbrick.exceptions import DataWarning


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


def test_drop_nan_rows_no_nans():
    """Test that an array with no nulls is returned intact."""
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    observed = drop_rows_containing_nans(data)
    np.testing.assert_array_equal(data, observed)


def test_drop_nan_rows():
    """Test that arrays with missing data are returned without nans."""
    data = np.array([
        [1, 2, np.nan],
        [4, 5, 6],
        [np.nan, np.nan, np.nan],
    ])

    expected = np.array([
        [4, 5, 6]
    ])

    observed = drop_rows_containing_nans(data)
    np.testing.assert_array_equal(expected, observed)
