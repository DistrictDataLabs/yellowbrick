"""
Test the RadViz feature analysis visualizers
"""

import unittest
import numpy as np

from yellowbrick.utils.nan_warnings import warn_if_nans_exist, count_nan_rows, \
    drop_rows_containing_nans


class TestNanWarnings(unittest.TestCase):
    def test_raise_warning_if_nans_exist(self):
        """
        Test that a warning is raised if any nans are in the data
        """
        data = np.array([
            [1, 2, 3],
            [1, 2, np.nan],
        ])

        self.assertWarns(UserWarning, warn_if_nans_exist, data)

    def test_count_rows_with_nans(self):
        """
        Test that a warning is raised if any nans are in the data
        """
        data0 = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])

        data2 = np.array([
            [np.nan, 2, 3],
            [1, np.nan, 3],
        ])

        data3 = np.array([
            [np.nan, 2, 3],
            [1, np.nan, 3],
            [np.nan, np.nan, np.nan],
        ])

        self.assertEqual(0, count_nan_rows(data0))
        self.assertEqual(2, count_nan_rows(data2))
        self.assertEqual(3, count_nan_rows(data3))

    def test_drop_nan_rows_no_nans(self):
        """
        Test that an array with no nulls is returned intact
        """
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        observed = drop_rows_containing_nans(data)
        np.testing.assert_array_equal(data, observed)

    def test_drop_nan_rows(self):
        """
        Test that an array with nulls is returned without null containing rows
        """
        data = np.array([
            [1, 2, np.nan],
            [4, 5, 6],
            [np.nan, np.nan, np.nan],
        ])

        expected = np.array([
            [4, 5, 6],
        ])

        observed = drop_rows_containing_nans(data)
        np.testing.assert_array_equal(expected, observed)
