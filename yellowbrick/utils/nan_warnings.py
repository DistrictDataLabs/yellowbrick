"""
Small helpers that help find missing data.
"""

import numpy as np
import warnings
from yellowbrick.exceptions import DataWarning


def warn_if_nans_exist(data):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(data)
    total = len(data)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values which are ' \
            'not plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)


def count_rows_with_nans(data):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if data.ndim == 2:
        return np.where(np.isnan(data).sum(axis=1) != 0, 1, 0).sum()


def count_nan_elements(data):
    """Count the number of elements in 1D arrays that are nan values."""
    if data.ndim == 1:
        return np.isnan(data).sum()


def drop_rows_containing_nans(data):
    """Drop rows in a numpy array that contain nan values."""
    return data[~np.isnan(data).any(axis=1)]
