"""
Small helpers that help find missing data.
"""

import numpy as np
import warnings
from yellowbrick.exceptions import DataWarning


def warn_if_nans_exist(data):
    """Warn if nans exist in a numpy array."""
    null_count = count_nan_rows(data)
    total = len(data)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values which are ' \
            'not plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)


def count_nan_rows(data):
    """Count the number of rows that contain any nan values."""
    if data.shape[0] >= 2:
        return np.where(np.isnan(data).sum(axis=1) != 0, 1, 0).sum()


def drop_rows_containing_nans(data):
    """Drop rows in a numpy array that contain nan values."""
    return data[~np.isnan(data).any(axis=1)]
