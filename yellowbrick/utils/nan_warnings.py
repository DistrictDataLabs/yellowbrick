"""
Small helpers that help find missing data.
"""

import numpy as np
import warnings


def warn_if_nans_exist(data):
    """Warn if nans exist in a numpy array."""
    null_count = count_nan_rows(data)
    total = len(data)
    percent = null_count / total

    # if np.isnan(data).any():
    if null_count > 0:
        warnings.warn(
            'Warning! Found {} rows of {} ({}%) with nan/null/None '
            'values which are cannot be plotted.'.format(null_count, total,
                                                         percent))


def count_nan_rows(data):
    """Count the number of rows that contain any nan values."""
    if data.shape[0] >= 2:
        return np.where(np.isnan(data).sum(axis=1) != 0, 1, 0).sum()


def drop_rows_containing_nans(data):
    """Drop rows in a numpy array that contain nan values."""
    return data[~np.isnan(data).any(axis=1)]
