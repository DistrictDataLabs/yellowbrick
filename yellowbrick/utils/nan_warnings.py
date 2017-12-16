"""
Small helpers that help find missing data.
"""

import numpy as np
import warnings
from yellowbrick.exceptions import DataWarning


def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)


def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()


def count_nan_elements(data):
    """Count the number of elements in 1D arrays that are nan values."""
    if data.ndim == 1:
        return np.isnan(data).sum()


def clean_data(X, y=None):
    """Clean rows that contain nans in X or y (if given)."""
    if y is not None:
        return clean_X_y(X, y)
    else:
        return X[~np.isnan(X).any(axis=1)]


def clean_X_y(X, y):
    """Remove rows from X and y where either contains nans."""
    y_nans = np.isnan(y)
    x_nans = np.isnan(X).any(axis=1)
    unioned_nans = np.logical_or(x_nans, y_nans)

    return X[~unioned_nans], y[~unioned_nans]
