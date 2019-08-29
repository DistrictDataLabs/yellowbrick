# yellowbrick.utils.nan_warnings
# Small helpers that help find and filter missing data.
#
# Author:   Aylr
# Created:  Thu Dec 28 11:37:42 2017 -0700
#
# Copyright (C) 2018 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: nan_warnings.py [d2276d6] Aylr@users.noreply.github.com $
#
"""
Small helpers that help find and filter missing data.
"""

import numpy as np
import warnings
from yellowbrick.exceptions import DataWarning


def filter_missing(X, y=None):
    """
    Removes rows that contain np.nan values in data. If y is given,
    X and y will be filtered together so that their shape remains identical.
    For example, rows in X with nans will also remove rows in y, or rows in y
    with np.nans will also remove corresponding rows in X.

    Parameters
    ------------
    X : array-like
        Data in shape (m, n) that possibly contains np.nan values

    y : array-like, optional
        Data in shape (m, 1) that possibly contains np.nan values

    Returns
    --------
    X' : np.array
       Possibly transformed X with any row containing np.nan removed

    y' : np.array
        If y is given, will also return possibly transformed y to match the
        shape of X'.

    Notes
    ------
    This function will return either a np.array if only X is passed or a tuple
    if both X and y is passed. Because all return values are indexable, it is
    important to recognize what is being passed to the function to determine
    its output.
    """
    if y is not None:
        return filter_missing_X_and_y(X, y)
    else:
        return X[~np.isnan(X).any(axis=1)]


def filter_missing_X_and_y(X, y):
    """Remove rows from X and y where either contains nans."""
    y_nans = np.isnan(y)
    x_nans = np.isnan(X).any(axis=1)
    unioned_nans = np.logical_or(x_nans, y_nans)

    return X[~unioned_nans], y[~unioned_nans]


def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = (
            "Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only "
            "complete rows will be plotted.".format(null_count, total, percent)
        )
        warnings.warn(warning_message, DataWarning)


def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()


def count_nan_elements(data):
    """Count the number of elements in 1D arrays that are nan values."""
    if data.ndim == 1:
        return np.isnan(data).sum()
