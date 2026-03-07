# yellowbrick.utils.types
# Detection utilities for Scikit-Learn and Numpy types for flexibility
#
# Author:   Benjamin Bengfort
# Created:  Fri May 19 10:51:13 2017 -0700
#
# Copyright (C) 2017 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: types.py [79cd8cf] benjamin@bengfort.com $

"""
Detection utilities for Scikit-Learn and Numpy types for flexibility
"""

##########################################################################
## Imports
##########################################################################

import inspect
import numpy as np

from sklearn.base import BaseEstimator, is_classifier, is_regressor, is_clusterer
from yellowbrick.contrib.wrapper import ContribEstimator


##########################################################################
## Model Type checking utilities
##########################################################################


def is_estimator(model):
    """
    Determines if a model is an estimator using issubclass and isinstance.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn clusterer, especially a
        Scikit-Learn estimator or Yellowbrick visualizer
    """
    if inspect.isclass(model):
        return issubclass(model, (BaseEstimator, ContribEstimator))

    return isinstance(model, (BaseEstimator, ContribEstimator))


# Alias for closer name to isinstance and issubclass
isestimator = is_estimator


# Alias for closer name to isinstance and issubclass
isclassifier = is_classifier


# Alias for closer name to isinstance and issubclass
isregressor = is_regressor


# Alias for closer name to isinstance and issubclass
isclusterer = is_clusterer


def is_gridsearch(estimator):
    """
    Returns True if the given estimator is a grid-search estimator.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn grid-search estimator.
    """

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    if inspect.isclass(estimator):
        return issubclass(estimator, (GridSearchCV, RandomizedSearchCV))

    return isinstance(estimator, (GridSearchCV, RandomizedSearchCV))


# Alias for closer name to isinstance and issubclass
isgridsearch = is_gridsearch


def is_probabilistic(estimator):
    """
    Returns True if the given estimator returns a y_score for it's decision
    function, e.g. has ``predict_proba`` or ``decision_function`` methods.

    Parameters
    ----------
    estimator : class or instance
        The object to test if is probabilistic, especially a Scikit-Learn
        estimator or Yellowbrick visualizer.
    """
    return any(
        [hasattr(estimator, "predict_proba"), hasattr(estimator, "decision_function")]
    )


# Alias for closer name to isinstance and issubclass
isprobabilistic = is_probabilistic


##########################################################################
## Data Type checking utilities
##########################################################################


def is_dataframe(obj):
    """
    Returns True if the given object is a Pandas Data Frame.

    Parameters
    ----------
    obj: instance
        The object to test whether or not is a Pandas DataFrame.
    """
    try:
        # This is the best method of type checking
        from pandas import DataFrame

        return isinstance(obj, DataFrame)
    except ImportError:
        # Pandas is not a dependency, so this is scary
        return obj.__class__.__name__ == "DataFrame"


# Alias for closer name to isinstance and issubclass
isdataframe = is_dataframe


def is_series(obj):
    """
    Returns True if the given object is a Pandas Series.

    Parameters
    ----------
    obj: instance
        The object to test whether or not is a Pandas Series.
    """
    try:
        # This is the best method of type checking
        from pandas import Series

        return isinstance(obj, Series)
    except ImportError:
        # Pandas is not a dependency, so this is scary
        return obj.__class__.__name__ == "Series"


# Alias for closer name to isinstance and issubclass
isseries = is_series


def is_structured_array(obj):
    """
    Returns True if the given object is a Numpy Structured Array.

    Parameters
    ----------
    obj: instance
        The object to test whether or not is a Numpy Structured Array.
    """
    if isinstance(obj, np.ndarray) and hasattr(obj, "dtype"):
        if obj.dtype.names is not None:
            return True
    return False


# Alias for closer name to isinstance and issubclass
isstructuredarray = is_structured_array
