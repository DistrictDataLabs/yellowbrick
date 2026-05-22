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

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from yellowbrick.contrib.wrapper import ContribEstimator


##########################################################################
## Model Type checking utilities
##########################################################################


def _get_estimator_type(estimator):
    """
    Return the estimator type string for an estimator (class or instance).

    Handles three generations of the sklearn API:
    - legacy ``_estimator_type`` class attribute (pre-sklearn 1.6)
    - ``__sklearn_tags__().estimator_type`` (sklearn 1.6+)
    - Mixin-based subclass detection for uninstantiated classes
    Also transparently unwraps Yellowbrick ``Wrapper`` instances so that
    visualizers delegate to their inner estimator.
    """
    # ContribEstimator and pre-sklearn-1.6 estimators set _estimator_type explicitly
    estimator_type = getattr(estimator, "_estimator_type", None)
    if estimator_type is not None:
        return estimator_type

    # For uninstantiated classes use Mixin subclass detection
    if inspect.isclass(estimator):
        if issubclass(estimator, ClassifierMixin):
            return "classifier"
        if issubclass(estimator, RegressorMixin):
            return "regressor"
        if issubclass(estimator, ClusterMixin):
            return "clusterer"
        return None

    # For Wrapper instances (e.g. ModelVisualizer / ScoreVisualizer),
    # delegate to the wrapped estimator rather than the visualizer itself
    # because the visualizer's own __sklearn_tags__ reports estimator_type=None
    wrapped = object.__getattribute__(estimator, "__dict__").get("_wrapped")
    if wrapped is not None and wrapped is not estimator:
        return _get_estimator_type(wrapped)

    # sklearn 1.6+ instances expose __sklearn_tags__()
    if hasattr(estimator, "__sklearn_tags__"):
        try:
            etype = estimator.__sklearn_tags__().estimator_type
            if etype is not None:
                return etype
        except Exception:
            # Some third-party estimators raise from __sklearn_tags__();
            # fall through to the Mixin-based check below.
            pass

    # Final fallback: Mixin-based subclass check on the instance's class.
    # Necessary when BaseEstimator precedes ClassifierMixin in the MRO (e.g.
    # class Foo(BaseEstimator, ClassifierMixin)) causing BaseEstimator's
    # __sklearn_tags__ to shadow ClassifierMixin's without calling super.
    cls = type(estimator)
    if issubclass(cls, ClassifierMixin):
        return "classifier"
    if issubclass(cls, RegressorMixin):
        return "regressor"
    if issubclass(cls, ClusterMixin):
        return "clusterer"

    return None


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


def is_classifier(estimator):
    """
    Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn clusterer, especially a
        Scikit-Learn estimator or Yellowbrick visualizer

    See also
    --------
    sklearn.base.is_classifier
        Equivalent function in scikit-learn.
    """
    return _get_estimator_type(estimator) == "classifier"


# Alias for closer name to isinstance and issubclass
isclassifier = is_classifier


def is_regressor(estimator):
    """
    Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn clusterer, especially a
        Scikit-Learn estimator or Yellowbrick visualizer

    See also
    --------
    sklearn.base.is_regressor
        Equivalent function in scikit-learn.
    """
    return _get_estimator_type(estimator) == "regressor"


# Alias for closer name to isinstance and issubclass
isregressor = is_regressor


def is_clusterer(estimator):
    """
    Returns True if the given estimator is a clusterer.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn clusterer, especially a
        Scikit-Learn estimator or Yellowbrick visualizer
    """
    return _get_estimator_type(estimator) == "clusterer"


# Alias for closer name to isinstance and issubclass
isclusterer = is_clusterer


def is_gridsearch(estimator):
    """
    Returns True if the given estimator is a clusterer.

    Parameters
    ----------
    estimator : class or instance
        The object to test if it is a Scikit-Learn clusterer, especially a
        Scikit-Learn estimator or Yellowbrick visualizer
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
