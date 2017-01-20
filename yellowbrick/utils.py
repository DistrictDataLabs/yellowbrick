# yellowbrick.utils
# Utility functions and helpers for the Yellowbrick library.
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
# Author:   Patrick O'Melveny <pvomelveny@gmail.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Thu Jun 02 15:33:18 2016 -0500
#
# Copyright (C) 2016 District Data LAbs
# For license information, see LICENSE.txt
#
# ID: utils.py [] jason.s.keung@gmail.com $

"""
Utility functions and helpers for the Yellowbrick library.
"""

##########################################################################
## Imports
##########################################################################

import inspect

from functools import wraps
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from yellowbrick.exceptions import YellowbrickTypeError


##########################################################################
## Model detection utilities
##########################################################################

def get_model_name(model):
    """
    Detects the model name for a Scikit-Learn model or pipeline

    Parameters
    ----------
    model: class or instance
        The object to determine the name for
    """
    if not is_estimator(model):
        raise YellowbrickTypeError(
            "Cannot detect the model name for non estimator: '{}'".format(
                type(model)
            )
        )

    else:
        if isinstance(model, Pipeline):
            return model.steps[-1][-1].__class__.__name__
        else:
            return model.__class__.__name__


##########################################################################
## Type checking utilities
##########################################################################

def is_estimator(model):
    """
    Determines if a model is an estimator using issubclass and isinstance.

    Parameters
    ----------
    model: class or instance
        The object to test whether or not is a Scikit-Learn estimator.
    """
    if inspect.isclass(model):
        return issubclass(model, BaseEstimator)

    return isinstance(model, BaseEstimator)

# Alias for closer name to isinstance and issubclass
isestimator = is_estimator


def is_classifier(estimator):
    """
    Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator: class or instance
        The object to test whether or not is a Scikit-Learn classifier.

    See also
    --------
    is_classifier
        `sklearn.is_classifier() <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L518>`_
    """
    # TODO: once we make ScoreVisualizer and ModelVisualizer pass through
    # wrappers as in Issue #90, these three lines become unnecessary.
    # NOTE: This must be imported here to avoid recursive import.
    from yellowbrick.base import Visualizer
    if isinstance(estimator, Visualizer):
        return is_classifier(estimator.estimator)

    # Test the _estimator_type property
    return getattr(estimator, "_estimator_type", None) == "classifier"

# Alias for closer name to isinstance and issubclass
isclassifier = is_classifier


def is_regressor(estimator):
    """
    Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    model: class or instance
        The object to test whether or not is a Scikit-Learn regressor.

    See also
    --------
    is_regressor
        `sklearn.is_regressor() <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L531>`_
    """
    # TODO: once we make ScoreVisualizer and ModelVisualizer pass through
    # wrappers as in Issue #90, these three lines become unnecessary.
    # NOTE: This must be imported here to avoid recursive import.
    from yellowbrick.base import Visualizer
    if isinstance(estimator, Visualizer):
        return is_regressor(estimator.estimator)

    # Test the _estimator_type property
    return getattr(estimator, "_estimator_type", None) == "regressor"

# Alias for closer name to isinstance and issubclass
isregressor = is_regressor


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


##########################################################################
## Decorators
##########################################################################

def memoized(fget):
    """
    Return a property attribute for new-style classes that only calls its
    getter on the first access. The result is stored and on subsequent
    accesses is returned, preventing the need to call the getter any more.

    Parameters
    ----------
    fget: function
        The getter method to memoize for subsequent access.

    See also
    --------
    python-memoized-property
        `python-memoized-property <https://github.com/estebistec/python-memoized-property>`_
    """
    attr_name = '_{0}'.format(fget.__name__)

    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)

    return property(fget_memoized)


class docutil(object):
    """
    This decorator can be used to apply the doc string from another function
    to the decorated function. This is used for our single call wrapper
    functions who implement the visualizer API without forcing the user to
    jump through all the hoops. The docstring of both the visualizer and the
    single call wrapper should be identical, this decorator ensures that we
    only have to edit one doc string.

    Usage::

        @docutil(Visualizer.__init__)
        def visualize(*args, **kwargs):
            pass

    The basic usage is that you instantiate the decorator with the function
    whose docstring you want to copy, then apply that decorator to the the
    function whose docstring you would like modified.

    Note that this decorator performs no wrapping of the target function.
    """

    def __init__(self, func):
        """Create a decorator to document other functions with the specified
        function's doc string.

        Parameters
        ----------
        func : function
            The function whose doc string we should decorate with
        """
        self.doc = func.__doc__

    def __call__(self, func):
        """Modify the decorated function with the stored doc string.

        Parameters
        ----------
        func: function
            The function to apply the saved doc string to.
        """
        func.__doc__ = self.doc
        return func
