# utils
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
#           Patrick O'Melveny <pvomelveny@gmail.com>
# Created:  Thurs Jun 2 15:33:18 2016 -0500
#
# For license information, see LICENSE.txt

"""
Utility functions for yellowbrick
"""

##########################################################################
## Imports
##########################################################################

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from yellowbrick.exceptions import YellowbrickTypeError

##########################################################################
## Model detection utilities
##########################################################################

def get_model_name(model):
    """
    Detects the model name for a Scikit-Learn model or pipeline
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
    """
    if type(model) == type:
        return issubclass(model, BaseEstimator)

    return isinstance(model, BaseEstimator)

# Alias for closer name to isinstance and issubclass
isestimator = is_estimator


def is_classifier(estimator):
    """
    Returns True if the given estimator is (probably) a classifier.
    From: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L526
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"

# Alias for closer name to isinstance and issubclass
isclassifier = is_classifier


def is_regressor(estimator):
    """
    Returns True if the given estimator is (probably) a regressor.
    From: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L531
    """
    from yellowbrick.base import Visualizer
    if isinstance(estimator, Visualizer):
        return is_regressor(estimator.estimator)
    return getattr(estimator, "_estimator_type", None) == "regressor"

# Alias for closer name to isinstance and issubclass
isregressor = is_regressor


def is_dataframe(obj):
    """
    Returns True if the given object is a Pandas Data Frame.
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
