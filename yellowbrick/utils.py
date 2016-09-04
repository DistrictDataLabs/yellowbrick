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

##########################################################################
## Model detection utilities
##########################################################################

def get_model_name(model):
    """
    Detects the model name for a Scikit-Learn model or pipeline
    """
    if not isinstance(model, BaseEstimator):
        raise TypeError
    else:
        if isinstance(model, Pipeline):
            return model.steps[-1][-1].__class__.__name__
        else:
            return model.__class__.__name__

def isestimator(model):
    """
    Determines if a model is an estimator using issubclass and isinstance.
    """
    if type(model) == type:
        return issubclass(model, BaseEstimator)

    return isinstance(model, BaseEstimator)
