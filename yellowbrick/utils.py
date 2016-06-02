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
