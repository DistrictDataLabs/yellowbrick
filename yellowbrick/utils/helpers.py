# yellowbrick.utils.helpers
# Helper functions and generic utilities for use in Yellowbrick code.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri May 19 10:39:30 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: helpers.py [79cd8cf] benjamin@bengfort.com $

"""
Helper functions and generic utilities for use in Yellowbrick code.
"""

##########################################################################
## Imports
##########################################################################

import re
import numpy as np

from sklearn.pipeline import Pipeline

from .types import is_estimator
from yellowbrick.exceptions import YellowbrickTypeError


##########################################################################
## Model and Feature Information
##########################################################################

def get_model_name(model):
    """
    Detects the model name for a Scikit-Learn model or pipeline.

    Parameters
    ----------
    model: class or instance
        The object to determine the name for. If the model is an estimator it
        returns the class name; if it is a Pipeline it returns the class name
        of the final transformer or estimator in the Pipeline.

    Returns
    -------
    name : string
        The name of the model or pipeline.
    """
    if not is_estimator(model):
        raise YellowbrickTypeError(
            "Cannot detect the model name for non estimator: '{}'".format(
                type(model)
            )
        )

    else:
        if isinstance(model, Pipeline):
            return get_model_name(model.steps[-1][-1])
        else:
            return model.__class__.__name__


def has_ndarray_int_columns(features, X):
    """ Checks if numeric feature columns exist in ndarray """
    _, ncols = X.shape
    if not all(d.isdigit() for d in features if isinstance(d, str)) or not isinstance(X, np.ndarray):
        return False
    ndarray_columns = np.arange(0, ncols)
    feature_cols = np.unique([int(d) for d in features])
    return all(np.in1d(feature_cols, ndarray_columns))

# Alias for closer name to isinstance and issubclass
hasndarrayintcolumns = has_ndarray_int_columns


##########################################################################
## Numeric Computations
##########################################################################

#From here: http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
def div_safe( numerator, denominator ):
    """
    Ufunc-extension that returns 0 instead of nan when dividing numpy arrays

    Parameters
    ----------
    numerator: array-like

    denominator: scalar or array-like that can be validly divided by the numerator

    returns a numpy array

    example: div_safe( [-1, 0, 1], 0 ) == [0, 0, 0]
    """
    #First handle scalars
    if np.isscalar(numerator):
        raise ValueError("div_safe should only be used with an array-like numerator")

    #Then numpy arrays
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide( numerator, denominator )
            result[ ~ np.isfinite( result )] = 0  # -inf inf NaN
        return result
    except ValueError as e:
        raise e


##########################################################################
## String Computations
##########################################################################

def slugify(text):
    """
    Returns a slug of given text, normalizing unicode data for file-safe
    strings. Used for deciding where to write images to disk.

    Parameters
    ----------
    text : string
        The string to slugify

    Returns
    -------
    slug : string
        A normalized slug representation of the text

    .. seealso:: http://yashchandra.com/2014/05/08/how-to-generate-clean-url-or-a-slug-in-python/
    """
    slug = re.sub(r'[^\w]+', ' ', text)
    slug = "-".join(slug.lower().strip().split())
    return slug
