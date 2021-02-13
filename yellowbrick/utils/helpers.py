# yellowbrick.utils.helpers
# Helper functions and generic utilities for use in Yellowbrick code.
#
# Author:   Benjamin Bengfort
# Author:   Rebecca Bilbro
# Created:  Fri May 19 10:39:30 2017 -0700
#
# Copyright (C) 2019 The scikit-yb developers
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
import inspect
import sklearn
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from yellowbrick.utils.types import is_estimator
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.contrib.wrapper import ContribEstimator


##########################################################################
## Model and Feature Information
##########################################################################

def is_fitted(estimator):
    """
    In order to ensure that we don't call ``fit`` on an already-fitted model,
    this utility function calls ``predict`` on the estimator, returning ``False``
    if it raises a ``sklearn.exceptions.NotFittedError`` and ``True`` otherwise.

    NOTE: This is the solution proposed to scikit-yb: https://bit.ly/2LWQxZO (see
    also: https://stackoverflow.com/a/39900933/6552250), though it remains unclear
    how it will perform with sklearn-style Estimators and Transformers from other
    3rd party libraries like Keras, XGBoost, etc.
    """
    try:
        estimator.predict(np.zeros((7, 3)))
    except sklearn.exceptions.NotFittedError:
        return False
    except AttributeError:
        # Some clustering models (LDA, PCA, Agglomerative) don't implement ``predict``
        try:
            check_is_fitted(
                estimator,
                [
                    "coef_",
                    "estimator_",
                    "labels_",
                    "n_clusters_",
                    "children_",
                    "components_",
                    "n_components_",
                    "n_iter_",
                    "n_batch_iter_",
                    "explained_variance_",
                    "singular_values_",
                    "mean_",
                ],
                all_or_any=any,
            )
            return True
        except sklearn.exceptions.NotFittedError:
            return False
    except Exception:
        # Assume it's fitted, since ``NotFittedError`` wasn't raised
        return True

    return True


def check_fitted(estimator, is_fitted_by="auto", **kwargs):
    """
    Determines whether or not to check if the model has been fitted, and will return
    ``True`` if so. The ``is_fitted_by`` argument is set to ``'auto'`` by default,
    such that the check leaves it to the ``is_fitted`` helper method to determine if
    a ``NotFitted`` error is raised. However, if the user prefers to override this
    automatic functionality (e.g. if a 3rd party sklearn-like estimator has been used
    that doesn't precisely implement the sklearn API), and ``is_fitted_by`` has been
    set to either ``True`` or ``False``, we assume the user has supplied the necessary
    information about whether or not the model is fit using the Visualizer's
    ``is_fitted`` parameter.

    .. todo:: add other measures for checking if an estimator is fitted e.g. by coefs

    Parameters
    -----------
    estimator : sklearn.Estimator
        The model to check fittedness

    is_fitted_by : bool or str, default: 'auto'
        If bool, that value is returned, otherwise ``is_fitted`` is used to check
        for an exception

    kwargs : dict
        Other optional parameters specific to the ``is_fitted_by`` mechanism.

    Returns
    --------
    is_fitted : bool
        Whether or not the model is already fitted
    """
    if isinstance(is_fitted_by, str) and is_fitted_by.lower() == "auto":
        return is_fitted(estimator)
    return bool(is_fitted_by)


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
            "Cannot detect the model name for non estimator: '{}'".format(type(model))
        )

    if isinstance(model, Pipeline):
        return get_model_name(model.steps[-1][-1])
    elif isinstance(model, ContribEstimator):
        return model.estimator.__class__.__name__
    else:
        return model.__class__.__name__


def has_ndarray_int_columns(features, X):
    """ Checks if numeric feature columns exist in ndarray """
    _, ncols = X.shape
    if not all(d.isdigit() for d in features if isinstance(d, str)) or not isinstance(
        X, np.ndarray
    ):
        return False
    ndarray_columns = np.arange(0, ncols)
    feature_cols = np.unique([int(d) for d in features])
    return all(np.in1d(feature_cols, ndarray_columns))


# Alias for closer name to isinstance and issubclass
hasndarrayintcolumns = has_ndarray_int_columns


def is_monotonic(a, increasing=True):
    """
    Tests whether a vector a has monotonicity.

    Parameters
    ----------
    a : array-like
        Array that should be tested for monotonicity

    increasing : bool, default: True
        Test if the array is montonically increasing, otherwise test if the
        array is montonically decreasing.
    """
    a = np.asarray(a)  # ensure a is array-like

    if a.ndim > 1:
        raise ValueError("not supported for multi-dimensonal arrays")

    if len(a) <= 1:
        return True

    if increasing:
        return np.all(a[1:] >= a[:-1], axis=0)
    return np.all(a[1:] <= a[:-1], axis=0)


def get_param_names(method):
    """
    Returns a list of keyword-only parameter names that may be
    passed into method.

    Parameters
    ----------
    method : function
        The method for which to return keyword-only parameters.

    Returns
    -------
    parameters : list
        A list of keyword-only parameter names for method.
    """
    try:
        signature = inspect.signature(method)
    except (ValueError, TypeError) as e:
        raise e

    parameters = [
        p
        for p in signature.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD
    ]

    return sorted([p.name for p in parameters])


##########################################################################
## Numeric Computations
##########################################################################

# From here: https://bit.ly/2xR64lI
def div_safe(numerator, denominator):
    """
    Ufunc-extension that returns 0 instead of nan when dividing numpy arrays

    Parameters
    ----------
    numerator: array-like

    denominator: scalar or array-like that can be validly divided by the numerator

    returns a numpy array

    example: div_safe( [-1, 0, 1], 0 ) == [0, 0, 0]
    """
    # First handle scalars
    if np.isscalar(numerator):
        raise ValueError("div_safe should only be used with an array-like numerator")

    # Then numpy arrays
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.true_divide(numerator, denominator)
            result[~np.isfinite(result)] = 0  # -inf inf NaN
        return result
    except ValueError as e:
        raise e


def prop_to_size(vals, mi=0.0, ma=5.0, power=0.5, log=False):
    """
    Converts an array of property values (e.g. a metric or score) to values
    that are more useful for marker sizes, line widths, or other visual
    sizes. The new sizes are computed as:

        y = mi + (ma -mi)(\frac{x_i - min(x){max(x) - min(x)})^{power}

    If ``log=True``, the natural logarithm of the property values is used instead.

    Parameters
    ----------
    prop : array-like, 1D
        An array of values of the property to scale between the size range.

    mi : float, default: 0.0
        The size to assign the smallest property (minimum size value).

    ma : float, default: 5.0
        The size to assign the largest property (maximum size value).

    power : float, default: 0.5
        Used to control how rapidly the size increases from smallest to largest.

    log : bool, default: False
        Use the natural logarithm to compute the property sizes

    Returns
    -------
    sizes : array, 1D
        The new size values, in the same shape as the input prop array
    """
    # ensure that prop is an array
    vals = np.asarray(vals)

    # apply natural log if specified
    if log:
        vals = np.log(vals)

    # avoid division by zero error
    delta = vals.max() - vals.min()
    if delta == 0.0:
        delta = 1.0

    return mi + (ma - mi) * ((vals - vals.min()) / delta) ** power


##########################################################################
# String Computations
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

    .. seealso:: https://bit.ly/2NW7s1j
    """
    slug = re.sub(r"[^\w]+", " ", text)
    slug = "-".join(slug.lower().strip().split())
    return slug
