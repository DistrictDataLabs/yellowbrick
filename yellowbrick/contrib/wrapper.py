# yellowbrick.contrib.wrapper
# Wrapper for third-party estimators that implement the sklearn API
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 02 14:47:54 2020 -0400
#
# Copyright (C) 2020 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: wrapper.py [] benjamin@bengfort.com $

"""
Wrapper for third-party estimators that implement the sklearn API but do not directly
subclass the ``sklearn.base.BaseEstimator`` class. This method is a quick way to get
other estimators into Yellowbrick, while avoiding weird errors and issues.
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.exceptions import YellowbrickAttributeError


##########################################################################
## Module Constants
##########################################################################

CLASSIFIER = "classifier"
REGRESSOR = "regressor"
CLUSTERER = "clusterer"
DENSITY_ESTIMATOR = "DensityEstimator"
OUTLIER_DETECTOR = "outlier_detector"


##########################################################################
## Functional API
##########################################################################

def wrap(estimator, estimator_type=None):
    """
    Wrap a third-party estimator that implements portions of the scikit-learn API to
    make it available to Yellowbrick visualizers. If the Yellowbrick visualizer cannot
    succeed, then a sensible error is raised instead.

    Parameters
    ----------
    estimator : object
        The non-sklearn estimator to wrap and use for Visualizers

    estimator_type : str, optional
        One of "classifier", "regressor", "clusterer", "DensityEstimator", or
        "outlier_detector" that allows the contrib estimator to pass the scikit-learn
        ``is_classifier``, etc. functions. If not specified, the _estimator_type attr
        is passed through to the underlying estimator.
    """
    return ContribEstimator(estimator, estimator_type)


def classifier(estimator):
    """
    Wrap a third-party classifier to make it available to Yellowbrick visualizers.

    Parameters
    ----------
    estimator : object
        The non-sklearn classifier to wrap and use for Visualizers
    """
    return wrap(estimator, CLASSIFIER)


def regressor(estimator):
    """
    Wrap a third-party regressor to make it available to Yellowbrick visualizers.

    Parameters
    ----------
    estimator : object
        The non-sklearn regressor to wrap and use for Visualizers
    """
    return wrap(estimator, REGRESSOR)


def clusterer(estimator):
    """
    Wrap a third-party clusterer to make it available to Yellowbrick visualizers.

    Parameters
    ----------
    estimator : object
        The non-sklearn clusterer to wrap and use for Visualizers
    """
    return wrap(estimator, CLUSTERER)


##########################################################################
## ContribEstimator - Third Pary Estimator Wrapper
##########################################################################

class ContribEstimator(object):
    """
    Wraps a third party estimator that implements the sckit-learn API and therefore
    could be used with Yellowbrick but doesn't subclass ``BaseEstimator``. Since there
    are a number of pitfalls, this object provides sensible errors and warnings rather
    than completely blowing up, allowing contrib users to identify issues and fix them,
    smoothing the path to getting third party estimators into the Yellowbrick ecosystem.

    Parameters
    ----------
    estimator : object
        The non-sklearn estimator to wrap and use for Visualizers

    estimator_type : str, optional
        One of "classifier", "regressor", "clusterer", "DensityEstimator", or
        "outlier_detector" that allows the contrib estimator to pass the scikit-learn
        ``is_classifier``, etc. functions. If not specified, the _estimator_type attr
        is passed through to the underlying estimator.
    """

    def __init__(self, estimator, estimator_type=None):
        self.estimator = estimator
        # Do not set estimator type if not specified to allow passthrough
        if estimator_type:
            self._estimator_type = estimator_type

    def __getattr__(self, attr):
        # proxy to the wrapped object
        try:
            return getattr(self.estimator, attr)
        except AttributeError:
            raise YellowbrickAttributeError((
                "estimator is missing the '{}' attribute, which is required for this "
                "visualizer - please see the third party estimators documentation."
            ).format(attr))
