# yellowbrick.base
# Abstract base classes and interface for Yellowbrick.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:20:59 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [] benjamin@bengfort.com $

"""
Abstract base classes and interface for Yellowbrick.
"""

from .exceptions import YellowbrickTypeError
from sklearn.base import BaseEstimator, TransformerMixin

##########################################################################
## Base class hierarhcy
##########################################################################

class BaseVisualization(object):
    """
    The root of the visual object hierarchy that defines how yellowbrick
    creates, stores, and renders visual artifcats using matplotlib.
    """

    def render(self):
        """
        Render is the primary entry point for producing the visualization.
        """
        raise NotImplementedError(
            "All visualizations must specify their own render methodology"
        )


class FeatureVisualization(BaseVisualization, BaseEstimator, TransformerMixin):
    """
    A feature visualization class accepts as input a DataFrame or Numpy array
    in order to investigate features individually or together.

    FeatureVisualization is itself a transformer so that it can be used in
    a Scikit-Learn Pipeline to perform automatic visual analysis during build.
    """

    def fit(self, X, y=None, **kwargs):
        pass

    def transform(self, X):
        pass

    def render(self, data=None):
        """
        A feature visualization renders data.
        """
        raise NotImplementedError(
            "Please specify how to render the feature visualization"
        )


class ModelVisualization(BaseVisualization, BaseEstimator):
    """
    A model visualization class accepts as input a Scikit-Learn estimator(s)
    and is itself an estimator (to be included in a Pipeline) in order to
    visualize the efficacy of a particular fitted model.
    """

    def fit(self, X, y=None, **kwargs):
        pass

    def predict(self, X):
        pass

    def render(self, model=None):
        """
        A model visualization renders a model
        """
        raise NotImplementedError(
            "Please specify how to render the model visualization"
        )
