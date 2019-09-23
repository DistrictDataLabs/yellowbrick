# yellowbrick.text.base
# Base classes for text feature visualizers and feature selection tools.
#
# Author:   Rebecca Bilbro
# Created:  Sat Jan 21 09:37:01 2017 -0500
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [75d9b20] rebecca.bilbro@bytecubed.com $

"""
Base classes for text feature visualizers and text feature selection tools.
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.base import Visualizer
from sklearn.base import TransformerMixin


##########################################################################
## Text Visualizers
##########################################################################


class TextVisualizer(Visualizer, TransformerMixin):
    """
    Base class for text feature visualization to investigate documents
    individually or as a full corpus.

    TextVisualizers are used after a text corpus has been transformed
    in some way (e.g. normalized through stemming or lemmatization, via
    stopwords removal, or through vectorization). Thus a TextVisualizer
    is itself a transformer and can be used in a Scikit-Learn Pipeline
    to perform automatic visual analysis during build.

    Accepts as input a DataFrame or Numpy array.
    """

    def __init__(self, ax=None, fig=None, **kwargs):
        """
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        Parameters
        ----------
        ax : axes
            the axis to plot the figure on

        fig : matplotlib Figure, default: None
            The figure to plot the Visualizer on. If None is passed in the current
            plot will be used (or generated if required).

        kwargs : dict
            Pass generic arguments to the drawing method

        """
        super(TextVisualizer, self).__init__(ax=ax, fig=fig, **kwargs)

    def fit(self, X, y=None, **fit_params):
        """
        This method performs preliminary computations in order to set up the
        figure, compute statistics, or perform other analyses. It can also
        call drawing methods in order to set up various non-instance-related
        figure elements.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        fit_params: dict
            keyword arguments for parameter fitting.

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        return self

    def transform(self, X):
        """
        Primarily a pass-through to ensure that the text visualizer will
        work in a pipeline setting. This method can also call drawing methods
        in order to ensure that the visualization is constructed.

        Returns
        -------
        X : numpy array
            This method must return a numpy array with the same shape as X.
        """
        return X

    def fit_transform_show(self, X, y=None, **kwargs):
        """
        Fit to data, transform it, then visualize it.

        Fits the text visualizer to X and y with optional parameters by
        passing in all of kwargs, then calls show with the same kwargs.
        This method must return the result of the transform method.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the show method

        Returns
        -------
        X : numpy array
            This method must return a numpy array with the same shape as X.
        """
        Xp = self.fit(X, y, **kwargs).transform(X)
        self.show(**kwargs)
        return Xp
