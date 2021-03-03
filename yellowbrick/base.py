# yellowbrick.base
# Abstract base classes and interface for Yellowbrick.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Fri Jun 03 10:20:59 2016 -0700
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [4a59c49] benjamin@bengfort.com $

"""
Abstract base classes and interface for Yellowbrick.
"""

import math
import warnings
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

from yellowbrick.utils import get_model_name
from yellowbrick.utils.wrapper import Wrapper
from yellowbrick.utils.helpers import check_fitted
from yellowbrick.exceptions import YellowbrickWarning
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickTypeError


##########################################################################
## Base class hierarchy
##########################################################################


class Visualizer(BaseEstimator):
    """
    The root of the visual object hierarchy that defines how yellowbrick
    creates, stores, and renders visual artifacts using matplotlib.

    Inherits from Scikit-Learn's BaseEstimator class.

    The base class for feature visualization and model visualization
    primarily ensures that styling arguments are passed in.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers. Optional keyword
        arguments include:

        =============   =======================================================
        Property        Description
        -------------   -------------------------------------------------------
        size            specify a size for the figure
        color           specify a color, colormap, or palette for the figure
        title           specify the title of the figure
        =============   =======================================================

    Notes
    -----
    Visualizers are objects that learn from data (e.g. estimators), therefore
    they must be ``fit()`` before they can be drawn or used. Visualizers also
    maintain a reference to an ``ax`` object, a Matplotlib Axes where the
    figures are drawn and rendered, as well as to a ``fig`` object, a Matplotlib
    Figure on which the Visualizer will be plotted.
    """

    def __init__(self, ax=None, fig=None, **kwargs):
        self.ax = ax
        self.fig = fig
        self.size = kwargs.pop("size", None)
        self.color = kwargs.pop("color", None)
        self.title = kwargs.pop("title", None)

    ## ////////////////////////////////////////////////////////////////////
    ## Primary Visualizer Properties
    ## ////////////////////////////////////////////////////////////////////

    @property
    def ax(self):
        """
        The matplotlib axes that the visualizer draws upon (can also be a grid
        of multiple axes objects). The visualizer uses :func:`matplotlib.pyplot.gca`
        to create an axes for the user if one has not been specified.
        """
        if not hasattr(self, "_ax") or self._ax is None:
            self._ax = plt.gca()
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = ax

    @property
    def fig(self):
        """
        The matplotlib fig that the visualizer draws upon. The visualizer uses
        the matplotlib method :func:`matplotlib.pyplot.gcf` to create a figure for
        the user if one has not been specified.
        """
        if not hasattr(self, "_fig") or self._fig is None:
            self._fig = plt.gcf()
        return self._fig

    @fig.setter
    def fig(self, fig):
        self._fig = fig

    @property
    def size(self):
        """
        Returns the actual size in pixels as set by matplotlib, or
        the user provided size if available.
        """
        if not hasattr(self, "_size") or self._size is None:
            self._size = self.fig.get_size_inches() * self.fig.dpi
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        if self._size is not None:
            width, height = size
            width_in_inches = width / self.fig.get_dpi()
            height_in_inches = height / self.fig.get_dpi()
            self.fig.set_size_inches(width_in_inches, height_in_inches)

    ## ////////////////////////////////////////////////////////////////////
    ## Estimator interface
    ## ////////////////////////////////////////////////////////////////////

    def fit(self, X, y=None, **kwargs):
        """
        Fits a visualizer to data and is the primary entry point for producing
        a visualization. Visualizers are Scikit-Learn Estimator objects, which
        learn from data in order to produce a visual analysis or diagnostic.
        They can do this either by fitting features related data or by fitting
        an underlying model (or models) and visualizing their results.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the drawing functionality or to the
            Scikit-Learn API. See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        return self

    ## ////////////////////////////////////////////////////////////////////
    ## Visualizer interface
    ## ////////////////////////////////////////////////////////////////////

    def draw(self, **kwargs):
        """
        The fitting or transformation process usually calls draw (not the
        user). This function is implemented for developers to hook into the
        matplotlib interface and to create an internal representation of the
        data the visualizer was trained on in the form of a figure or axes.

        Parameters
        ----------

        kwargs: dict
            generic keyword arguments.

        """
        raise NotImplementedError("Visualizers must implement a drawing interface.")

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        The user calls show and show calls finalize. Developers should
        implement visualizer-specific finalization methods like setting titles
        or axes labels, etc.
        """
        return self.ax

    def show(self, outpath=None, clear_figure=False, **kwargs):
        """
        Makes the magic happen and a visualizer appear! You can pass in a path to
        save the figure to disk with various backends, or you can call it with no
        arguments to show the figure either in a notebook or in a GUI window that
        pops up on screen.

        Parameters
        ----------
        outpath: string, default: None
            path or None. Save figure to disk or if None show in window

        clear_figure: boolean, default: False
            When True, this flag clears the figure after saving to file or
            showing on screen. This is useful when making consecutive plots.

        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        Developers of visualizers don't usually override show, as it is
        primarily called by the user to render the visualization.
        """
        # Ensure that draw has been called
        if self._ax is None:
            warn_message = (
                "{} does not have a reference to a matplotlib.Axes "
                "the figure may not render as expected!"
            )
            warnings.warn(
                warn_message.format(self.__class__.__name__), YellowbrickWarning
            )

        # Finalize the figure
        self.finalize()

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

        if clear_figure:
            self.fig.clear()

        # Return ax to ensure display in notebooks
        return self.ax

    def poof(self, *args, **kwargs):
        """
        This method is deprecated, please use ``show()`` instead.
        """
        warnings.warn(
            "this method is deprecated, please use show() instead", DeprecationWarning
        )
        return self.show(*args, **kwargs)

    ## ////////////////////////////////////////////////////////////////////
    ## Helper Functions
    ## ////////////////////////////////////////////////////////////////////

    def set_title(self, title=None):
        """
        Sets the title on the current axes.

        Parameters
        ----------
        title: string, default: None
            Add title to figure or if None leave untitled.
        """
        title = self.title or title
        if title is not None:
            self.ax.set_title(title)


##########################################################################
## Model Visualizers
##########################################################################


class ModelVisualizer(Visualizer, Wrapper):
    """
    The ModelVisualizer class wraps a Scikit-Learn estimator (usually a
    predictive model like a regressor, classifier, or clusterer) so that all
    functionality that belongs to the estimator can be accessed from the
    visualizer, thereby allowing visualzers to be proxies for model objects,
    simply drawing on behalf of the wrapped model.

    Parameters
    ----------
    estimator : a Scikit-Learn estimator
        A Scikit-Learn estimator to wrap functionality for, usually regressor,
        classifier, or clusterer predictive model. If the estimator is not fitted,
        it is fit when the visualizer is fitted, unless otherwise specified by
        ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined by other Visualizers.

    Notes
    -----
    Model visualizers can wrap either fitted or unfitted models.
    """

    def __init__(self, estimator, ax=None, fig=None, is_fitted="auto", **kwargs):
        self.estimator = estimator
        self.is_fitted = is_fitted
        self.name = get_model_name(self.estimator)

        # Initialize base classes independently
        Wrapper.__init__(self, self.estimator)
        Visualizer.__init__(self, ax=ax, fig=fig, **kwargs)

    def get_params(self, deep=True):
        """
        After v0.24 - scikit-learn is able to determine that ``self.estimator`` is
        nested and fetches its params using ``estimator__param``. This functionality is
        pretty cool but it's a pretty big overhaul to change our "wrapped" estimator API
        to a "nested" estimator API, therefore we override ``get_params`` to flatten out
        the estimator params.
        """
        params = super(ModelVisualizer, self).get_params(deep=deep)
        for param in list(params.keys()):
            if param.startswith("estimator__"):
                params[param[len("estimator__"):]] = params.pop(param)
        return params

    def set_params(self, **params):
        """
        The latest version of scikit-learn is able to determine that ``self.estimator``
        is nested and sets its params using ``estimator__param``. In order to maintain
        the Yellowbrick "wrapped" API, this method finds any params belonging to the
        underlying estimator and sets them directly.
        """
        estimator_keys = list(self.estimator.get_params(deep=False).keys())
        estimator_params = {
            key: params.pop(key)
            for key in estimator_keys
            if key in params
        }

        self.estimator.set_params(**estimator_params)
        return super(ModelVisualizer, self).set_params(**params)

    def fit(self, X, y=None, **kwargs):
        """
        Fits the wrapped estimator so that subclasses that override fit can
        ensure that the estimator is fit using super rather than a direct call
        down to the estimator. Score estimators tend to expect a fitted model.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the drawing functionality or to the
            scikit-learn API. See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        if not check_fitted(self.estimator, is_fitted_by=self.is_fitted):
            self.estimator.fit(X, y, **kwargs)
        return self


##########################################################################
## Score Visualizers
##########################################################################


class ScoreVisualizer(ModelVisualizer):
    """
    The ScoreVisualizer reports the performance of a Scikit-Learn estimator
    (usually a predictive model like a regressor, classifier, or clusterer) in
    a visual manner. They hook into the Scikit-Learn pipeline through the
    ``score(X_test, y_test)`` method, reporting not just a single numeric
    score, but also a visual report of the score in model space.

    Parameters
    ----------
    model : a Scikit-Learn estimator
        A Scikit-Learn estimator to wrap functionality for, usually regressor,
        classifier, or clusterer predictive model. If the estimator is not fitted,
        it is fit when the visualizer is fitted, unless otherwise specified by
        ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizersself.

    Notes
    -----
    Score visualizers can wrap either fitted or unfitted models.
    """

    def score(self, X, y, **kwargs):
        """
        The primary entry point for score visualizers is the score method,
        which makes predictions based on X and scores them relative to y.

        Returns
        -------
        score : float or array-like
            Returns the score of the underlying model, which is model-specific,
            e.g. accuracy for classifiers, R2 for regressors, etc.
        """
        raise NotImplementedError("ScoreVisualizer subclasses should implement score")


##########################################################################
## Multiple Models
##########################################################################


class VisualizerGrid(Visualizer):
    """
    Used as a base class for visualizers that use subplots.

    Parameters
    ----------
    visualizers : A list of instantiated visualizers

    nrows: integer, default: None
        The number of rows desired, if you would like a fixed number of rows.
        Specify only one of nrows and ncols, the other should be None. If you
        specify nrows, there will be enough columns created to fit all the
        visualizers specified in the visualizers list.

    ncols: integer, default: None
        The number of columns desired, if you would like a fixed number of columns.
        Specify only one of nrows and ncols, the other should be None. If you
        specify ncols, there will be enough rows created to fit all the
        visualizers specified in the visualizers list.

    axarr: matplotlib.axarr, default: None.
        If you want to put the plot onto an existing axarr, specify it here. Otherwise
        a new one will be created.

    kwargs : additional keyword arguments, default: None
        Any additional keyword arguments will be passed on to the fit() method and
        therefore passed on to the fit() method of the wrapped estimators, if
        applicable. Otherwise ignored.

    Examples
    --------
    >>> from yellowbrick.base import VisualizerGrid
    >>> from sklearn.linear_model import LogisticRegression
    >>> from yellowbrick.classifier import ConfusionMatrix
    >>> from yellowbrick.classifier import ClassBalance
    >>> model = LogisticRegression()
    >>> visualizers = [ClassBalance(model),ConfusionMatrix(model)]
    >>> mv = VisualizerGrid(visualizers, ncols=2)
    >>> mv.fit(X_train, y_train)
    >>> mv.score(X_test, y_test)
    >>> mv.show()
    """

    def __init__(self, visualizers=[], nrows=None, ncols=None, axarr=None, **kwargs):
        # Class static params
        self.SUBPLOT_DEFAULT_PIXELS = 400

        # Allocate passed parameters
        self._visualizers = visualizers
        plotcount = len(visualizers)
        if nrows is None and ncols is None:
            # TODO: enhancement would be to also allow a 2-d array  of visualizers
            # instead of just a 1-d left-to-right + top-to-bottom list
            self.ncols = 1
            self.nrows = plotcount
        elif ncols is None:
            self.nrows = nrows
            self.ncols = int(math.ceil(plotcount / self.nrows))
        elif nrows is None:
            self.ncols = ncols
            self.nrows = int(math.ceil(plotcount / self.ncols))
        else:
            raise YellowbrickValueError(
                "You can only specify either nrows or ncols, \
                the other will be calculated based on the length of the list of \
                visualizers."
            )

        if axarr is None:
            fig, axarr = plt.subplots(self.nrows, self.ncols, squeeze=False)

        self.axarr = axarr

        idx = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                try:
                    self.visualizers[idx].ax = self.axarr[row, col]
                # If len(visualizers) isn't evenly divisibly by rows/columns,
                # we want to create the illusion of empty space by hiding the axis
                except IndexError:
                    self.axarr[row, col].axis("off")

                idx += 1

        self.kwargs = kwargs

    @property
    def visualizers(self):
        return self._visualizers

    @visualizers.setter
    def visualizers(self, value):
        raise AttributeError(
            "Visualizers list can only be set during class instantiation."
        )

    @property
    def ax(self):
        """
         Override Visualizer.ax to return the current axis
         """
        return plt.gca()

    @ax.setter
    def ax(self, ax):
        raise YellowbrickTypeError(
            "cannot set new axes objects on multiple visualizers"
        )

    def fit(self, X, y, **kwargs):

        for vz in self.visualizers:
            vz.fit(X, y, **kwargs)

        return self

    def score(self, X, y):

        for idx in range(len(self.visualizers)):
            self.visualizers[idx].score(X, y)

        return self

    def show(self, outpath=None, clear_figure=False, **kwargs):

        if self.axarr is None:
            return

        # Finalize all visualizers
        for idx in range(len(self.visualizers)):
            self.visualizers[idx].finalize()

        # Choose a reasonable default size if the user has not manually specified one
        # self.size() uses pixels rather than matplotlib's default of inches
        if not hasattr(self, "_size") or self._size is None:
            self._width = self.SUBPLOT_DEFAULT_PIXELS * self.ncols
            self._height = self.SUBPLOT_DEFAULT_PIXELS * self.nrows
            self.size = (self._width, self._height)

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

        if clear_figure:
            plt.gcf().clear()

        # Return Axes array to ensure show works in notebooks
        return self.axarr
