# yellowbrick.base
# Abstract base classes and interface for Yellowbrick.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:20:59 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [4a59c49] benjamin@bengfort.com $

"""
Abstract base classes and interface for Yellowbrick.
"""

import matplotlib.pyplot as plt
import math

from .utils.wrapper import Wrapper
from sklearn.base import BaseEstimator
from .utils import get_model_name, isestimator
from sklearn.model_selection import cross_val_predict as cvp
from .exceptions import YellowbrickValueError, YellowbrickTypeError

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
    maintain a reference to an ``ax`` object, a matplotlib Axes where the
    figures are drawn and rendered.
    """

    def __init__(self, ax=None, **kwargs):
        self.ax = ax
        self.size  = kwargs.pop('size', None)
        self.color = kwargs.pop('color', None)
        self.title = kwargs.pop('title', None)

    ##////////////////////////////////////////////////////////////////////
    ## Primary Visualizer Properties
    ##////////////////////////////////////////////////////////////////////

    @property
    def ax(self):
        """
        The matplotlib axes that the visualizer draws upon (can also be a grid
        of multiple axes objects). The visualizer automatically creates an
        axes for the user if one has not been specified.
        """
        if not hasattr(self, "_ax") or self._ax is None:
            self._ax = plt.gca()
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = ax

    @property
    def size(self):
        """
        Returns the actual size in pixels as set by matplotlib, or
        the user provided size if available.
        """
        if not hasattr(self, "_size") or self._size is None:
            fig = plt.gcf()
            self._size = fig.get_size_inches()*fig.dpi
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        if self._size is not None:
            fig = plt.gcf()
            width, height = size
            width_in_inches = width / fig.get_dpi()
            height_in_inches = height / fig.get_dpi()
            fig.set_size_inches(width_in_inches, height_in_inches)

    ##////////////////////////////////////////////////////////////////////
    ## Estimator interface
    ##////////////////////////////////////////////////////////////////////

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

    ##////////////////////////////////////////////////////////////////////
    ## Visualizer interface
    ##////////////////////////////////////////////////////////////////////

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
        raise NotImplementedError(
            "Visualizers must implement a drawing interface."
        )

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        The user calls poof and poof calls finalize. Developers should
        implement visualizer-specific finalization methods like setting titles
        or axes labels, etc.
        """
        return self.ax

    def poof(self, outpath=None, **kwargs):
        """
        Poof makes the magic happen and a visualizer appear! You can pass in
        a path to save the figure to disk with various backends, or you can
        call it with no arguments to show the figure either in a notebook or
        in a GUI window that pops up on screen.

        Parameters
        ----------
        outpath: string, default: None
            path or None. Save  figure to disk or if None show in window

        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        Developers of visualizers don't usually override poof, as it is
        primarily called by the user to render the visualization.
        """
        # Ensure that draw has been called
        if self._ax is None: return

        # Finalize the figure
        self.finalize()

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

    ##////////////////////////////////////////////////////////////////////
    ## Helper Functions
    ##////////////////////////////////////////////////////////////////////

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
    model : Estimator
        A Scikit-Learn estimator to wrap functionality for, usually regressor,
        classifier, or clusterer predictive model.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined by other Visualizers.

    Notes
    -----
    Model visualizers can wrap either fitted or unfitted models.
    """

    def __init__(self, model, ax=None, **kwargs):
        """
        Parameters
        ----------
        ax: matplotlib axes
            the axis to plot the figure on.

        kwargs: dict
            keyword arguments for Scikit-Learn model
        """
        self.estimator = model
        self.name = get_model_name(self.estimator)

        Wrapper.__init__(self, self.estimator)
        Visualizer.__init__(self, ax=ax, **kwargs)

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
            Scikit-Learn API. See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        self.estimator.fit(X, y)
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
    model : Estimator
        A Scikit-Learn estimator to wrap functionality for, usually regressor,
        classifier, or clusterer predictive model.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizersself.

    Returns
    -------
    score : float or array-like
        Returns the score of the underlying model, which is model-speciifc,
        e.g. accuracy for classifiers, R2 for regressors, etc.

    Notes
    -----
    Score visualizers can wrap either fitted or unfitted models.
    """

    def score(self, X, y, **kwargs):
        """
        The primary entry point for score visualizers is the score method,
        which makes predictions based on X and scores them relative to y.
        """
        raise NotImplementedError(
            "ScoreVisualizer subclasses should implement score"
        )



##########################################################################
## Multiple Models and Mixins
##########################################################################

class MultiModelMixin(object):
    """
    Does predict for each of the models and generates subplots.
    """

    def __init__(self, models, ax=None, **kwargs):
        # Ensure models is a collection, if it's a single estimator then we
        # wrap it in a list so that the API doesn't break during render.
        """
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        Parameters
        ----------
        models: Scikit-Learn estimator
            the Scikit-Learn models being compared with each other.

        kwargs: dict
            keyword arguments.
        """
        # TODO: How to handle the axes in this mixin?
        self.ax = ax

        if all(isestimator, models):
            models = [models]

        # Keep track of the models
        self.models = models
        self.names  = kwargs.pop('names', list(map(get_model_name, models)))

    def generate_subplots(self):
        """
        Generates the subplots for the number of given models.
        """
        _, axes = plt.subplots(len(self.models), sharex=True, sharey=True)
        return axes

    def predict(self, X, y):
        """
        Returns a generator containing the predictions for each of the
        internal models (using cross_val_predict and a CV=12).

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.

        """
        for model in self.models:
            yield cvp(model, X, y, cv=12)



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
        If you want to put the plot onto an existing axarr, specify it here. Otherwise a new
        one will be created.

    kwargs : additional keyword arguments, default: None
        Any additional keyword arguments will be passed on to the fit() method and therefore
        passed on to the fit() method of the wrapped estimators, if applicable. Otherwise ignored.

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
    >>> mv.poof()
    """
    def __init__(self, visualizers = [], nrows = None, ncols = None, axarr = None, **kwargs):
        #Class static params
        self.SUBPLOT_DEFAULT_PIXELS = 400

        #Allocate passed parameters
        self._visualizers = visualizers
        plotcount = len(visualizers)
        if nrows == None and ncols == None:
            #TODO: enhancement would be to also allow a 2-d array  of visualizers instead of just a 1-d left-to-right + top-to-bottom list
            self.ncols = 1
            self.nrows = plotcount
        elif ncols == None:
            self.nrows = nrows
            self.ncols = int(math.ceil(plotcount / self.nrows))
        elif nrows == None:
            self.ncols = ncols
            self.nrows = int(math.ceil(plotcount / self.ncols))
        else:
            raise YellowbrickValueError("You can only specify either nrows or ncols, \
                the other will be calculated based on the length of the list of visualizers.")


        if axarr == None:
            fig, axarr = plt.subplots(self.nrows, self.ncols, squeeze = False)

        self.axarr = axarr

        idx = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                try:
                    self.visualizers[idx].ax = self.axarr[row, col]
                #If len(visualizers) isn't evenly divisibly by rows/columns,
                #we want to create the illusion of empty space by hiding the axis
                except IndexError:
                    self.axarr[row,col].axis('off')

                idx += 1

        self.kwargs = kwargs

    @property
    def visualizers(self):
        return self._visualizers

    @visualizers.setter
    def visualizers(self,value):
        raise AttributeError("Visualizers list can only be set during class instantiation.")

    @property
    def ax(self):
         """
         Override Visualizer.ax to return the current axis
         """
         return plt.gca()

    @ax.setter
    def ax(self, ax):
         raise YellowbrickTypeError("cannot set new axes objects on multiple visualizers")


    def fit(self,X,y,**kwargs):

        for vz in self.visualizers:
            vz.fit(X,y,**kwargs)

        return self

    def score(self,X,y):

        for idx in range(len(self.visualizers)):
            self.visualizers[idx].score(X,y)

        return self

    def poof(self, outpath=None, **kwargs):

        if self.axarr is None: return

        #Finalize all visualizers
        for idx in range(len(self.visualizers)):
            self.visualizers[idx].finalize()

        #Choose a reasonable default size if the user has not manually specified one
        # self.size() uses pixels rather than matplotlib's default of inches
        if not hasattr(self, "_size") or self._size is None:
            self._width = self.SUBPLOT_DEFAULT_PIXELS * self.ncols
            self._height = self.SUBPLOT_DEFAULT_PIXELS * self.nrows
            self.size = (self._width,self._height);

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()
