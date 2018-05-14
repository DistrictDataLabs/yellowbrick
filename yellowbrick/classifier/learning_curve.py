# yellowbrick.classifier.learning_curve
# Implements a learning curve visualization for classification
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
# Created:  Mon May 22 09:22:00 2017 -0500
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: learning_curve.py [] jason.s.keung@gmail.com $

"""
Implementations a learning curve visualizer for classification.
"""

##########################################################################
## Imports
##########################################################################
import numpy as np

from sklearn.model_selection import learning_curve

from yellowbrick.base import ModelVisualizer
from yellowbrick.exceptions import YellowbrickError

##########################################################################
## LearningCurveVisualizer Visualizer
##########################################################################

class LearningCurveVisualizer(ModelVisualizer):
    """
    Generate a simple plot of the test and training learning curve.
    Learning curves demonstrate is a plot of proxy measures for implied
    learning with experience.

    * The X axis represents experience, or the number of training samples.
    * The Y axis represents learning, or the train and cross validation scores.

    Parameters
    ----------

    model : a Scikit-Learn estimator

    train_sizes: ndarray or Series, default: np.linspace(.1, 1.0, 5)
        An array that represents the proportion of data for the learning curve

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

    see scikit-learn `cross-validation guide <http://scikit-learn.org/stable/modules/cross_validation.html>`_
    for more information

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.classifier import LearningCurveVisualizer
    >>> from sklearn.naive_bayes import GaussianNB
    >>> model = LearningCurveVisualizer(GaussianNB())
    >>> model.fit(X, y)
    >>> model.poof()
    """
    def __init__(self, model, train_sizes=None, cv=None, n_jobs=1, **kwargs):

        # Call super to initialize the class
        super(LearningCurveVisualizer, self).__init__(model, **kwargs)

        # Set parameters
        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = np.linspace(.1, 1.0, 5) if train_sizes is None else train_sizes

        if not (isinstance(self.train_sizes, np.ndarray)):
            raise YellowbrickError('train_sizes must be np.ndarray or pd.Series')

        # to be set later
        self.train_scores = None
        self.test_scores = None
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores_mean = None
        self.test_scores_std = None

    def fit(self, X, y, **kwargs):
        """
        The fit method is the primary drawing input for the learning curve
        visualization since it has both the X and y data required for the
        viz.

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the learning curve visualizer
        """
        self.train_sizes, self.train_scores, self.test_scores = learning_curve(
                self.estimator, X, y, cv=self.cv, n_jobs=self.n_jobs, train_sizes=self.train_sizes)

        self.train_scores_mean = np.mean(self.train_scores, axis=1)
        self.train_scores_std = np.std(self.train_scores, axis=1)
        self.test_scores_mean = np.mean(self.test_scores, axis=1)
        self.test_scores_std = np.std(self.test_scores, axis=1)

        self.draw(**kwargs)

        return self

    def draw(self, **kwargs):
        """
        Renders the learning curve across each axis.

        Parameters
        ----------

        kwargs: keyword arguments passed to Scikit-Learn API.
        """
        self.ax.fill_between(self.train_sizes, self.train_scores_mean - self.train_scores_std,
                         self.train_scores_mean + self.train_scores_std, alpha=0.1,
                         color='b')

        self.ax.fill_between(self.train_sizes, self.test_scores_mean - self.test_scores_std,
                         self.test_scores_mean + self.test_scores_std, alpha=0.1, color='g')

        self.ax.plot(self.train_sizes, self.train_scores_mean, 'o-', color='b',
                 label="Training Score")

        self.ax.plot(self.train_sizes, self.test_scores_mean, 'o-', color='g',
                 label="Cross-validation Score")

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.
        """
        # Set title if no title provided
        self.set_title('Learning Curve for {}'.format(self.name))
        self.ax.legend(('Training Score', 'Cross-validation Score'), frameon=True, loc='best')
        self.ax.set_xlabel('Training Samples')
        self.ax.set_ylabel('Score')

##########################################################################
## Quick Methods
##########################################################################

def learning_curve_plot(X, y, model, ax=None, train_sizes=None,
           cv=None, n_jobs=1, **kwargs):
    """
    Displays a learning curve based on number of samples vs training and
    cross validation scores. The learning curve aims to show how a model
    learns and improves with experience.

    This helper function is a quick wrapper to utilize the LearningCurveVisualizer
    for one-off analysis.

    Parameters
    ----------

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    model : a Scikit-Learn estimator

    train_sizes: ndarray or Series, default: np.linspace(.1, 1.0, 5)
        An array that represents the proportion of data for the learning curve

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the learning curve were drawn on.
    """

    # Instantiate the visualizer
    visualizer = LearningCurveVisualizer(model, train_sizes, cv, n_jobs, **kwargs)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)

    # Return the axes object on the visualizer
    return visualizer.ax
