# yellowbrick.model_selection.cross_validation
# Implements cross-validation score plotting for model selection.
#
# Author:   Prema Damodaran Roman
# Created:  Wed June 6 2018 13:32:00 -0500
# Author:   Rebecca Bilbro
# Updated:  Fri Aug 10 13:15:43 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: cross_validation.py [7f47800] pdamo24@gmail.com $

"""
Implements cross-validation score plotting for model selection.
"""

##########################################################################
# Imports
##########################################################################

import numpy as np
import matplotlib.ticker as ticker

from yellowbrick.base import ModelVisualizer
from sklearn.model_selection import cross_val_score


##########################################################################
# CVScores Visualizer
##########################################################################


class CVScores(ModelVisualizer):
    """
    CVScores displays cross-validated scores as a bar chart, with the
    average of the scores plotted as a horizontal line.

    Parameters
    ----------

    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.
        Note that the object is cloned for each validation.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        See the scikit-learn `cross-validation guide <https://goo.gl/FS3VU6>`_
        for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

        See scikit-learn `cross-validation guide <https://goo.gl/FS3VU6>`_
        for more information on the possible metrics that can be used.

    color: string
        Specify color for barchart

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    -------
    cv_scores_ : ndarray shape (n_splits, )
        The cross-validated scores from each subsection of the data

    cv_scores_mean_ : float
        Average cross-validated score across all subsections of the data


    Examples
    --------

    >>> from sklearn import datasets, svm
    >>> iris = datasets.load_iris()
    >>> clf = svm.SVC(kernel='linear', C=1)
    >>> X = iris.data
    >>> y = iris.target
    >>> visualizer = CVScores(estimator=clf, cv=5, scoring='f1_macro')
    >>> visualizer.fit(X,y)
    >>> visualizer.show()

    Notes
    -----

    This visualizer is a wrapper for
    `sklearn.model_selection.cross_val_score <https://goo.gl/4v7dfL>`_.

    Refer to the scikit-learn
    `cross-validation guide <https://goo.gl/FS3VU6>`_
    for more details.

    """

    def __init__(self, estimator, ax=None, cv=None, scoring=None, color=None, **kwargs):
        super(CVScores, self).__init__(estimator, ax=ax, **kwargs)

        self.cv = cv
        self.color = color
        self.scoring = scoring

    def fit(self, X, y, **kwargs):
        """
        Fits the learning curve with the wrapped model to the specified data.
        Draws training and test score curves and saves the scores to the
        estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : instance

        """

        self.cv_scores_ = cross_val_score(
            self.estimator, X, y, cv=self.cv, scoring=self.scoring
        )
        self.cv_scores_mean_ = self.cv_scores_.mean()

        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Creates the bar chart of the cross-validated scores generated from the
        fit method and places a dashed horizontal line that represents the
        average value of the scores.
        """

        width = kwargs.pop("width", 0.3)
        linewidth = kwargs.pop("linewidth", 1)

        xvals = np.arange(1, len(self.cv_scores_) + 1, 1)
        self.ax.bar(xvals, self.cv_scores_, width=width, color=self.color)
        self.ax.axhline(
            self.cv_scores_mean_,
            color=self.color,
            label="Mean score = {:0.3f}".format(self.cv_scores_mean_),
            linestyle="--",
            linewidth=linewidth,
        )

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """

        # Set the title of the figure
        self.set_title("Cross Validation Scores for {}".format(self.name))

        # Add the legend
        loc = kwargs.pop("loc", "best")
        edgecolor = kwargs.pop("edgecolor", "k")
        self.ax.legend(frameon=True, loc=loc, edgecolor=edgecolor)

        # set spacing between the x ticks
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        # Set the axis labels
        self.ax.set_xlabel("Training Instances")
        self.ax.set_ylabel("Score")


##########################################################################
# Quick Method
##########################################################################


def cv_scores(
    estimator, X, y, ax=None, cv=None, scoring=None, color=None, show=True, **kwargs
):
    """
    Displays cross validation scores as a bar chart and the
    average of the scores as a horizontal line

    This helper function is a quick wrapper to utilize the
    CVScores visualizer for one-off analysis.

    Parameters
    ----------

    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.
        Note that the object is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

    see the scikit-learn
    `cross-validation guide <https://goo.gl/FS3VU6>`_
    for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

        See scikit-learn `cross-validation guide <https://goo.gl/FS3VU6>`_
        for more information on the possible metrics that can be used.

    color: string
        Specify color for barchart

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    visualizer : CVScores
        The fitted visualizer.

    """

    # Initialize the visualizer
    visualizer = CVScores(
        estimator, ax=ax, cv=cv, scoring=scoring, color=None, **kwargs
    )

    # Fit and show the visualizer
    visualizer.fit(X, y)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer object
    return visualizer
