# yellowbrick.classifier.threshold
# Threshold classifier visualizer for Yellowbrick.
#
# Author:   Nathan Danielsen <ndanielsen@gmail.com.com>
# Created:  Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
import bisect

import numpy as np
from scipy.stats import mstats

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style.colors import resolve_colors
from yellowbrick.base import ModelVisualizer
from yellowbrick.utils import isclassifier


##########################################################################
# Quick Methods
##########################################################################


def thresholdviz(model,
                 X,
                 y,
                 color=None,
                 n_trials=50,
                 test_size_percent=0.1,
                 quantiles=(0.1, 0.5, 0.9),
                 random_state=0,
                 **kwargs):
    """Quick method for ThresholdVisualizer.
    Visualizes the bounds of precision, recall and queue rate at different
    thresholds for binary targets after a given number of trials.

    The visualization shows the threshold precentage on the x-axis which can be
    compared against the queue rate, precision, and recall as percentages on
    the y-axis. The default that each of the medium curves is set at the 90%%
    central interval, but can be adjusted.

    This visualization will help the user determine given their tolerances for
    precision, queue and recall the appropriate threshold to set in their
    application.

    See also::
        ``http://blog.insightdatalabs.com/visualizing-classifier-thresholds/``

    Parameters
    ----------

    model : a Scikit-Learn classifier, required
        Should be an instance of a classifier otherwise a will raise a
        YellowbrickTypeError exception on instantiation.

    color : string, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis

    n_trials : integer, default: 50
        Number of trials to conduct via train_test_split

    quantiles : sequence, default: (0.1, 0.5, .9)
        Setting the quantiles for visualizing model variability using
        scipy.stats.mstats.mquantiles

    random_state : integer, default: None
        Random state integer for sampling in train_test_split

    kwargs : keyword arguments passed to the super class.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = ThresholdVisualizer(
        model,
        color=color,
        n_trials=n_trials,
        test_size_percent=test_size_percent,
        quantiles=quantiles,
        random_state=random_state,
        **kwargs)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit_poof(X, y)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
# Static ThresholdVisualizer Visualizer
##########################################################################


class ThresholdVisualizer(ModelVisualizer):
    """Visualizes the bounds of precision, recall and queue rate at different
    thresholds for binary targets after a given number of trials.

    The visualization shows the threshold precentage on the x-axis which can be
    compared against the queue rate, precision, and recall as percentages on
    the y-axis. The default that each of the medium curves is set at the 90%%
    central interval, but can be adjusted.

    This visualization will help the user determine given their tolerances for
    precision, queue and recall the appropriate threshold to set in their
    application.

    See also::
        ``http://blog.insightdatalabs.com/visualizing-classifier-thresholds/``

    Parameters
    ----------

    model : a Scikit-Learn classifier, required
        Should be an instance of a classifier otherwise a will raise a
        YellowbrickTypeError exception on instantiation.

    color : string, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis

    n_trials : integer, default: 50
        Number of trials to conduct via train_test_split

    quantiles : sequence, default: (0.1, 0.5, .9)
        Setting the quantiles for visualizing model variability using
        scipy.stats.mstats.mquantiles

    random_state : integer, default: None
        Random state integer for sampling in train_test_split

    kwargs : keyword arguments passed to the super class.
    """

    def __init__(self,
                 model,
                 n_trials=50,
                 test_size_percent=0.1,
                 quantiles=(0.1, 0.5, 0.9),
                 random_state=None,
                 **kwargs):
        # Check to see if model is an instance of a classifier.
        # Should return an error if it isn't.
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
            )
        super(ThresholdVisualizer, self).__init__(model, **kwargs)

        self.estimator = model
        self.n_trials = n_trials
        self.test_size_percent = test_size_percent
        self.quantiles = quantiles
        self.random_state = random_state

        # to be set later
        self.plot_data = None

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the visualizer
        """
        self.plot_data = []

        for _ in range(self.n_trials):
            train_X, test_X, train_y, test_y = train_test_split(
                X,
                y,
                test_size=self.test_size_percent,
                random_state=self.random_state # defaults to None
                )
            self.estimator.fit(train_X, train_y)
            # get prediction probabilities for each
            predictions = self.estimator.predict_proba(test_X)[:, 1]

            precision, recall, thresholds = precision_recall_curve(
                test_y, predictions)
            # add one to each so that thresh ends at 1
            thresholds = np.append(thresholds, 1)
            queue_rate = []
            for threshold in thresholds:
                queue_rate.append((predictions >= threshold).mean())

            trial_data = {
                'thresholds': thresholds,
                'precision': precision,
                'recall': recall,
                'queue_rate': queue_rate
            }
            self.plot_data.append(trial_data)

        return self.draw()

    def draw(self, *kwargs):
        """
        Renders the visualization

        Parameters
        ----------
        kwargs: dict
            keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self.ax : AxesSubplot of the visualizer
            Returns the AxesSubplot instance of the visualizer
        """
        # Set the colors from the supplied values or reasonable defaults
        color_values = resolve_colors(n_colors=3, colors=self.color)

        uniform_thresholds = np.linspace(0, 1, num=101)
        uniform_precision_plots = []
        uniform_recall_plots = []
        uniform_queue_rate_plots = []

        for data in self.plot_data:
            uniform_precision = []
            uniform_recall = []
            uniform_queue_rate = []
            for ut in uniform_thresholds:
                index = bisect.bisect_left(data['thresholds'], ut)
                uniform_precision.append(data['precision'][index])
                uniform_recall.append(data['recall'][index])
                uniform_queue_rate.append(data['queue_rate'][index])

            uniform_precision_plots.append(uniform_precision)
            uniform_recall_plots.append(uniform_recall)
            uniform_queue_rate_plots.append(uniform_queue_rate)

        uplots = (uniform_precision_plots, uniform_recall_plots, uniform_queue_rate_plots)

        for uniform_plot, color in zip(uplots, color_values):
            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(uniform_plot, prob=self.quantiles, axis=0)

            # Draw the median line
            self.ax.plot(uniform_thresholds, median, color=color)

            # Draw the fill between the lower and upper bounds
            self.ax.fill_between(uniform_thresholds, upper, lower, alpha=0.5, linewidth=0, color=color)

        return self.ax

    def finalize(self, **kwargs):
        """Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.
        """
        super(ThresholdVisualizer, self).finalize(**kwargs)

        # Set the title
        if self.title is None:
            self.set_title("Threshold Plot of Binary Classifier")

        self.ax.legend(
            ('precision', 'recall', 'queue_rate'), frameon=True, loc='best')
        self.ax.set_xlabel('threshold')
        self.ax.set_ylabel('percent')

    def fit_poof(self, X, y=None, **kwargs):
        """Convience method to fit, draw and poof / finalize the visualizer in
        one step after instantiation.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the visualizer
        """
        self.fit(X, y)
        self.poof()
        return self

ThreshViz = ThresholdVisualizer
