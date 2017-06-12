# yellowbrick.classifier.threshold
# Threshold classifier visualizer for Yellowbrick.
#
# Author:   Nathan Danielsen <ndanielsen@gmail.com.com>
# Created:  Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from collections import OrderedDict

import bisect
from scipy.stats import mstats

from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve

from yellowbrick.style.palettes import get_color_cycle
from yellowbrick.style.colors import resolve_colors
from yellowbrick.base import ModelVisualizer
from yellowbrick.utils import get_model_name, isclassifier

##########################################################################
# Quick Methods
##########################################################################


def thresholdviz(model,
                 X,
                 y,
                 colormap=None,
                 color=None,
                 n_trials=50,
                 test_size_percent=0.1,
                 quantiles=(0.1, 0.5, 0.9),
                 random_state=0,
                 title=None,
                 **kwargs):
    """Quick method for ThresholdVisualizer.
    Visualizes precision, recall and queue rate at different different
    thresholds for binary targets.

    Parameters
    ----------

    model : a Scikit-Learn classifier, required
        Should be an instance of a classifier otherwise a will raise a
        YellowbrickTypeError exception on instantiation.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    color : String, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    n_trials : Integer, default: 50
        Number of trials to conduct via train_test_split

    quantiles : Sequence, default: (0.1, 0.5, .9)
        Setting the quantiles for visualizing model variability using
        scipy.stats.mstats.mquantiles

    random_state : Integer, default: 0
        Random state integer for sampling in train_test_split

    title : String, default: None
        Title of the visualization

    kwargs : keyword arguments passed to the super class.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = ThresholdVisualizer(
        model,
        colormap=colormap,
        color=color,
        n_trials=n_trials,
        test_size_percent=test_size_percent,
        quantiles=quantiles,
        random_state=random_state,
        title=title,
        **kwargs)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit_draw_poof(X, y)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
# Static ThresholdVisualizer Visualizer
##########################################################################


class ThresholdVisualizer(ModelVisualizer):
    """Visualizes precision, recall and queue rate at different different
    thresholds for binary targets.

    Parameters
    ----------

    model : a Scikit-Learn classifier, required
        Should be an instance of a classifier otherwise a will raise a
        YellowbrickTypeError exception on instantiation.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    color : String, default: None
        Optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    n_trials : Integer, default: 50
        Number of trials to conduct via train_test_split

    quantiles : Sequence, default: (0.1, 0.5, .9)
        Setting the quantiles for visualizing model variability using
        scipy.stats.mstats.mquantiles

    random_state : Integer, default: 0
        Random state integer for sampling in train_test_split

    title : String, default: None
        Title of the visualization

    kwargs : keyword arguments passed to the super class.
    """

    def __init__(self,
                 model,
                 colormap=None,
                 color=None,
                 n_trials=50,
                 test_size_percent=0.1,
                 quantiles=(0.1, 0.5, 0.9),
                 random_state=0,
                 title=None,
                 **kwargs):
        # Check to see if model is an instance of a classifier.
        # Should return an error if it isn't.
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
            )
        super(ThresholdVisualizer, self).__init__(model, **kwargs)

        self.estimator = model
        self.colormap = kwargs.pop('colormap', None)
        self.color = kwargs.pop('color', None)
        self.title = title
        self.n_trials = n_trials
        self.test_size_percent = test_size_percent
        self.quantiles = quantiles
        self.random_state = np.random.RandomState(random_state)

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

        for trial in range(self.n_trials):
            train_X, test_X, train_y, test_y = train_test_split(
                X,
                y,
                test_size=self.test_size_percent,
                random_state=self.random_state)
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

        return self

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
        # set the colors
        if self.colormap is not None or self.color is not None:
            color_values = resolve_colors(
                num_colors=3, colormap=self.colormap, color=self.color)
        else:
            color_values = get_color_cycle()

        # unpack first three colors for visualization
        recall_color, precision_color, queue_rate_color = color_values[:3]

        if self.ax is None:
            self.ax = plt.gca(xlim=[-1, 1], ylim=[-1, 1])

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

        lower_precision, median_precision, upper_precision = mstats.mquantiles(
            uniform_precision_plots, prob=self.quantiles, axis=0)
        lower_recall, median_recall, upper_recall = mstats.mquantiles(
            uniform_recall_plots, prob=self.quantiles, axis=0)
        lower_queue_rate, median_queue_rate, upper_queue_rate = mstats.mquantiles(
            uniform_queue_rate_plots, prob=self.quantiles, axis=0)

        self.ax.plot(
            uniform_thresholds, median_precision, color=precision_color)
        self.ax.plot(uniform_thresholds, median_recall, color=recall_color)
        self.ax.plot(
            uniform_thresholds, median_queue_rate, color=queue_rate_color)

        self.ax.fill_between(
            uniform_thresholds,
            upper_precision,
            lower_precision,
            alpha=0.5,
            linewidth=0,
            color=precision_color)
        self.ax.fill_between(
            uniform_thresholds,
            upper_recall,
            lower_recall,
            alpha=0.5,
            linewidth=0,
            color=recall_color)
        self.ax.fill_between(
            uniform_thresholds,
            upper_queue_rate,
            lower_queue_rate,
            alpha=0.5,
            linewidth=0,
            color=queue_rate_color)

        self.ax.axis('auto')

        return self.ax

    def finalize(self, **kwargs):
        """Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title
        if self.title is not None:
            self.set_title(self.title)
        leg = self.ax.legend(
            ('precision', 'recall', 'queue_rate'), frameon=True, loc='best')
        self.ax.set_xlabel('threshold')
        self.ax.set_ylabel('percent')

    def fit_draw_poof(self, X, y=None, **kwargs):
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
        self.draw()
        self.poof()
        return self

ThreshViz = ThresholdVisualizer
