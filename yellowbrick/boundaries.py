# yellowbrick.boundaries
# Decision boundaries classifier visualizer for Yellowbrick.
#
# Author:   Nathan Danielsen <ndanielsen@gmail.com.com>
# Created:  Sat Mar 12 14:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from collections import OrderedDict

from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.base import ModelVisualizer
from yellowbrick.utils import get_model_name
from yellowbrick.style.palettes import color_sequence
from yellowbrick.style.colors import resolve_colors
from matplotlib.colors import ListedColormap
from yellowbrick.utils import is_dataframe
from yellowbrick.style.palettes import PALETTES
import itertools


class DecisionBoundariesVisualizer(ModelVisualizer):
    """
    DecisionBoundariesVisualizer is a bivariate data visualization algorithm that plots
    the decision boundaries of each class.
    """
    def __init__(self, model, colors=None, classes=None, features=None, show_scatter=True, step_size=0.0025, markers=None, **kwargs):
        """
        Pass in a unfitted model to generate a decision boundaries visualization.

        Parameters
        ----------

        :param model: the Scikit-Learn estimator
            Should be an instance of a classifier, else the __init__ will
            return an error.

        :param colors: string or matplotlib cmap
            By default, uses the yellowbrick 'set1' color palette

        :param classes: a list of class names for the legend
            If classes is None and a y value is passed to fit then the classes
            are selected from the target vector.

        :param features: list of strings
            The names of the features or columns

        :param show_scatter: boolean
            If boolean is True, then a scatter plot with points will be drawn
            on top of the decision boundary graph

        :param step_size: float percentage
            Determines the step size for creating the numpy meshgrid that will later
            become the foundation of the decision boundary graph. The default value
            of 0.0025 means that the step size for constructing the meshgrid
            will be 0.25%% of differenes of the max and min of x and y for each
            feature.

        :param markers: iterable of strings
            Matplotlib style markers for points on the scatter plot points

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(DecisionBoundariesVisualizer, self).__init__(self)

        self.colors = kwargs.pop('colors', PALETTES['set1'])
        self.classes_ = classes
        self.features_ = features
        self.estimator = model
        self.name = get_model_name(self.estimator)
        self.show_scatter = show_scatter
        self.step_size = step_size
        self.markers = itertools.cycle(kwargs.pop('markers', (',', '+', 'o', '*', 'v', 'h', 'd') ))

        # these are set later
        self.Z = None
        self.xx = None
        self.yy = None
        self.class_labels = None

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the decision boundaries
        visualization since it has both the X and y data required for the
        viz and the transform method does not.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the visualizer
        """

        nrows, ncols = X.shape

        # Assign each class a unique number for drawing
        if self.classes_ is None:
            self.classes_ = {label:str(kls_num) for kls_num, label in enumerate(set(y))}
            self.class_labels = None
        elif len(set(y)) == len(self.classes_):
            self.classes_ = {label:str(kls_num) for kls_num, label in enumerate(self.classes_)}
            self.class_labels = dict(zip(set(y), self.classes_))
        else:
            raise YellowbrickTypeError("Number of classes must be the same length of number of target y")
        # Handle the feature names if they're None.
        if self.features_ is None:
                # If X is a data frame, get the columns off it.
                if is_dataframe(X):
                    self.features_ = X.columns

                # Otherwise create numeric labels for each column.
                else:
                    self.features_ = [
                        str(cdx) for cdx in range(ncols)
                    ]

        self.estimator.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Create the axes if they don't exist
        if self.ax is None:
            self.ax = plt.gca(xlim=[x_min, x_max ], ylim=[y_min, y_max ])

        # set the step increment for drawing the boundary graph
        x_step = (x_max - x_min) * self.step_size
        y_step = (y_max - y_min) * self.step_size

        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                             np.arange(y_min, y_max, y_step))
        Z = self.estimator.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        self.Z_shape = Z.reshape(self.xx.shape)
        return self

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates a decision boundary plot,
        and if self.scatter is True, it will scatter plot that draws
        each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # Get the shape of the data
        nrows, ncols = X.shape

        num_colors = len(self.classes_) * 2
        color_cycle = iter(resolve_colors(color=self.colors, num_colors=num_colors))
        colors = OrderedDict([(c, next(color_cycle)) for c in self.classes_.keys() ] )

        self.ax.pcolormesh(self.xx, self.yy, self.Z_shape, cmap=ListedColormap(colors.values()) )

        # Create a data structure to hold the scatter plot representations
        to_plot = OrderedDict()
        for index in self.classes_.values():
            to_plot[index] = [[], []]

        # Add each row of the data set to to_plot for plotting
        for i, row in enumerate(X):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            x_, y_   = row_[0], row_[1]
            # look up the y class name if given in init
            if self.class_labels is not None:
                target = self.class_labels[y[i]]
            else:
                target = y[i]
            index = self.classes_[target]
            to_plot[index][0].append(x_)
            to_plot[index][1].append(y_)

        # Add the scatter plots from the to_plot function
        # TODO: store these plots to add more instances to later
        # TODO: make this a separate function

        if self.show_scatter:
            for kls, index in self.classes_.items():
                self.ax.scatter(to_plot[index][0], to_plot[index][1], marker=next(self.markers), color=colors[kls], alpha=.6, s=30, edgecolors='black', label=str(kls), **kwargs)
        else:
            labels = [Patch(color=colors[kls], label=kls) for kls in self.classes_.keys() ]
            self.ax.legend(handles=labels)

        self.ax.axis('auto')

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Divide out the two features
        title = 'Decisions Boundaries'

        feature_one, feature_two = self.features_
        title = 'Decisions Boundaries: {feature_one} vs {feature_two}'.format(**locals())

        self.set_title(title)
        # Add the legend
        self.ax.legend(loc='best')
        self.ax.set_xlabel(feature_one)
        self.ax.set_ylabel(feature_two)


    def fit_draw(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y then returns
        visualization of features or fitted model.
        """
        self.fit(X, y, **kwargs)
        self.draw(X, y, **kwargs)

    def fit_draw_poof(self, X, y=None, **kwargs):
        self.fit_draw(X, y, **kwargs)
        self.poof(**kwargs)
