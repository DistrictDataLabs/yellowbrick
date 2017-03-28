# yellowbrick.neighbors.knn
# KNN neighbors classifier visualizers for Yellowbrick.
#
# Author:   Nathan Danielsen <rbilbro@gmail.com.com>
# Created:  Sat Mar 12 14:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import OrderedDict


from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.base import ModelVisualizer, Visualizer
from yellowbrick.mixins import BivariateFeatureMixin
from yellowbrick.utils import get_model_name, isestimator, isclassifier
from yellowbrick.style.palettes import color_sequence, color_palette, LINE_COLOR
from yellowbrick.style.colors import resolve_colors, get_color_cycle
from matplotlib.colors import ListedColormap
from yellowbrick.utils import is_dataframe
from yellowbrick.style.palettes import PALETTES


class KnnDecisionBoundariesVisualizer(ModelVisualizer, BivariateFeatureMixin):
    """
    KnnDecisionBoundariesVisualizer is a bivariate data visualization algorithm that plots
    the decision boundaries of each class.
    """
    def __init__(self, model, colors=None, classes=None, features=None, **kwargs):
        """
        Pass in a fitted neighbors model to generate decision boundaries.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param model: the Scikit-Learn estimator
            Should be an instance of a classifier, else the __init__ will
            return an error.

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(KnnDecisionBoundariesVisualizer, self).__init__(self)

        self.colors = kwargs.pop('colors', PALETTES['paired'])
        self.classes_ = classes
        self.features_ = features
        self.estimator = model
        self.name = get_model_name(self.estimator)

        # these are set later
        self.Z = None
        self.xx = None
        self.yy = None

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the parallel coords
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
            Returns the instance of the transformer/visualizer
        """

        nrows, ncols = X.shape

        # Store the classes for the legend if they're None.
        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in set(y)]

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

        h = .02
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.estimator.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        self.Z_shape = Z.reshape(self.xx.shape)

        return self



    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates a scatter plot that draws
        each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # Get the shape of the data
        nrows, ncols = X.shape

        # Create the axes if they don't exist
        if self.ax is None:
            self.ax = plt.gca(xlim=[-1,1], ylim=[-1,1])

        color_cycle = cycler(color=self.colors)

        iter_color_cycle = iter(color_cycle)
        point_color = OrderedDict()
        boundary_color = OrderedDict()
        for class_ in self.classes_:
            point_color[class_] = next(iter_color_cycle)['color']
            boundary_color[class_] = next(iter_color_cycle)['color']

        self.ax.pcolormesh(self.xx, self.yy, self.Z_shape, cmap=ListedColormap(boundary_color.values()) )

        # Create a data structure to hold the scatter plot representations
        to_plot = {}
        for kls in self.classes_:
            to_plot[kls] = [[], []]

        # Add each row of the data set to to_plot for plotting
        # TODO: make this an independent function for override
        for i, row in enumerate(X):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            x_, y_   = row_[0], row_[1]
            kls = self.classes_[y[i]]

            to_plot[kls][0].append(x_)
            to_plot[kls][1].append(y_)

        # Add the scatter plots from the to_plot function
        # TODO: store these plots to add more instances to later
        # TODO: make this a separate function
        for i, kls in enumerate(self.classes_):
            self.ax.scatter(to_plot[kls][0], to_plot[kls][1], color=point_color[kls], label=str(kls), **kwargs)

        self.ax.axis('equal')

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Divide out the two features
        feature_one, feature_two = self.features_

        # Set the title
        self.set_title(
            title='Decisions Boundaries: {feature_one} vs {feature_two}'.format(**locals())
        )
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
