# yellowbrick.contrib.classifier.boundaries
# Decision boundaries classifier visualizer for Yellowbrick.
#
# Author:   Nathan Danielsen
# Created:  Sat Mar 12 14:17:29 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: boundaries.py [a60bc41] nathan.danielsen@gmail.com $

import itertools
import numpy as np

from collections import OrderedDict

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.utils import is_dataframe, is_structured_array
from yellowbrick.utils import has_ndarray_int_columns


##########################################################################
# Quick Methods
##########################################################################


def decisionviz(
    estimator,
    X,
    y,
    ax=None,
    x_name=None,
    y_name=None,
    features=None,
    classes=None,
    show_scatter=True,
    step_size=0.0025,
    markers=None,
    pcolormesh_alpha=0.8,
    scatter_alpha=1.0,
    encoder=None,
    is_fitted="auto",
    force_model=False,
    **kwargs
):
    """DecisionBoundariesVisualizer is a bivariate data visualization algorithm
        that plots the decision boundaries of each class.

    This helper function is a quick wrapper to utilize the
    DecisionBoundariesVisualizers for one-off analysis.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    x_name : string, default: None
        The feature name that corresponds to a column name or index postion
        in the matrix that will be plotted against the x-axis

    y_name : string, default: None
        The feature name that corresponds to a column name or index postion
        in the matrix that will be plotted against the y-axis

    features : list of strings, default: None
        The names of the features or columns

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    show_scatter : boolean, default: True
        If boolean is True, then a scatter plot with points will be drawn
        on top of the decision boundary graph

    step_size : float percentage, default: 0.0025
        Determines the step size for creating the numpy meshgrid that will
        later become the foundation of the decision boundary graph. The
        default value of 0.0025 means that the step size for constructing
        the meshgrid will be 0.25%% of differenes of the max and min of x
        and y for each feature.

    markers : iterable of strings, default: ,od*vh+
        Matplotlib style markers for points on the scatter plot points

    pcolormesh_alpha : float, default: 0.8
        Sets the alpha transparency for the meshgrid of model boundaries

    scatter_alpha : float, default: 1.0
        Sets the alpha transparency for the scatter plot points

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Returns
    -------
    viz : DecisionBoundariesVisualizer
        Returns the fitted and finalized visualizer
    """
    # Instantiate the visualizer
    visualizer = DecisionBoundariesVisualizer(
        estimator,
        ax=ax,
        x=x_name,
        y=y_name,
        features=features,
        classes=classes,
        show_scatter=show_scatter,
        step_size=step_size,
        markers=markers,
        pcolormesh_alpha=pcolormesh_alpha,
        scatter_alpha=scatter_alpha,
        encoder=encoder,
        is_fitted=is_fitted,
        force_model=force_model,
        **kwargs
    )

    # Fit, draw and finalize the visualizer
    visualizer.fit(X, y)
    visualizer.finalize()

    # Return the axes object on the visualizer
    return visualizer


##########################################################################
# Static ScatterVisualizer Visualizer
##########################################################################


class DecisionBoundariesVisualizer(ClassificationScoreVisualizer):
    """
    DecisionBoundariesVisualizer is a bivariate data visualization algorithm
    that plots the decision boundaries of each class.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    x : string, default: None
        The feature name that corresponds to a column name or index postion
        in the matrix that will be plotted against the x-axis

    y : string, default: None
        The feature name that corresponds to a column name or index postion
        in the matrix that will be plotted against the y-axis

    features : list of strings, default: None
        The names of the features or columns

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    show_scatter : boolean, default: True
        If boolean is True, then a scatter plot with points will be drawn
        on top of the decision boundary graph

    step_size : float percentage, default: 0.0025
        Determines the step size for creating the numpy meshgrid that will
        later become the foundation of the decision boundary graph. The
        default value of 0.0025 means that the step size for constructing
        the meshgrid will be 0.25%% of differenes of the max and min of x
        and y for each feature.

    markers : iterable of strings, default: ,od*vh+
        Matplotlib style markers for points on the scatter plot points

    pcolormesh_alpha : float, default: 0.8
        Sets the alpha transparency for the meshgrid of model boundaries

    scatter_alpha : float, default: 1.0
        Sets the alpha transparency for the scatter plot points

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    """

    def __init__(
        self,
        estimator,
        ax=None,
        x=None,
        y=None,
        features=None,
        classes=None,
        show_scatter=True,
        step_size=0.0025,
        markers=None,
        pcolormesh_alpha=0.8,
        scatter_alpha=1.0,
        encoder=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        super(DecisionBoundariesVisualizer, self).__init__(
            estimator,
            ax=ax,
            classes=classes,
            encoder=encoder,
            is_fitted=is_fitted,
            force_model=force_model,
        )

        self.x = x
        self.y = y
        self.features_ = features
        self.estimator = estimator
        self.show_scatter = show_scatter
        self.step_size = step_size
        self.markers = itertools.cycle(
            kwargs.pop("markers", (",", "o", "d", "*", "v", "h", "+"))
        )
        self.pcolormesh_alpha = pcolormesh_alpha
        self.scatter_alpha = scatter_alpha

        # these are set later
        self.Z = None
        self.Z_shape = None
        self.xx = None
        self.yy = None
        self.class_labels = None

        if self.x is not None and self.y is not None and self.features_ is not None:
            raise YellowbrickValueError("Please specify x,y or features, not both.")

        if self.x is not None and self.y is not None and self.features_ is None:
            self.features_ = [self.x, self.y]

        # Ensure with init that features doesn't have more than two features
        if features is not None:
            if len(features) != 2:
                raise YellowbrickValueError(
                    "DecisionBoundariesVisualizer only accepts two features."
                )

    def _select_feature_columns(self, X):
        """ """

        if len(X.shape) == 1:
            X_flat = X.copy().view(np.float64).reshape(len(X), -1)
        else:
            X_flat = X

        _, ncols = X_flat.shape

        if ncols == 2:
            X_two_cols = X
            if self.features_ is None:
                self.features_ = ["Feature One", "Feature Two"]

        # Handle the feature names if they're None.
        elif self.features_ is not None and is_dataframe(X):
            X_two_cols = X[self.features_].values

        # handle numpy named/ structured array
        elif self.features_ is not None and is_structured_array(X):
            X_selected = X[self.features_]
            X_two_cols = X_selected.copy().view(np.float64).reshape(len(X_selected), -1)

        # handle features that are numeric columns in ndarray matrix
        elif self.features_ is not None and has_ndarray_int_columns(self.features_, X):
            f_one, f_two = self.features_
            X_two_cols = X[:, [int(f_one), int(f_two)]]

        else:
            raise YellowbrickValueError(
                """
                ScatterVisualizer only accepts two features, please
                explicitly set these two features in the init kwargs or
                pass a matrix/ dataframe in with only two columns."""
            )

        return X_two_cols

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
        X = self._select_feature_columns(X)
        self.classes_ = self._labels()

        # Assign each class a unique number for drawing
        if self.classes_ is None:
            self.classes_ = {
                label: str(kls_num) for kls_num, label in enumerate(np.unique(y))
            }
            self.class_labels = None
        elif len(set(y)) == len(self.classes_):
            self.classes_ = {
                label: str(kls_num) for kls_num, label in enumerate(self.classes_)
            }
            self.class_labels = dict(zip(set(y), self.classes_))
        else:
            raise YellowbrickTypeError(
                """Number of classes must be the same length of number of
                target y"""
            )

        # ensure that only
        self.estimator.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = (
            X[:, 0].min() - (X[:, 0].min() * 0.01),
            X[:, 0].max() + (X[:, 0].max() * 0.01),
        )
        y_min, y_max = (
            X[:, 1].min() - (X[:, 1].min() * 0.01),
            X[:, 1].max() + (X[:, 1].max() * 0.01),
        )

        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])
        # set the step increment for drawing the boundary graph
        x_step = (x_max - x_min) * self.step_size
        y_step = (y_max - y_min) * self.step_size

        self.xx, self.yy = np.meshgrid(
            np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step)
        )

        # raise Exception(self.yy.ravel().shape)
        Z = self.estimator.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        self.Z_shape = Z.reshape(self.xx.shape)
        return self

    def draw(self, X, y=None, **kwargs):
        """
        Called from the fit method, this method creates a decision boundary
        plot, and if self.scatter is True, it will scatter plot that draws
        each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # ensure that if someone is passing in another X such as X_test, that
        # features will be properly handled
        X = self._select_feature_columns(X)

        color_cycle = iter(
            resolve_colors(colors=self.class_colors_, n_colors=len(self.classes_))
        )
        colors = OrderedDict([(c, next(color_cycle)) for c in self.classes_.keys()])

        self.ax.pcolormesh(
            self.xx,
            self.yy,
            self.Z_shape,
            alpha=self.pcolormesh_alpha,
            cmap=ListedColormap(colors.values()),
        )

        # Create a data structure to hold the scatter plot representations
        to_plot = OrderedDict()
        for index in self.classes_.values():
            to_plot[index] = [[], []]

        # Add each row of the data set to to_plot for plotting
        for i, row in enumerate(X):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            x_, y_ = row_[0], row_[1]
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
                self.ax.scatter(
                    to_plot[index][0],
                    to_plot[index][1],
                    marker=next(self.markers),
                    color=colors[kls],
                    alpha=self.scatter_alpha,
                    s=30,
                    edgecolors="black",
                    label=str(kls),
                    **kwargs
                )
        else:
            labels = [
                Patch(color=colors[kls], label=kls) for kls in self.classes_.keys()
            ]
            self.ax.legend(handles=labels)

    def finalize(self, **kwargs):
        """
        Sets the title and axis labels and adds a legend.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Divide out the two features
        feature_one, feature_two = self.features_

        self.set_title(self.title)
        # Add the legend
        self.ax.legend(loc="best", frameon=True)
        self.ax.set_xlabel(feature_one)
        self.ax.set_ylabel(feature_two)

    def fit_draw(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y then returns
        visualization of features or fitted model.
        """
        self.fit(X, y, **kwargs)
        self.draw(X, y, **kwargs)

    def fit_draw_show(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y then returns
        visualization of features or fitted model.
        Then calls show to finalize.
        """
        self.fit_draw(X, y, **kwargs)
        return self.show(**kwargs)


DecisionViz = DecisionBoundariesVisualizer
