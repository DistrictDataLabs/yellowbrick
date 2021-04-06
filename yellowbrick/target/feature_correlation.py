# yellowbrick.classifier.feature_correlation
# Feature correlation to dependent variable visualizer.
#
# Author    Zijie (ZJ) Poh
# Created:  Wed Jul 29 15:30:40 2018 -0700
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: feature_correlation.py [33aec16] 8103276+zjpoh@users.noreply.github.com $

"""
Feature Correlation to Dependent Variable Visualizer.
"""

##########################################################################
# Imports
##########################################################################

import numpy as np

from yellowbrick.utils import is_dataframe
from yellowbrick.target.base import TargetVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import pearsonr

##########################################################################
# Supported Correlation Computations
##########################################################################

CORRELATION_LABELS = {
    "pearson": "Pearson Correlation",
    "mutual_info-regression": "Mutual Information",
    "mutual_info-classification": "Mutual Information",
}

CORRELATION_METHODS = {
    "mutual_info-regression": mutual_info_regression,
    "mutual_info-classification": mutual_info_classif,
}

##########################################################################
# Class Feature Correlation
##########################################################################


class FeatureCorrelation(TargetVisualizer):
    """
    Displays the correlation between features and dependent variables.

    This visualizer can be used side-by-side with
    ``yellowbrick.features.JointPlotVisualizer`` that plots a feature
    against the target and shows the distribution of each via a
    histogram on each axis.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    method : str, default: 'pearson'
        The method to calculate correlation between features and target.
        Options include:

            - 'pearson', which uses ``scipy.stats.pearsonr``
            - 'mutual_info-regression', which uses ``mutual_info-regression``
              from ``sklearn.feature_selection``
            - 'mutual_info-classification', which uses ``mutual_info_classif``
              from ``sklearn.feature_selection``

    labels : list, default: None
        A list of feature names to use. If a DataFrame is passed to fit and
        features is None, feature names are selected as the column names.

    sort : boolean, default: False
        If false, the features are are not sorted in the plot; otherwise
        features are sorted in ascending order of correlation.

    feature_index : list,
        A list of feature index to include in the plot.

    feature_names : list of feature names
        A list of feature names to include in the plot.
        Must have labels or the fitted data is a DataFrame with column names.
        If feature_index is provided, feature_names will be ignored.

    color: string
        Specify color for barchart

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : np.array
        The feature labels

    scores_ : np.array
        Correlation between features and dependent variable.

    Examples
    --------

    >>> viz = FeatureCorrelation()
    >>> viz.fit(X, y)
    >>> viz.show()
    """

    def __init__(
        self,
        ax=None,
        method="pearson",
        labels=None,
        sort=False,
        feature_index=None,
        feature_names=None,
        color=None,
        **kwargs
    ):
        super(FeatureCorrelation, self).__init__(ax, **kwargs)

        self.correlation_labels = CORRELATION_LABELS
        self.correlation_methods = CORRELATION_METHODS

        if method not in self.correlation_labels:
            raise YellowbrickValueError(
                "Method {} not implement; choose from {}".format(
                    method, ", ".join(self.correlation_labels)
                )
            )

        # Parameters
        self.sort = sort
        self.color = color
        self.method = method
        self.labels = labels
        self.feature_index = feature_index
        self.feature_names = feature_names

    def fit(self, X, y, **kwargs):
        """
        Fits the estimator to calculate feature correlation to
        dependent variable.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Keyword arguments passed to the fit method of the estimator.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        self._create_labels_for_features(X)

        self._select_features_to_plot(X)

        # Calculate Features correlation with target variable
        if self.method == "pearson":
            self.scores_ = np.array(
                [pearsonr(x, y, **kwargs)[0] for x in np.asarray(X).T]
            )
        else:
            self.scores_ = np.array(
                self.correlation_methods[self.method](X, y, **kwargs)
            )

        # If feature indices are given, plot only the given features
        if self.feature_index:
            self.scores_ = self.scores_[self.feature_index]
            self.features_ = self.features_[self.feature_index]

        # Sort features by correlation
        if self.sort:
            sort_idx = np.argsort(self.scores_)
            self.scores_ = self.scores_[sort_idx]
            self.features_ = self.features_[sort_idx]

        self.draw()
        return self

    def draw(self):
        """
        Draws the feature correlation to dependent variable, called from fit.
        """
        pos = np.arange(self.scores_.shape[0]) + 0.5

        self.ax.barh(pos, self.scores_, color=self.color)

        # Set the labels for the bars
        self.ax.set_yticks(pos)
        self.ax.set_yticklabels(self.features_)

        return self.ax

    def finalize(self):
        """
        Finalize the drawing setting labels and title.
        """
        self.set_title("Features correlation with dependent variable")

        self.ax.set_xlabel(self.correlation_labels[self.method])

        self.ax.grid(False, axis="y")

    def _create_labels_for_features(self, X):
        """
        Create labels for the features

        NOTE: this code is duplicated from MultiFeatureVisualizer
        """
        if self.labels is None:
            # Use column names if a dataframe
            if is_dataframe(X):
                self.features_ = np.array(X.columns)
            # Otherwise use the column index as the labels
            else:
                _, ncols = X.shape
                self.features_ = np.arange(0, ncols)
        else:
            self.features_ = np.array(self.labels)

    def _select_features_to_plot(self, X):
        """
        Select features to plot.

        feature_index is always used as the filter and
        if filter_names is supplied, a new feature_index
        is computed from those names.
        """
        if self.feature_index:
            if self.feature_names:
                raise YellowbrickWarning(
                    "Both feature_index and feature_names "
                    "are specified. feature_names is ignored"
                )
            if min(self.feature_index) < 0 or max(self.feature_index) >= X.shape[1]:
                raise YellowbrickValueError("Feature index is out of range")
        elif self.feature_names:
            self.feature_index = []
            features_list = self.features_.tolist()
            for feature_name in self.feature_names:
                try:
                    self.feature_index.append(features_list.index(feature_name))
                except ValueError:
                    raise YellowbrickValueError("{} not in labels".format(feature_name))


##########################################################################
# Quick Method
##########################################################################


def feature_correlation(
    X,
    y,
    ax=None,
    method="pearson",
    labels=None,
    sort=False,
    feature_index=None,
    feature_names=None,
    color=None,
    show=True,
    **kwargs
):
    """
    Displays the correlation between features and dependent variables.

    This visualizer can be used side-by-side with
    yellowbrick.features.JointPlotVisualizer that plots a feature
    against the target and shows the distribution of each via a
    histogram on each axis.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    method : str, default: 'pearson'
        The method to calculate correlation between features and target.
        Options include:

            - 'pearson', which uses ``scipy.stats.pearsonr``
            - 'mutual_info-regression', which uses ``mutual_info-regression``
              from ``sklearn.feature_selection``
            - 'mutual_info-classification', which uses ``mutual_info_classif``
              from ``sklearn.feature_selection``

    labels : list, default: None
        A list of feature names to use. If a DataFrame is passed to fit and
        features is None, feature names are selected as the column names.

    sort : boolean, default: False
        If false, the features are are not sorted in the plot; otherwise
        features are sorted in ascending order of correlation.

    feature_index : list,
        A list of feature index to include in the plot.

    feature_names : list of feature names
        A list of feature names to include in the plot.
        Must have labels or the fitted data is a DataFrame with column names.
        If feature_index is provided, feature_names will be ignored.

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
    visualizer : FeatureCorrelation
        Returns the fitted visualizer.
    """

    # Instantiate the visualizer
    visualizer = FeatureCorrelation(
        ax=ax,
        method=method,
        labels=labels,
        sort=sort,
        color=color,
        feature_index=feature_index,
        feature_names=feature_names,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer
    return visualizer
