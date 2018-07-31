# yellowbrick.classifier.feature_correlation
# Feature correlation to dependent variable visualizer.
#
# Author    Zijie (ZJ) Poh <poh.zijie@gmail.com>
# Created:  Wed Jul 29 15:30:40 2018 -0700
#
# ID: feature_correlation.py [] poh.zijie@gmail.com $

"""
Feature Correlation to Dependent Variable Visualizer.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.target.base import TargetVisualizer
from yellowbrick.utils import is_dataframe
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from scipy.stats import pearsonr


##########################################################################
## Class Feature Correlation
##########################################################################

class FeatureCorrelation(TargetVisualizer):
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

    method : one of {'pearson', 'mutual_info'}, default: 'pearson'
        The method to calculate correlation between features and target.

    classification : boolean, default: False
        This parameter is only used for method = 'mutual_info'.

    labels : list, default: None
        A list of feature names to use. If a DataFrame is passed to fit and
        features is None, feature names are selected as the column names.

    sort : boolean, default: False
        If false, the features are are not sorted in the plot; otherwise
        features are sorted in ascending order of correlation.

    feature_index : ndarray or a list of feature names
        An array of feature index or feature names to include in the plot.
        If a list of feature names is provided, then X must be a DataFrame.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : np.array
        The feature labels

    corr_ : np.array
        Correlation between features and dependent variable.

    Examples
    --------

    >>> viz = FeatureCorrelation()
    >>> viz.fit(X, y)
    >>> viz.poof()
    """

    METHOD_FUNC = {
        'pearson': lambda X, y: [pearsonr(x, y)[0] for x in np.asarray(X).T],
        'mutual_info': lambda X, y: mutual_info_regression(X, y),
        'mutual_info_classif': lambda X, y: mutual_info_classif(X, y)
    }
    METHOD_LABEL = {
        'pearson': 'Pearson Correlation',
        'mutual_info': 'Mutual Information',
        'mutual_info_classif': 'Mutual Information'
    }

    def __init__(self, ax=None, method='pearson', classification=False,
                 labels=None, sort=False, feature_index=None, **kwargs):
        super(FeatureCorrelation, self).__init__(ax=None, **kwargs)

        if method not in self.METHOD_FUNC:
            raise YellowbrickValueError(
                'Method {} not implement; choose from {}'.format(
                    method, ", ".join(self.METHOD_FUNC.keys())
                )
            )

        if classification and method == 'mutual_info':
            method = 'mutual_info_classif'

        # Parameters
        self.set_params(
            method=method,
            labels=labels,
            sort=sort,
            feature_index=feature_index
        )

    def fit(self, X, y):
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
        if self.feature_index and isinstance(self.feature_index[0], str):
            if not is_dataframe(X):
                raise YellowbrickValueError(
                    'Feature index is a list of string '
                    'but X is not a DataFrame'
                )
            else:
                self.feature_index = [
                    i for i in range(X.shape[1])
                    if X.columns[i] in self.feature_index
                ]

        # Calculate Features correlation with target variable
        self.corr_ = np.array(self.METHOD_FUNC[self.method](X, y))

        # Create labels for the feature importances
        # NOTE: this code is duplicated from MultiFeatureVisualizer
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

        # If feature indices are given, plot only the given features
        if self.feature_index:
            self.corr_ = self.corr_[self.feature_index]
            self.features_ = self.features_[self.feature_index]

        # Sort features by correlation
        if self.sort:
            sort_idx = np.argsort(self.corr_)
            self.corr_ = self.corr_[sort_idx]
            self.features_ = self.features_[sort_idx]

        self.draw()
        return self

    def draw(self):
        """
        Draws the feature correlation to dependent variable, called from fit.
        """
        pos = np.arange(self.corr_.shape[0]) + 0.5

        self.ax.barh(pos, self.corr_)

        # Set the labels for the bars
        self.ax.set_yticks(pos)
        self.ax.set_yticklabels(self.features_)

        return self.ax

    def finalize(self):
        """
        Finalize the drawing setting labels and title.
        """
        self.set_title('Features correlation with dependent variable')

        self.ax.set_xlabel(self.METHOD_LABEL[self.method])

        self.ax.grid(False, axis='y')


##########################################################################
## Quick Method
##########################################################################

def feature_correlation(X, y, ax=None, method='pearson', classification=False,
                        labels=None, sort=False, feature_index=None, **kwargs):
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

    method : one of {'pearson', 'mutual_info'}, default: 'pearson'
        The method to calculate correlation between features and target.

    classification : boolean, default: False
        This parameter is only used for method = 'mutual_info'.

    labels : list, default: None
        A list of feature names to use. If a DataFrame is passed to fit and
        features is None, feature names are selected as the column names.

    sort : boolean, default: False
        If false, the features are are not sorted in the plot; otherwise
        features are sorted in ascending order of correlation.

    feature_index : ndarray or a list of feature names
        An array of feature index or feature names to include in the plot.
        If a list of feature names is provided, then X must be a DataFrame.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """

    # Instantiate the visualizer
    viz = FeatureCorrelation(X, y, ax, method, classification, labels, sort,
                             feature_index, **kwargs)

    # Fit and transform the visualizer (calls draw)
    viz.fit(X, y)
    viz.finalize()

    # Return the axes object on the visualizer
    return viz.ax
