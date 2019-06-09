# yellowbrick.regressor.influence
# Visualize the influence and leverage of individual instances on a regression model.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Sun Jun 09 15:21:17 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: influence.py [] benjamin@bengfort.com $

"""
Visualize the influence and leverage of individual instances on a regression model.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import scipy as sp

from yellowbrick.base import Visualizer
from sklearn.linear_model import LinearRegression


##########################################################################
## Cook's Distance
##########################################################################

class CooksDistance(Visualizer):
    """
    Cook's Distance is a measure of how influential an instance is to the computation of
    a regression, e.g. if the instance is removed would the estimated coeficients of the
    underlying model be substantially changed? Because of this, Cook's Distance is
    generally used to detect outliers in standard, OLS regression. In fact, a general
    rule of thumb is that D(i) > 4/n is a good threshold for determining highly
    influential points as outliers.

    This implementation of Cook's Distance assumes Ordinary Least Squares regression,
    and therefore embeds a ``sklearn.linear_model.LinearRegression`` under the hood.
    Distance is computed via the non-whitened leverage of the projection matrix,
    computed inside of ``fit()``. The results of this visualizer are therefore similar
    to, but not as advanced, as a similar computation using statsmodels. Computing the
    influence for other regression models requires leave one out validation and can be
    expensive to compute.

    For more see:
    https://songhuiming.github.io/pages/2016/11/27/linear-regression-in-python-outliers-leverage-detect/

    Note that Cook's Distance is very similar to DFFITS, and that the Cook's Distance
    scores can be easily transformed to DFFITS scores.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    influence_threshold : bool, default: True
        Draw a horizontal line at D(i) == 4/n to easily identify the most influential
        points on the final regression.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence the final
        visualization (e.g. size or title parameters).
    """

    def __init__(self, ax=None, influence_threshold=True, **kwargs):
        super(CooksDistance, self).__init__(ax=ax, **kwargs)
        self.influence_threshold = influence_threshold

    def fit(self, X, y):
        # Fit a linear model to X and y to compute MSE
        self.model = LinearRegression().fit(X, y)

        # Leverage is computed as the diagonal of the projection matrix of X
        # TODO: whiten X before computing leverage
        leverage = (X * np.linalg.pinv(X).T).sum(1)

        # Compute the rank and the degrees of freedom of the OLS model
        rank = np.linalg.matrix_rank(X)
        df = X.shape[0] - rank

        # Compute the MSE from the residuals
        residuals = y - self.model.predict(X)
        mse = np.dot(residuals, residuals) / df

        # Compute Cook's distance
        residuals_studentized = residuals / np.sqrt(mse) / np.sqrt(1-leverage)
        self.distance_ = residuals_studentized**2 / X.shape[1]
        self.distance_ *= leverage / (1-leverage)

        # Compute the p-values of Cook's Distance
        self.p_values_ = sp.stats.f.sf(self.distance_, X.shape[1], df)

        self.draw()
        return self

    def draw(self):
        # Draw a stem plot with the influence for each instance
        self.ax.stem(self.distance_, markerfmt=",")

        # Draw the threshold for most influential points
        if self.influence_threshold:
            self.ax.axhline(4/len(self.distance_), c='r', ls='--', lw=1)

    def finalize(self):
        self.ax.set_xlabel("instance")
        self.ax.set_ylabel("influence")
        self.ax.set_title("Cook's Distance Outlier Detection")


def cooks_distance(X, y, ax=None, influence_threshold=True, **kwargs):
    """
    Cook's Distance is a measure of how influential an instance is to the computation of
    a regression, e.g. if the instance is removed would the estimated coeficients of the
    underlying model be substantially changed? Because of this, Cook's Distance is
    generally used to detect outliers in standard, OLS regression. In fact, a general
    rule of thumb is that D(i) > 4/n is a good threshold for determining highly
    influential points as outliers.

    This implementation of Cook's Distance assumes Ordinary Least Squares regression,
    and therefore embeds a ``sklearn.linear_model.LinearRegression`` under the hood.
    Distance is computed via the non-whitened leverage of the projection matrix,
    computed inside of ``fit()``. The results of this visualizer are therefore similar
    to, but not as advanced, as a similar computation using statsmodels. Computing the
    influence for other regression models requires leave one out validation and can be
    expensive to compute.

    For more see:
    https://songhuiming.github.io/pages/2016/11/27/linear-regression-in-python-outliers-leverage-detect/

    Note that Cook's Distance is very similar to DFFITS, and that the Cook's Distance
    scores can be easily transformed to DFFITS scores.

    Parameters
    ----------
    X : array-like, 2D
        The exogenous design matrix, e.g. training data.

    y : array-like, 1D
        The endogenous response variable, e.g. target data.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    influence_threshold : bool, default: True
        Draw a horizontal line at D(i) == 4/n to easily identify the most influential
        points on the final regression.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence the final
        visualization (e.g. size or title parameters).
    """
    viz = CooksDistance(ax=ax, influence_threshold=influence_threshold, **kwargs)
    viz.fit(X, y)
    return viz