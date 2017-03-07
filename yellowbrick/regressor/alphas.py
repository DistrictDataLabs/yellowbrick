# yellowbrick.regressor.alphas
# Implements alpha selection visualizers for regularization
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Mar 06 19:22:07 2017 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: alphas.py [] benjamin@bengfort.com $

"""
Implements alpha selection visualizers for regularization
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from .base import RegressionScoreVisualizer
from ..exceptions import YellowbrickTypeError
from ..exceptions import YellowbrickValueError


## Packages for export
__all__ = [
    "AlphaSelection", "ManualAlphaSelection"
]


##########################################################################
## AlphaSelection Visualizer
##########################################################################

class AlphaSelection(RegressionScoreVisualizer):
    """
    The Alpha Selection Visualizer demonstrates how different values of alpha
    influence model selection during the regularization of linear models.
    Generally speaking, alpha increases the affect of regularization, e.g. if
    alpha is zero there is no regularization and the higher the alpha, the
    more the regularization parameter influences the final model.

    Regularization is designed to penalize model complexity, therefore the
    higher the alpha, the less complex the model, decreasing the error due to
    variance (overfit). Alphas that are too high on the other hand increase
    the error due to bias (underfit). It is important, therefore to choose an
    optimal Alpha such that the error is minimized in both directions.

    To do this, typically you would you use one of the "RegressionCV" models
    in Scikit-Learn. E.g. instead of using the ``Ridge`` (L2) regularizer, you
    can use ``RidgeCV`` and pass a list of alphas, which will be selected
    based on the cross-validation score of each alpha. This visualizer wraps
    a "RegressionCV" model and visualizes the alpha/error curve. Use this
    visualization to detect if the model is responding to regularization, e.g.
    as you increase or decrease alpha, the model responds and error is
    decreased. If the visualization shows a jagged or random plot, then
    potentially the model is not sensitive to that type of regularization and
    another is required (e.g. L1 or ``Lasso`` regularization).

    Parameters
    ----------

    model : a Scikit-Learn regressor
        Should be an instance of a regressor, and specifically one whose name
        ends with "CV" otherwise a will raise a YellowbrickTypeError exception
        on instantiation. To use non-CV regressors see:
        ``ManualAlphaSelection``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.regressor import AlphaSelection
    >>> from sklearn.linear_model import LassoCV
    >>> model = AlphaSelection(LassoCV())
    >>> model.fit(X, y)
    >>> model.poof()

    Notes
    -----

    This class expects an estimator whose name ends with "CV". If you wish to
    use some other estimator, please see the ``ManualAlphaSelection``
    Visualizer for manually iterating through all alphas and selecting the
    best one.

    This Visualizer hoooks into the Scikit-Learn API during ``fit()``. In
    order to pass a fitted model to the Visualizer, call the ``draw()`` method
    directly after instantiating the visualizer with the fitted model.

    Note, each "RegressorCV" module has many different methods for storing
    alphas and error. This visualizer attempts to get them all and is known
    to work for RidgeCV, LassoCV, LassoLarsCV, and ElasticNetCV. If your
    favorite regularization method doesn't work, please submit a bug report.

    For RidgeCV, make sure ``store_cv_values=True``.
    """

    def __init__(self, model, ax=None, **kwargs):

        # Check to make sure this is a "RegressorCV"
        name = model.__class__.__name__
        if not name.endswith("CV"):
            raise YellowbrickTypeError((
                "'{}' is not a CV regularization model;"
                " try ManualAlphaSelection instead."
            ).format(name))

        # Set the store_cv_values parameter on RidgeCV
        if 'store_cv_values' in model.get_params().keys():
            model.set_params(store_cv_values=True)

        # Call super to initialize the class
        super(AlphaSelection, self).__init__(model, ax=ax, **kwargs)

    def fit(self, X, y, **kwargs):
        """
        A simple pass-through method; calls fit on the estimator and then
        draws the alpha-error plot.
        """
        self.estimator.fit(X, y, **kwargs)
        self.draw()
        return self

    def draw(self):
        """
        Draws the alpha plot based on the values on the estimator.
        """
        if self.ax is None:
            self.ax = plt.gca()

        # Search for the correct parameters on the estimator.
        alphas = self._find_alphas_param()
        errors = self._find_errors_param()

        # Plot the alpha against the error
        name = self.name[:-2].lower() # Remove the CV from the label
        self.ax.plot(alphas, errors, label=name)

        # Annotate the selected alpha
        alpha = self.estimator.alpha_
        idx = np.where(alphas == alpha)
        error = errors[idx]

        # Draw the line
        self.ax.annotate(
            "$\\alpha={:0.3f}$".format(alpha), xy=(alpha,error),
            xycoords='data', xytext=(12, -12),
            fontsize=12, textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title as well as the
        X and Y axis labels and adding the legend.
        """
        # Set the title
        self.set_title(
            '{} Alpha Error'.format(self.name)
        )

        # Set the x and y labels
        self.ax.set_xlabel("alpha")
        self.ax.set_ylabel("error (or score)")

        # Set the legend
        self.ax.legend(loc='best')

    def _find_alphas_param(self):
        """
        Searches for the parameter on the estimator that contains the array of
        alphas that was used to produce the error selection. If it cannot find
        the parameter then a YellowbrickValueError is raised.
        """

        for attr in ("alphas", "alphas_", "cv_alphas_"):
            try:
                return getattr(self.estimator, attr)
            except AttributeError:
                continue

        raise YellowbrickValueError(
            "could not find alphas param on {} estimator".format(
                self.estimator.__class__.__name__
            )
        )

    def _find_errors_param(self):
        """
        Searches for the parameter on the estimator that contains the array of
        errors that was used to determine the optimal alpha. If it cannot find
        the parameter then a YellowbrickValueError is raised.
        """

        if hasattr(self.estimator, 'cv_values_'):
            return self.estimator.cv_values_.mean(0)

        for attr in ('mse_path_', 'cv_mse_path_'):
            if hasattr(self.estimator, attr):
                return getattr(self.estimator, attr).mean(1)

        raise YellowbrickValueError(
            "could not find errors param on {} estimator".format(
                self.estimator.__class__.__name__
            )
        )

##########################################################################
## ManualAlphaSelection Visualizer
##########################################################################

class ManualAlphaSelection(AlphaSelection):
    """
    Parameters
    ----------

    model : a Scikit-Learn regressor
        Should be an instance of a regressor, and specifically one whose name
        ends with "CV" otherwise a will raise a YellowbrickTypeError exception
        on instantiation. To use non-CV regressors see:
        ``ManualAlphaSelection``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    alphas : ndarray or Series, default: np.logspace(-10, 2, 200)
        An array of alphas to fit each model with

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
    """
