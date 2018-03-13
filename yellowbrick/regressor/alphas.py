# yellowbrick.regressor.alphas
# Implements alpha selection visualizers for regularization
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Mar 06 19:22:07 2017 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: alphas.py [7d3f5e6] benjamin@bengfort.com $

"""
Implements alpha selection visualizers for regularization
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from functools import partial

from .base import RegressionScoreVisualizer
from ..exceptions import YellowbrickTypeError
from ..exceptions import YellowbrickValueError

from sklearn.model_selection import cross_val_score

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
        # Search for the correct parameters on the estimator.
        alphas = self._find_alphas_param()
        errors = self._find_errors_param()


        alpha = self.estimator.alpha_ # Get decision from the estimator
        name = self.name[:-2].lower() # Remove the CV from the label

        # Plot the alpha against the error
        self.ax.plot(alphas, errors, label=name)

        # Draw a dashed vline at the alpha
        label = "$\\alpha={:0.3f}$".format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)

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

        # NOTE: The order of the search is very important!
        for attr in ("cv_alphas_", "alphas_", "alphas",):
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

        # NOTE: The order of the search is very important!
        if hasattr(self.estimator, 'mse_path_'):
            return self.estimator.mse_path_.mean(1)

        if hasattr(self.estimator, 'cv_values_'):
            return self.estimator.cv_values_.mean(0)

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
    The ``AlphaSelection`` visualizer requires a "RegressorCV", that is a
    specialized class that performs cross-validated alpha-selection on behalf
    of the model. If the regressor you wish to use doesn't have an associated
    "CV" estimator, or for some reason you would like to specify more control
    over the alpha selection process, then you can use this manual alpha
    selection visualizer, which is essentially a wrapper for
    ``cross_val_score``, fitting a model for each alpha specified.

    Parameters
    ----------

    model : a Scikit-Learn regressor
        Should be an instance of a regressor, and specifically one whose name
        doesn't end with "CV". The regressor must support a call to
        ``set_params(alpha=alpha)`` and be fit multiple times. If the
        regressor name ends with "CV" a ``YellowbrickValueError`` is raised.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    alphas : ndarray or Series, default: np.logspace(-10, 2, 200)
        An array of alphas to fit each model with

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        This argument is passed to the
        ``sklearn.model_selection.cross_val_score`` method to produce the
        cross validated score for each alpha.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

        This argument is passed to the
        ``sklearn.model_selection.cross_val_score`` method to produce the
        cross validated score for each alpha.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.regressor import ManualAlphaSelection
    >>> from sklearn.linear_model import Ridge
    >>> model = ManualAlphaSelection(
    ...     Ridge(), cv=12, scoring='neg_mean_squared_error'
    ... )
    ...
    >>> model.fit(X, y)
    >>> model.poof()

    Notes
    -----

    This class does not take advantage of estimator-specific searching and is
    therefore less optimal and more time consuming than the regular
    "RegressorCV" estimators.
    """

    def __init__(self, model, ax=None, alphas=None,
                 cv=None, scoring=None, **kwargs):

        # Check to make sure this is not a "RegressorCV"
        name = model.__class__.__name__
        if name.endswith("CV"):
            raise YellowbrickTypeError((
                "'{}' is a CV regularization model;"
                " try AlphaSelection instead."
            ).format(name))

        # Call super to initialize the class
        super(ManualAlphaSelection, self).__init__(model, ax=ax, **kwargs)

        # Set manual alpha selection parameters
        self.alphas = alphas or np.logspace(-10, -2, 200)
        self.errors = None
        self.score_method = partial(cross_val_score, cv=cv, scoring=scoring)

    def fit(self, X, y, **args):
        """
        The fit method is the primary entry point for the manual alpha
        selection visualizer. It sets the alpha param for each alpha in the
        alphas list on the wrapped estimator, then scores the model using the
        passed in X and y data set. Those scores are then aggregated and
        drawn using matplotlib.
        """
        self.errors = []
        for alpha in self.alphas:
            self.estimator.set_params(alpha=alpha)
            scores = self.score_method(self.estimator, X, y)
            self.errors.append(scores.mean())

        # Convert errors to an ND array and draw
        self.errors = np.array(self.errors)
        self.draw()

        # Always make sure to return self from fit
        return self

    def draw(self):
        """
        Draws the alphas values against their associated error in a similar
        fashion to the AlphaSelection visualizer.
        """
        # Plot the alpha against the error
        self.ax.plot(self.alphas, self.errors, label=self.name.lower())

        # Draw a dashed vline at the alpha with maximal error
        alpha = self.alphas[np.where(self.errors == self.errors.max())][0]
        label = "$\\alpha_{{max}}={:0.3f}$".format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)

        # Draw a dashed vline at the alpha with minimal error
        alpha = self.alphas[np.where(self.errors == self.errors.min())][0]
        label = "$\\alpha_{{min}}={:0.3f}$".format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)

        return self.ax
