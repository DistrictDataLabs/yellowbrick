# yellowbrick.regressor.base
# Base classes for regressor Visualizers.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Fri Jun 03 10:30:36 2016 -0700
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [7d3f5e6] benjamin@bengfort.com $

"""
Base classes for regressor Visualizers.
"""

##########################################################################
## Imports
##########################################################################

from ..utils import isregressor
from ..base import ScoreVisualizer
from ..exceptions import YellowbrickTypeError


## Packages for export
__all__ = ["RegressionScoreVisualizer"]


##########################################################################
## Regression Visualization Base Object
##########################################################################


class RegressionScoreVisualizer(ScoreVisualizer):
    """Base class for regressor model selection.

    The RegressionScoreVisualizer wraps a regression model to produce a
    visualization when the score method is called, usually to allow the user
    to effectively compare the performance between models.

    The base class provides helper functionality to ensure that regression
    visualizers consistently store the trained score for access post visualization
    and that a correct regressor is passed to the visualizer.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a regressor. If the model is
        not a regressor, an exception is raised.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs: dict
        Keyword arguments passed to the super class.

    Attributes
    ----------
    score_ : float
        An evaluation metric of the regressor on test data produced when
        ``score()`` is called. This metric is between 0 and 1 -- higher scores are
        generally better. For regressors, this score is usually the r2_score, but
        ensure you check the underlying model for more details about the metric.
    """

    def __init__(self, estimator, ax=None, fig=None, force_model=False, **kwargs):
        if not force_model and not isregressor(estimator):
            raise YellowbrickTypeError(
                "This estimator is not a regressor; try a classifier or "
                "clustering score visualizer instead!"
            )

        self.force_model = force_model
        super(RegressionScoreVisualizer, self).__init__(
            estimator, ax=ax, fig=fig, **kwargs
        )

    def score(self, X, y, **kwargs):
        """
        The score function is the hook for visual interaction. Pass in test
        data and the visualizer will create predictions on the data and
        evaluate them with respect to the test values. The evaluation will
        then be passed to draw() and the result of the estimator score will
        be returned.

        Parameters
        ----------
        X : array-like
            X (also X_test) are the dependent variables of test set to predict

        y : array-like
            y (also y_test) is the independent actual variables to score against

        Returns
        -------
        score : float
            The R^2 score of the underlying regressor
        """
        self.score_ = self.estimator.score(X, y)
        return self.score_
