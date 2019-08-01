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
    """
    Base class for all ScoreVisualizers that evaluate a regression estimator.

    The primary functionality of this class is to perform a check to ensure
    the passed in estimator is a regressor, otherwise it raises a
    ``YellowbrickTypeError``.

    .. todo:: enhance the docstrings here and for score
    """

    def __init__(self, model, ax=None, fig=None, force_model=False, **kwargs):
        if not force_model and not isregressor(model):
            raise YellowbrickTypeError(
                "This estimator is not a regressor; try a classifier or "
                "clustering score visualizer instead!"
            )

        self.force_model = force_model
        super(RegressionScoreVisualizer, self).__init__(model, ax=ax, fig=fig, **kwargs)

    def score(self, X, y, **kwargs):
        """
        The score method is the primary entry point for drawing.

        Returns
        -------
        score : float
            The R^2 score of the underlying regressor
        """
        self.score_ = self.estimator.score(X, y)
        return self.score_
