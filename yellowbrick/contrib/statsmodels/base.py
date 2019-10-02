# yellowbrick.contrib.statsmodels.base
# A basic wrapper for statsmodels that emulates a scikit-learn estimator.
#
# Author:  Ian Ozsvald
# Created: Wed Jan 10 12:47:00 2018 -0500
#
# ID: base.py [d6ebc39] benjamin@bengfort.com $

"""
A basic wrapper for statsmodels that emulates a scikit-learn estimator.
"""

##########################################################################
## Imports
##########################################################################

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator


##########################################################################
## statsmodels Estimator
##########################################################################


class StatsModelsWrapper(BaseEstimator):
    """
    Wrap a statsmodels GLM as a sklearn (fake) BaseEstimator for YellowBrick.

    Examples
    --------
    First import the external libraries and helper utilities:

    >>> import statsmodels.api as sm
    >>> from functools import partial

    Instantiate a partial with the statsmodels API:

    >>> glm_gaussian_partial = partial(sm.GLM, family=sm.families.Gaussian())
    >>> sm_est = StatsModelsWrapper(glm_gaussian_partial)

    Create a Yellowbrick visualizer to visualize prediction error:

    >>> visualizer = PredictionError(sm_est)
    >>> visualizer.fit(X_train, y_train)
    >>> visualizer.score(X_test, y_test)

    For statsmodels usage, calling .summary() etc:

    >>> gaussian_model = glm_gaussian_partial(y_train, X_train)

    Notes
    -----
    .. note:: This wrapper is trivial, options and extra things like weights
        are not currently handled.
    """

    def __init__(self, glm_partial, stated_estimator_type="regressor", scorer=r2_score):

        # YellowBrick checks the attribute to see if it is a
        # regressor/clusterer/classifier
        self._estimator_type = stated_estimator_type

        # assume user passes in a partial which we can instantiate later
        self.glm_partial = glm_partial

        # needs a default scoring function, regression uses r^2 in sklearn
        self.scorer = scorer

    def fit(self, X, y):
        """
        Pretend to be a sklearn estimator, fit is called on creation
        """

        # note that GLM takes endog (y) and then exog (X):
        # this is the reverse of sklearn's methods
        self.glm_model = self.glm_partial(y, X)
        self.glm_results = self.glm_model.fit()
        return self

    def predict(self, X):
        return self.glm_results.predict(X)

    def score(self, X, y):
        return self.scorer(y, self.predict(X))
