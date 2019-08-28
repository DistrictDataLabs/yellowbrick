# tests.test_contrib.test_statsmodels.test_base
# Tests for the statsmodels estimator wrapper.
#
# Author:  Ian Ozsvald
# Created: Wed Jan 10 12:47:00 2018 -0500
#
# ID: test_base.py [d6ebc39] benjamin@bengfort.com $

"""
Tests for the statsmodels estimator wrapper.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import functools
import numpy as np

from yellowbrick.contrib.statsmodels import StatsModelsWrapper

try:
    import statsmodels.api as sm
except ImportError:
    sm = None


##########################################################################
## Test Cases
##########################################################################


@pytest.mark.skipif(sm is None, reason="test requires statsmodels")
def test_stats_models_wrapper():
    """
    A trivial test of the StatsModelsWrapper
    """
    X = np.array([[1], [2], [3]])
    y = np.array([1.1, 2, 3])

    glm_gaussian = functools.partial(sm.GLM, family=sm.families.Gaussian())
    sm_est = StatsModelsWrapper(glm_gaussian)

    assert sm_est.fit(X, y) is sm_est, "fit did not return self"
    assert sm_est.predict(X).shape == (3,)
    assert 0.0 <= sm_est.score(X, y) <= 1.0
