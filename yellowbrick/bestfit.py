# yellowbrick.bestfit
# Uses Scikit-Learn to compute a best fit function, then draws it in the plot.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun Jun 26 17:27:08 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: bestfit.py [56236f3] benjamin@bengfort.com $

"""
Uses Scikit-Learn to compute a best fit function, then draws it in the plot.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse

from operator import itemgetter
from yellowbrick.exceptions import YellowbrickValueError


##########################################################################
## Module Constants
##########################################################################

# Names of the various estimator functions
LINEAR      = 'linear'
QUADRATIC   = 'quadratic'
EXPONENTIAL = 'exponential'
LOG         = 'log'
SELECT_BEST = 'select_best'


##########################################################################
## Draw Line of Best Fit
##########################################################################

def draw_best_fit(X, y, ax, estimator='linear', **kwargs):
    """
    Uses Scikit-Learn to fit a model to X and y then uses the resulting model
    to predict the curve based on the X values. This curve is drawn to the ax
    (matplotlib axis) which must be passed as the third variable.

    The estimator function can be one of the following:

        'linear':      Uses OLS to fit the regression
        'quadratic':   Uses OLS with Polynomial order 2
        'exponential': Not implemented yet
        'log':         Not implemented yet
        'select_best': Selects the best fit via MSE

    The remaining keyword arguments are passed to ax.plot to define and
    describe the line of best fit.
    """

    # Estimators are the types of best fit lines that can be drawn.
    estimators = {
        LINEAR: fit_linear,               # Uses OLS to fit the regression
        QUADRATIC: fit_quadratic,         # Uses OLS with Polynomial order 2
        EXPONENTIAL: fit_exponential,     # Not implemented yet
        LOG: fit_log,                     # Not implemented yet
        SELECT_BEST: fit_select_best,     # Selects the best fit via MSE
    }

    # Check to make sure that a correct estimator value was passed in.
    if estimator not in estimators:
        raise YellowbrickValueError(
            "'{}' not a valid type of estimator; choose from {}".format(
                estimator, ", ".join(estimators.keys())
            )
        )

    # Then collect the estimator function from the mapping.
    estimator = estimators[estimator]

    # Ensure that X and y are the same length
    if len(X) != len(y):
        raise YellowbrickValueError((
            "X and y must have same length:"
             " X len {} doesn't match y len {}!"
        ).format(len(X), len(y)))

    # Ensure that X and y are np.arrays
    X = np.array(X)
    y = np.array(y)

    # Verify that X is a two dimensional array for Scikit-Learn esitmators
    # and that its dimensions are (n, 1) where n is the number of rows.
    if X.ndim < 2:
        X = X[:,np.newaxis] # Reshape X into the correct dimensions

    if X.ndim > 2:
        raise YellowbrickValueError(
            "X must be a (1,) or (n,1) dimensional array not {}".format(x.shape)
        )

    # Verify that y is a (n,) dimensional array
    if y.ndim > 1:
        raise YellowbrickValueError(
            "y must be a (1,) dimensional array not {}".format(y.shape)
        )

    # Uses the estimator to fit the data and get the model back.
    model = estimator(X, y)

    # Plot line of best fit onto the axes that were passed in.
    # TODO: determin if xlim or X.min(), X.max() are better params
    xr = np.linspace(*ax.get_xlim(), num=100)
    ax.plot(xr, model.predict(xr[:,np.newaxis]), **kwargs)
    return ax


##########################################################################
## Estimator Functions
##########################################################################

def fit_select_best(X, y):
    """
    Selects the best fit of the estimators already implemented by choosing the
    model with the smallest mean square error metric for the trained values.
    """
    models = [fit(X,y) for fit in [fit_linear, fit_quadratic]]
    errors = map(lambda model: mse(y, model.predict(X)), models)

    return min(zip(models, errors), key=itemgetter(1))[0]


def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model


def fit_quadratic(X, y):
    """
    Uses OLS with Polynomial order 2.
    """
    model = make_pipeline(
        PolynomialFeatures(2), linear_model.LinearRegression()
    )
    model.fit(X, y)
    return model


def fit_exponential(X, y):
    """
    Fits an exponential curve to the data.
    """
    raise NotImplementedError("Exponential best fit lines are not implemented")


def fit_log(X, y):
    """
    Fit a logrithmic curve to the data.
    """
    raise NotImplementedError("Logrithmic best fit lines are not implemented")



if __name__ == '__main__':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    path = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "concrete.xls")
    if not os.path.exists(path):
        raise Exception("Could not find path for testing")

    xkey = 'Fine Aggregate (component 7)(kg in a m^3 mixture)'
    ykey = 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'
    data = pd.read_excel(path)

    fig, axe = plt.subplots()
    axe.scatter(data[xkey], data[ykey])
    draw_best_fit(data[xkey], data[ykey], axe, 'select_best')

    plt.show()
